"""Injected faults (architecture: fault table; ADR-0008).

Each fault acts through the same channels a real one would: a sensor offset, a stuck
reading, a degraded actuator or a disturbance. Nothing here touches the published data
directly; the signatures emerge from the process, sensors and control loops.

A fault is active from `onset_h` (hours since batch start) until `end_h`, if set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from common.models import FaultType
from simulator.sensors import SensorBank


@dataclass(slots=True)
class ActuatorMods:
    """What the active faults do to the actuators this step."""

    kla_factor: float = 1.0
    jacket_offset: float = 0.0
    feed_blocked: bool = False
    contaminant_growth: float = 0.0


@dataclass(slots=True)
class Fault:
    kind: FaultType
    onset_h: float
    params: dict[str, float | str] = field(default_factory=dict)
    end_h: float | None = None
    noticed: bool = False  # stuck sensor: did the device-side check notice?

    def active(self, t_h: float) -> bool:
        return t_h >= self.onset_h and (self.end_h is None or t_h < self.end_h)

    def param(self, key: str) -> float:
        value = self.params.get(key, DEFAULTS[self.kind][key])
        return float(value)

    @property
    def stuck_tag(self) -> str:
        return str(self.params.get("tag", "do"))

    def apply(self, t_h: float, sensors: SensorBank, mods: ActuatorMods) -> None:
        """Apply this fault's effect at time `t_h` (called every integration step)."""
        age = t_h - self.onset_h
        match self.kind:
            case FaultType.PH_PROBE_DRIFT:
                sensors.offsets["ph"] = self.param("rate") * age
            case FaultType.DO_SPARGER_FOULING:
                floor, tau = self.param("floor"), self.param("tau_h")
                mods.kla_factor *= floor + (1.0 - floor) * math.exp(-age / tau)
            case FaultType.TEMP_CONTROL_LOSS:
                amp, period = self.param("amplitude"), self.param("period_h")
                mods.jacket_offset += amp * math.sin(2.0 * math.pi * age / period)
            case FaultType.FEED_PUMP_FAILURE:
                mods.feed_blocked = True
            case FaultType.STUCK_SENSOR:
                if self.stuck_tag not in sensors.stuck:
                    sensors.stick(self.stuck_tag)
            case FaultType.CONTAMINATION:
                mods.contaminant_growth = self.param("growth")

    def release(self, sensors: SensorBank) -> None:
        """Undo lasting sensor effects when the fault ends (e.g. a probe is replaced)."""
        if self.kind is FaultType.PH_PROBE_DRIFT:
            sensors.offsets.pop("ph", None)
        elif self.kind is FaultType.STUCK_SENSOR:
            sensors.unstick(self.stuck_tag)


DEFAULTS: dict[FaultType, dict[str, float | str]] = {
    FaultType.PH_PROBE_DRIFT: {"rate": 0.02},  # pH units per hour of probe offset
    FaultType.DO_SPARGER_FOULING: {"floor": 0.04, "tau_h": 8.0},
    FaultType.TEMP_CONTROL_LOSS: {"amplitude": 1.6, "period_h": 2.0},
    FaultType.FEED_PUMP_FAILURE: {},
    FaultType.STUCK_SENSOR: {"tag": "do", "notice_probability": 0.3, "duration_h": 24.0},
    FaultType.CONTAMINATION: {"growth": 0.25},
}


def make(kind: FaultType, onset_h: float, rng: np.random.Generator, **params: float | str) -> Fault:
    """Create a fault; the stuck-sensor 'noticed' draw happens here, once."""
    fault = Fault(kind=FaultType(kind), onset_h=onset_h, params=dict(params))
    duration = params.get("duration_h", DEFAULTS[fault.kind].get("duration_h"))
    if duration is not None:
        fault.end_h = onset_h + float(duration)
    if fault.kind is FaultType.STUCK_SENSOR:
        fault.noticed = bool(rng.random() < fault.param("notice_probability"))
    return fault
