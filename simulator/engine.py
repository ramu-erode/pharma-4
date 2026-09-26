"""The bioreactor simulation engine for one batch on one unit: the shared core (plan
task 1.8).

`BatchRun.advance(until_h)` integrates the process and returns what a plant would emit
in that interval, as the plain dataclasses of `simulator.messages`.

The live runner, backfill and the test harness all drive this class; none of them
reimplements any of it. The API and OSD engines (ADR-0019) follow the same interface.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from common.models import (
    BatchEnded,
    BatchStarted,
    BatchStatus,
    FaultLabel,
    FaultType,
    Levers,
    Operation,
    OperationChanged,
    OperatorEvent,
    PhaseChanged,
    PhaseClass,
    PhaseState,
    Quality,
)
from simulator import batches as b
from simulator import faults as fl
from simulator.control import DoCascade, PhLoop, TemperatureLoop, air_flow, temperature_sp
from simulator.messages import (
    DCS_TAGS_FILE,
    EventOut,
    LabelOut,
    LabOut,
    RawOut,
    SimMessage,
    StateOut,
    TruthOut,
    load_dcs_tags,
)
from simulator.process import Actuators, Params, TrueState, inoculate, step, viability
from simulator.sensors import SensorBank

__all__ = [
    "DCS_TAGS_FILE", "BatchRun", "EventOut", "LabOut", "LabelOut", "RawOut", "SimMessage",
    "StateOut", "TruthOut", "load_dcs_tags",
]  # fmt: skip

LAB_UNITS: dict[str, str] = {
    "vcd": "1e6 cells/mL",
    "viability": "%",
    "glucose": "g/L",
    "lactate": "g/L",
    "ph_offline": "pH",
    "titer": "g/L",
}

HARVEST_TEMP = 20.0
RECALIBRATE_PH = 0.1  # |probe - blood gas| that triggers a probe recalibration
INITIAL_VOLUME = 1500.0


# --- engine ---------------------------------------------------------------------------


@dataclass(slots=True)
class _FaultRun:
    fault: fl.Fault
    label_id: str
    onset_sent: bool = False
    end_sent: bool = False


@dataclass
class BatchRun:
    spec: b.BatchSpec
    params: Params = field(default_factory=Params)
    dt_s: float = 5.0
    publish_period_s: float = 60.0
    record_truth: bool = False
    quiet: bool = False  # emit nothing (ground-truth forward runs)
    tags: dict[str, str] = field(default_factory=load_dcs_tags)

    def __post_init__(self) -> None:
        steps = self.publish_period_s / self.dt_s
        if steps < 1 or abs(steps - round(steps)) > 1e-9:
            raise ValueError("publish period must be a whole multiple of the integration step")
        self.rng = np.random.default_rng(self.spec.seed)
        self.levers: Levers = self.spec.levers
        self.state = TrueState(vol=INITIAL_VOLUME)
        self.temp_loop, self.ph_loop, self.do_loop = TemperatureLoop(), PhLoop(), DoCascade()
        self.sensors = SensorBank()
        self.holds = self.spec.holds if self.spec.holds is not None else b.draw_holds(self.rng)
        # Hold states only change at these instants (or when an operation restarts phases).
        self._hold_edges = sorted({e for h in self.holds for e in (h.start_h, h.end_h)})
        self._holds_dirty = bool(self.holds)
        self.dt_h = self.dt_s / 3600.0
        self.steps_per_publish = round(steps)
        self.k = 0
        self.operation = Operation.IDLE
        self.op_started_h = 0.0
        self.phases: dict[PhaseClass, PhaseState] = {}
        self.status = BatchStatus.RUNNING
        self.done = False
        self.early_harvest = False
        self.next_lab_day = 0
        self.tsp = 36.5
        self.act = Actuators(36.5, 80.0, 0.1, 0.0, 0.0, 0.0, 0.0)
        self._out: list[SimMessage] = []
        self._device = self.spec.cell.replace("-", "")
        self._raw = [(f"{self._device}.{suffix}", var) for suffix, var in self.tags.items()]
        self._faults: list[_FaultRun] = []
        for f in self.spec.faults:
            self._add_fault(f.kind, b.t_of_day(f.onset_day), dict(f.params))
        self._start()

    # -- public API -----------------------------------------------------------------------

    @property
    def t_h(self) -> float:
        return self.k * self.dt_h

    @property
    def batch_day(self) -> float:
        return b.batch_day(self.t_h)

    def t_of_day(self, day: float) -> float:
        """Hours since batch start at batch day `day` (days count from inoculation)."""
        return b.t_of_day(day)

    def ts(self, t_h: float | None = None) -> datetime:
        return self.spec.start + timedelta(
            seconds=round((self.t_h if t_h is None else t_h) * 3600, 3)
        )

    def advance(self, until_h: float) -> list[SimMessage]:
        """Integrate until `until_h` hours since batch start (or the end of the batch)."""
        while not self.done and self.t_h < until_h - 1e-9:
            self._step()
        out, self._out = self._out, []
        return out

    def run_to_end(self) -> list[SimMessage]:
        return self.advance(math.inf)

    def inject(self, kind: FaultType, **params: float | str) -> str:
        """Inject a fault now (live demo). Returns the label id."""
        return self._add_fault(FaultType(kind), self.t_h, params)

    def clear(self, kind: FaultType) -> None:
        for fr in self._faults:
            if fr.fault.kind is kind and fr.fault.active(self.t_h):
                fr.fault.end_h = self.t_h

    def abort(self, reason: str = "operator abort") -> list[SimMessage]:
        if not self.done:
            self._end(BatchStatus.ABORTED, reason)
        out, self._out = self._out, []
        return out

    def change_lever(
        self,
        name: str,
        value: float,
        operator: str,
        recommendation_id: str | None = None,
        reason: str | None = None,
    ) -> list[SimMessage]:
        """An operator changes a recipe parameter mid-batch (ADR-0012). Raises ValueError
        for a lever whose window has passed (the shift day once shifted)."""
        if name not in Levers.model_fields:
            raise ValueError(f"unknown parameter {name!r}")
        if name == "shift_day" and (
            self.operation not in (Operation.SETUP, Operation.INOCULATION, Operation.GROWTH)
            or value <= self.batch_day
        ):
            raise ValueError("the temperature shift can no longer be moved")
        old = float(getattr(self.levers, name))
        self.levers = self.levers.model_copy(update={name: float(value)})
        self._emit(
            EventOut(
                "operator",
                OperatorEvent(
                    operator=operator,
                    parameter=name,
                    old=old,
                    new=float(value),
                    recommendation_id=recommendation_id,
                    reason=reason,
                ),
                self.ts(),
            )
        )
        out, self._out = self._out, []
        return out

    def snapshot(self) -> BatchRun:
        """A deep copy for counterfactual runs (simulator.truth)."""
        return copy.deepcopy(self)

    # -- lifecycle ------------------------------------------------------------------------

    def _start(self) -> None:
        t = self.ts(0.0)
        self._emit(StateOut("batch", self.spec.batch_id, t))
        self._emit(
            EventOut(
                "batch",
                BatchStarted(
                    batch_id=self.spec.batch_id,
                    recipe=self.spec.recipe_id,
                    campaign=self.spec.campaign,
                    planned_levers=self.levers,
                ),
                t,
            )
        )
        self._set_operation(Operation.SETUP)

    def _set_operation(self, new: Operation) -> None:
        t, old = self.ts(), self.operation
        for phase in list(self.phases):
            self._set_phase(phase, PhaseState.COMPLETE, old)
        self.phases.clear()
        for loop in (self.temp_loop, self.ph_loop, self.do_loop):
            loop.pi.held = False
        self._holds_dirty = bool(self.holds)
        self.operation, self.op_started_h = new, self.t_h
        self._emit(StateOut("operation", new.value, t))
        self._emit(
            EventOut(
                "batch",
                OperationChanged(batch_id=self.spec.batch_id, previous=old, current=new),
                t,
            )
        )
        for phase in b.PHASES.get(new, ()):
            self._set_phase(phase, PhaseState.RUNNING, new)

    def _set_phase(self, phase: PhaseClass, state: PhaseState, operation: Operation) -> None:
        if state is PhaseState.COMPLETE:
            self.phases.pop(phase, None)
        else:
            self.phases[phase] = state
        t = self.ts()
        self._emit(StateOut(f"phase/{phase.value.lower()}", state.value, t))
        self._emit(
            EventOut(
                "batch",
                PhaseChanged(
                    batch_id=self.spec.batch_id, operation=operation, phase=phase, state=state
                ),
                t,
            )
        )

    def _end(self, status: BatchStatus, reason: str | None) -> None:
        if status is BatchStatus.ABORTED and self.operation not in (
            Operation.SETUP,
            Operation.IDLE,
        ):
            self._lab(final=True)  # the confirming sample an abort decision rests on
        self.status = status
        self._set_operation(Operation.IDLE)
        for fr in self._faults:
            if fr.onset_sent and not fr.end_sent:
                self._send_label(fr, end=True)
        t = self.ts()
        self._emit(StateOut("batch", None, t))
        self._emit(
            EventOut(
                "batch", BatchEnded(batch_id=self.spec.batch_id, status=status, reason=reason), t
            )
        )
        self.done = True

    def _transition(self) -> None:
        t, op, day = self.t_h, self.operation, self.batch_day
        if op is Operation.SETUP and t >= b.SETUP_H - 1e-9:
            self.state = inoculate(self.state, 0.5)
            self._set_operation(Operation.INOCULATION)
        elif op is Operation.INOCULATION and t >= b.SETUP_H + b.INOCULATION_H - 1e-9:
            self._set_operation(Operation.GROWTH)
        elif op is Operation.GROWTH and day >= self.levers.shift_day:
            self._set_operation(Operation.TEMP_SHIFT)
        elif op is Operation.TEMP_SHIFT and t >= self.op_started_h + b.TEMP_SHIFT_H - 1e-9:
            self._set_operation(Operation.PRODUCTION)
        elif op is Operation.PRODUCTION and (day >= self.spec.harvest_day or self.early_harvest):
            self._set_operation(Operation.HARVEST)
            self._lab(final=True)
        elif op is Operation.HARVEST and t >= self.op_started_h + b.HARVEST_H - 1e-9:
            self._end(
                BatchStatus.EARLY_HARVEST if self.early_harvest else BatchStatus.COMPLETE,
                "viability below 60%" if self.early_harvest else None,
            )

    # -- the integration step -------------------------------------------------------------

    def _step(self) -> None:
        self._transition()
        if self.done:
            return
        t, dt, day = self.t_h, self.dt_h, self.batch_day
        s, lv, sensors, running = self.state, self.levers, self.sensors, self.phases

        mods = fl.ActuatorMods()
        for fr in self._faults:
            f = fr.fault
            if f.active(t):
                if not fr.onset_sent:
                    self._send_label(fr, end=False)
                f.apply(t, sensors, mods)
            elif fr.onset_sent and not fr.end_sent and f.end_h is not None and t >= f.end_h:
                f.release(sensors)
                self._send_label(fr, end=True)

        if self._holds_dirty or (self._hold_edges and t >= self._hold_edges[0]):
            while self._hold_edges and t >= self._hold_edges[0]:
                self._hold_edges.pop(0)
            self._update_holds(t)
            self._holds_dirty = False

        if self.operation is Operation.HARVEST:
            self.tsp = HARVEST_TEMP
        else:
            self.tsp = temperature_sp(day, lv.shift_day, lv.prod_temp)
        jacket = self.temp_loop.update(self.tsp, sensors.control("temp", s.temp), dt)
        jacket += mods.jacket_offset

        if PhaseClass.PH_CTRL in running:
            co2, base = self.ph_loop.update(lv.ph_sp, sensors.control("ph", s.ph), dt)
        else:
            co2, base = 0.0, 0.0
        if PhaseClass.DO_CTRL in running:
            agitation, o2 = self.do_loop.update(lv.do_sp, sensors.control("do", s.do), dt)
        else:
            agitation, o2 = self.do_loop.agit_min, 0.0

        feed = 0.0
        if (
            running.get(PhaseClass.FEED_ADD) is PhaseState.RUNNING
            and not mods.feed_blocked
            and b.FEED_FIRST_DAY <= day < b.FEED_LAST_DAY + 1
            and (day - math.floor(day)) * 24.0 < b.FEED_BOLUS_H
        ):
            feed = b.FEED_BOLUS_FRACTION * INITIAL_VOLUME * lv.feed_mult / b.FEED_BOLUS_H

        self.act = Actuators(
            jacket=jacket,
            agitation=agitation,
            air=air_flow(max(day, 0.0)),
            o2=o2,
            co2=co2,
            base=base,
            feed=feed,
            kla_factor=mods.kla_factor,
            contaminant_growth=mods.contaminant_growth,
        )
        self.state = step(s, self.act, self.params, self.spec.traits, dt)
        self.k += 1

        if self.k % self.steps_per_publish == 0:
            self._publish_tick()
        if (
            self.operation in (Operation.GROWTH, Operation.TEMP_SHIFT, Operation.PRODUCTION)
            and self.t_h >= b.t_of_day(self.next_lab_day) + b.LAB_OFFSET_H - 1e-9
        ):
            self._lab(final=False)

    def _update_holds(self, t: float) -> None:
        for phase in b.HOLDABLE:
            current = self.phases.get(phase)
            if current is None:
                continue
            held = any(h.phase is phase and h.start_h <= t < h.end_h for h in self.holds)
            wanted = PhaseState.HELD if held else PhaseState.RUNNING
            if wanted is not current:
                self._set_phase(phase, wanted, self.operation)
                if phase is PhaseClass.PH_CTRL:
                    self.ph_loop.pi.held = held
                elif phase is PhaseClass.DO_CTRL:
                    self.do_loop.pi.held = held

    # -- outputs --------------------------------------------------------------------------

    def _emit(self, msg: SimMessage) -> None:
        if not self.quiet:
            self._out.append(msg)

    def _publish_tick(self) -> None:
        if self.state.contaminant > b.ABORT_CONTAMINANT:
            self._end(BatchStatus.ABORTED, "contamination")
            return
        if self.quiet:
            return
        t = self.ts()
        uncertain = {
            fr.fault.stuck_tag
            for fr in self._faults
            if fr.fault.kind is FaultType.STUCK_SENSOR
            and fr.fault.noticed
            and fr.fault.active(self.t_h)
        }
        for tag, var in self._raw:
            q = Quality.UNCERTAIN if var in uncertain else Quality.GOOD
            self._out.append(RawOut(tag, self._read(var), t, q))
        if self.record_truth:
            self._out.append(
                TruthOut(
                    t,
                    self.t_h,
                    copy.copy(self.state),
                    self.operation,
                    {"temperature": self.tsp, "ph": self.levers.ph_sp, "do": self.levers.do_sp},
                )
            )

    def _read(self, var: str) -> float:
        s, a, sample, rng = self.state, self.act, self.sensors.sample, self.rng
        match var:
            case "temp":
                return sample("temp", s.temp, rng)
            case "temp_sp":
                return self.tsp
            case "ph":
                return sample("ph", s.ph, rng)
            case "ph_sp":
                return self.levers.ph_sp
            case "do":
                return sample("do", s.do, rng)
            case "do_sp":
                return self.levers.do_sp
            case "agitation" | "air" | "o2" | "co2":
                return sample(var, getattr(a, var), rng)
            case "feed_total" | "base_total":
                return sample(var, getattr(s, var), rng)
            case "pressure":
                return sample("pressure", 70.0 + 4.0 * (a.air + a.o2), rng)
            case "weight":
                return sample("weight", s.vol, rng)
            case "spare_rtd":
                return sample("spare_rtd", 22.0 + 1.5 * math.sin(2 * math.pi * self.t_h / 24), rng)
        raise KeyError(var)

    def _lab(self, final: bool) -> None:
        day = self.next_lab_day
        if not final:
            self.next_lab_day += 1
        s, rng = self.state, self.rng
        via = viability(s)
        ph_offline = s.ph + float(rng.normal(0.0, 0.01))
        self._recalibrate_if_needed(ph_offline)
        if not self.quiet:
            t = self.ts()
            values = {
                "vcd": s.xv * (1.0 + rng.normal(0.0, 0.05)),
                "viability": min(100.0, via + rng.normal(0.0, 1.5)),
                "glucose": s.glc * (1.0 + rng.normal(0.0, 0.03)),
                "lactate": s.lac * (1.0 + rng.normal(0.0, 0.03)),
                "ph_offline": ph_offline,
            }
            if final or day >= b.TITER_FIRST_DAY:
                values["titer"] = s.titer * (1.0 + rng.normal(0.0, 0.03))
            for name, value in values.items():
                self._out.append(LabOut(name, max(0.0, float(value)), LAB_UNITS[name], t))
        if not final and self.operation is Operation.PRODUCTION and via < b.EARLY_HARVEST_VIABILITY:
            self.early_harvest = True

    # -- faults ---------------------------------------------------------------------------

    def _recalibrate_if_needed(self, ph_offline: float) -> None:
        """The daily blood-gas check disagrees with the probe: the operator recalibrates it,
        which ends a probe-drift fault (the realistic way such a fault is found)."""
        measured = self.sensors.control("ph", self.state.ph)
        if abs(measured - ph_offline) <= RECALIBRATE_PH:
            return
        for fr in self._faults:
            f = fr.fault
            if f.kind is FaultType.PH_PROBE_DRIFT and f.active(self.t_h):
                f.end_h = self.t_h

    def _add_fault(self, kind: FaultType, onset_h: float, params: dict[str, float | str]) -> str:
        fault = fl.make(kind, onset_h, self.rng, **params)
        label_id = f"{self.spec.batch_id}-F{len(self._faults) + 1}"
        self._faults.append(_FaultRun(fault, label_id))
        return label_id

    def _send_label(self, fr: _FaultRun, end: bool) -> None:
        f = fr.fault
        if end:
            fr.end_sent = True
        else:
            fr.onset_sent = True
        label = FaultLabel(
            id=fr.label_id,
            fault=f.kind,
            cell=self.spec.cell,
            batch_id=self.spec.batch_id,
            onset=self.ts(f.onset_h),
            end=self.ts(min(self.t_h, f.end_h) if f.end_h is not None else self.t_h)
            if end
            else None,
            params={k: v for k, v in f.params.items()},
        )
        self._emit(LabelOut(label, self.ts()))
