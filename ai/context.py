"""Batch context the anomaly layer needs besides the values themselves (ADR-0011):
which operation is running, batch age, and which signals' controlling phase is HELD.

Training builds it from TimescaleDB (`tag_attribution`); the harness from engine output;
live from retained `state/*` plus bindings read from the graph once per batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ai.features import PV

PLANT_FILE = Path(__file__).resolve().parents[1] / "config" / "plant.yaml"
TAG_MAP_FILE = Path(__file__).resolve().parents[1] / "edge" / "tag-map.yaml"
SCORED_OPERATIONS = ("Growth", "Production")
SHIFT_SETTLE_MIN = 120.0  # layers 2-3 stay quiet this long after TempShift ends


@dataclass
class Context:
    inoculation_min: float | None = None
    # (operation, start_min, end_min or None)
    operations: list[tuple[str, float, float | None]] = field(default_factory=list)
    # (phase, start_min, end_min or None)
    holds: list[tuple[str, float, float | None]] = field(default_factory=list)

    def operation_at(self, minute: float) -> str | None:
        for name, start, end in reversed(self.operations):
            if start <= minute and (end is None or minute < end):
                return name
        return None

    def inoculated_at(self, minute: float) -> float:
        """Inoculation minute if it had happened by `minute`, else the batch origin (0).
        A window may only use what was known at its end (train/serve parity)."""
        inoc = self.inoculation_min
        return inoc if inoc is not None and inoc <= minute else 0.0

    def age_h(self, minute: float) -> float:
        """Hours since inoculation: the alignment axis. (Not time since the operation
        started: the daily feed bolus is scheduled on batch days, so only batch age keeps
        it in the same bin for every batch.)"""
        return (minute - self.inoculated_at(minute)) / 60.0

    def settling(self, minute: float) -> bool:
        """TempShift and its settling time: no statistical scoring (architecture)."""
        for name, start, end in self.operations:
            if (
                name == "TempShift"
                and start <= minute
                and (end is None or minute < end + SHIFT_SETTLE_MIN)
            ):
                return True
        return False

    def held_phases(self, minute: float) -> set[str]:
        return {
            p for p, start, end in self.holds if start <= minute and (end is None or minute < end)
        }

    def open_operation(self, name: str, minute: float) -> None:
        if self.operations and self.operations[-1][2] is None:
            prev, start, _ = self.operations[-1]
            self.operations[-1] = (prev, start, minute)
        if name != "Idle":
            self.operations.append((name, minute, None))
        if name == "Inoculation" and self.inoculation_min is None:
            self.inoculation_min = minute

    def hold(self, phase: str, minute: float) -> None:
        self.holds.append((phase, minute, None))

    def release(self, phase: str, minute: float) -> None:
        for i, (p, start, end) in enumerate(self.holds):
            if p == phase and end is None:
                self.holds[i] = (p, start, minute)


def controlled_by_config(plant_file: Path = PLANT_FILE) -> dict[str, set[str]]:
    """Phase -> the PV signals it *controls*, from config (the same facts the graph holds).
    Used offline; live, the service reads the bindings from the graph instead."""
    plant = yaml.safe_load(plant_file.read_text())
    tag_map = yaml.safe_load(TAG_MAP_FILE.read_text())["tags"]
    loop_signals: dict[str, set[str]] = {}
    for suffix, spec in tag_map.items():
        if spec["class"] == "pv" and spec["name"] in PV:
            loop_signals.setdefault(suffix.split(".")[0], set()).add(spec["name"])
    ems = plant["equipment_modules"]
    out: dict[str, set[str]] = {}
    for phase, bindings in plant["phase_classes"].items():
        signals: set[str] = set()
        for b in bindings:
            if b["role"] != "control":
                continue
            cms = ems[b["module"]]["control_modules"] if b["module"] in ems else [b["module"]]
            for cm in cms:
                signals |= loop_signals.get(cm, set())
        out[phase] = signals
    return out
