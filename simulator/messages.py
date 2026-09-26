"""What an engine emits: plain dataclasses, no pydantic on the hot path (plan task 1.8).

Every process engine (bioreactor, API, OSD) emits these, so the live runner, backfill
and the harness handle all of them the same way:

- `RawOut`   raw DCS samples every publish period (for edge/raw)
- `LabOut`   LIMS results (lab/*)
- `StateOut` state/batch, state/operation, state/phase/<p> changes
- `EventOut` events/batch, events/operator and events/material
- `LabelOut` ground-truth fault labels (_sim/faults, ADR-0012)
- `TruthOut` true state at each publish tick, only when `record_truth=True` (tests)

`cell` says which unit of a train a message belongs to (ADR-0018). None means the unit
the batch started on, which is the only unit for a bioreactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from common.models import FaultLabel, Operation, Quality

DCS_TAGS_FILE = Path(__file__).with_name("dcs_tags.yaml")


@dataclass(slots=True)
class RawOut:
    tag: str  # full raw tag, e.g. BR101.AIC-102.PV
    value: float
    t: datetime
    q: Quality = Quality.GOOD
    cell: str | None = None


@dataclass(slots=True)
class LabOut:
    name: str
    value: float
    unit: str
    t: datetime
    cell: str | None = None


@dataclass(slots=True)
class StateOut:
    name: str  # "batch", "operation" or "phase/<phase>"
    value: str | None
    t: datetime
    cell: str | None = None


@dataclass(slots=True)
class EventOut:
    name: str  # "batch", "operator" or "material"
    v: BaseModel
    t: datetime
    cell: str | None = None


@dataclass(slots=True)
class LabelOut:
    label: FaultLabel
    t: datetime
    cell: str | None = None


@dataclass(slots=True)
class TruthOut:
    t: datetime
    t_h: float
    state: Any  # the process's true state (a copy)
    operation: Operation
    sp: dict[str, float]
    cell: str | None = None


SimMessage = RawOut | LabOut | StateOut | EventOut | LabelOut | TruthOut


def load_dcs_tags(cls: str = "bioreactor", path: Path = DCS_TAGS_FILE) -> dict[str, str]:
    """Raw tag suffix -> simulator variable for one equipment class,
    e.g. {"AIC-102.PV": "ph"}."""
    return dict(yaml.safe_load(path.read_text())["classes"][cls]["tags"])
