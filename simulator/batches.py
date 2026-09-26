"""Batch definition and the ISA-88 structure of a run (ADR-0011).

Timeline, in hours since batch start (t = 0 is BATCH_START):

    Setup (6 h) -> Inoculation (1 h) -> Growth -> TempShift (6 h) -> Production -> Harvest (6 h)

Batch day counts from inoculation (t = 6 h). Growth ends on the shift day, Production on
the harvest day (14, or earlier on low viability). Transitions are decided while the
batch runs, because an operator can move the shift day and a batch can end early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from common.models import Campaign, FaultType, Levers, Operation, PhaseClass
from simulator.process import Traits

SETUP_H = 6.0
INOCULATION_H = 1.0
TEMP_SHIFT_H = 6.0
HARVEST_H = 6.0
HARVEST_DAY = 14.0
FEED_FIRST_DAY = 3
FEED_LAST_DAY = 13
FEED_BOLUS_H = 0.5
FEED_BOLUS_FRACTION = 0.03  # of the initial volume, per bolus, at feed_mult = 1
TITER_FIRST_DAY = 5
LAB_OFFSET_H = 1.0  # daily sample time after each batch-day boundary
EARLY_HARVEST_VIABILITY = 60.0
ABORT_CONTAMINANT = 30.0

# The bioreactor's phase classes (the other processes' phases are theirs, ADR-0019).
BIO_PHASES = (PhaseClass.TEMP_CTRL, PhaseClass.PH_CTRL, PhaseClass.DO_CTRL, PhaseClass.FEED_ADD)

PHASES: dict[Operation, tuple[PhaseClass, ...]] = {
    Operation.SETUP: (PhaseClass.TEMP_CTRL, PhaseClass.PH_CTRL, PhaseClass.DO_CTRL),
    Operation.INOCULATION: (PhaseClass.TEMP_CTRL, PhaseClass.PH_CTRL, PhaseClass.DO_CTRL),
    Operation.GROWTH: BIO_PHASES,
    Operation.TEMP_SHIFT: BIO_PHASES,
    Operation.PRODUCTION: BIO_PHASES,
    Operation.HARVEST: (PhaseClass.TEMP_CTRL,),
}

HOLDABLE = (PhaseClass.PH_CTRL, PhaseClass.DO_CTRL, PhaseClass.FEED_ADD)


def batch_id(start: datetime, seq: int) -> str:
    """B<start year>-<global 4-digit sequence> (ADR-0009)."""
    if not 0 <= seq <= 9999:
        raise ValueError(f"batch sequence out of range: {seq}")
    return f"B{start.year}-{seq:04d}"


def batch_day(t_h: float) -> float:
    return (t_h - SETUP_H) / 24.0


def t_of_day(day: float) -> float:
    return SETUP_H + day * 24.0


@dataclass(frozen=True, slots=True)
class FaultSpec:
    kind: FaultType
    onset_day: float  # batch day
    params: dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hold:
    phase: PhaseClass
    start_h: float
    end_h: float


@dataclass(frozen=True, slots=True)
class BatchSpec:
    batch_id: str
    cell: str
    start: datetime
    recipe_id: str
    campaign: Campaign
    levers: Levers
    seed: int
    traits: Traits = field(default_factory=Traits)
    faults: tuple[FaultSpec, ...] = ()
    holds: tuple[Hold, ...] | None = None  # None: draw from the seed
    harvest_day: float = HARVEST_DAY


def draw_traits(rng: np.random.Generator) -> Traits:
    return Traits(
        growth=float(rng.normal(1.0, 0.03)),
        productivity=float(rng.normal(1.0, 0.05)),
        metabolism=float(rng.normal(1.0, 0.04)),
    )


def draw_holds(rng: np.random.Generator) -> tuple[Hold, ...]:
    """A few short, random phase HOLDs per batch, as an operator would cause."""
    holds = []
    for _ in range(int(rng.poisson(1.2))):
        start = t_of_day(float(rng.uniform(1.0, 12.0)))
        holds.append(
            Hold(
                phase=HOLDABLE[int(rng.integers(len(HOLDABLE)))],
                start_h=start,
                end_h=start + float(rng.uniform(0.33, 1.5)),
            )
        )
    return tuple(sorted(holds, key=lambda h: h.start_h))
