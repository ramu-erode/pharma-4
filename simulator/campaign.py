"""Plan the historical batches for backfill (implementation plan task 2.4). Pure.

The history is designed so the yield levers can be learned (Q10):

- about 3/4 manufacturing batches, on whichever recipe version was in effect on the
  start date, with small run-to-run jitter on each lever;
- about 1/4 process-characterisation (PC) batches in two blocks, their levers from a
  Latin hypercube across the PARs;
- about 15% carry one fault; contamination is rare.

Two reactors run back to back, staggered by half a cycle, ending the day before `end`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from common.models import Campaign, FaultType, Levers
from simulator import batches as b
from simulator import recipes
from simulator.recipes import LEVER_NAMES

CELLS = ("BR-101", "BR-102")
CYCLE = timedelta(days=16.5)  # 14.5-day batch + 2 days turnaround
FAULT_FRACTION = 0.15
PC_FRACTION = 0.25
STUCK_TAGS = ("do", "ph", "temp", "pressure", "weight", "agitation")

# Relative weights of fault types; contamination is rare.
FAULT_WEIGHTS: dict[FaultType, float] = {
    FaultType.PH_PROBE_DRIFT: 1.0,
    FaultType.DO_SPARGER_FOULING: 1.0,
    FaultType.TEMP_CONTROL_LOSS: 1.0,
    FaultType.FEED_PUMP_FAILURE: 0.9,
    FaultType.STUCK_SENSOR: 0.9,
    FaultType.CONTAMINATION: 0.3,
}


@dataclass(frozen=True, slots=True)
class Plan:
    specs: tuple[b.BatchSpec, ...]

    def by_campaign(self, campaign: Campaign) -> list[b.BatchSpec]:
        return [s for s in self.specs if s.campaign is campaign]


def latin_hypercube(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """n points in [0, 1)^dims, one per stratum in every dimension."""
    u = (rng.random((n, dims)) + np.arange(n)[:, None]) / n
    for d in range(dims):
        u[:, d] = u[rng.permutation(n), d]
    return u


def plan(n: int, end: datetime, seed: int) -> Plan:
    """Plan `n` batches finishing before `end` (normally the day before first boot)."""
    if n < 1:
        raise ValueError("need at least one batch")
    rng = np.random.default_rng([seed, 1])
    end_day = datetime(end.year, end.month, end.day, tzinfo=UTC) - timedelta(days=1)

    # Starts: each reactor back to back, BR-102 half a cycle behind BR-101.
    # The staggered reactor finishes half a cycle later, so start half a cycle earlier.
    per_cell = (n + 1) // 2
    first = end_day - CYCLE * per_cell - CYCLE / 2
    starts = sorted(
        (first + CYCLE * i + (CYCLE / 2) * c, cell)
        for c, cell in enumerate(CELLS)
        for i in range(per_cell)
    )[:n]

    pc = _pc_indices(n)
    lhs = latin_hypercube(max(len(pc), 1), len(LEVER_NAMES), rng)
    faults = _assign_faults(n, rng)

    specs = []
    for seq, (start, cell) in enumerate(starts):
        recipe = recipes.current(start.date())
        if seq in pc:
            u = lhs[pc.index(seq)]
            levers = Levers(
                **{
                    k: lo + (hi - lo) * float(u[i])
                    for i, (k, (lo, hi)) in enumerate((k, recipe.par[k]) for k in LEVER_NAMES)
                }
            )
            campaign = Campaign.PC
        else:
            levers = jittered(recipe, rng)
            campaign = Campaign.MFG
        batch_rng = np.random.default_rng([seed, seq])
        specs.append(
            b.BatchSpec(
                batch_id=b.batch_id(start, seq),
                cell=cell,
                start=start,
                recipe_id=recipe.id,
                campaign=campaign,
                levers=levers,
                seed=int(batch_rng.integers(2**31)),
                traits=b.draw_traits(batch_rng),
                faults=faults.get(seq, ()),
            )
        )
    return Plan(tuple(specs))


def jittered(recipe: recipes.Recipe, rng: np.random.Generator) -> Levers:
    """A manufacturing batch's levers: nominal plus run-to-run jitter, inside the PARs."""
    values = {
        k: getattr(recipe.nominal, k) + float(rng.normal(0.0, recipe.jitter[k]))
        for k in LEVER_NAMES
    }
    return recipe.clip(Levers(**values))


def manufacturing(
    recipe_id: str, n: int, first_start: datetime, seed: int, first_seq: int = 9000
) -> list[b.BatchSpec]:
    """`n` fault-free manufacturing batches of one recipe (test harness, experiments)."""
    rng = np.random.default_rng([seed, 2])
    recipe = recipes.get(recipe_id)
    specs = []
    for i in range(n):
        start = first_start + CYCLE * (i // 2) + (CYCLE / 2) * (i % 2)
        batch_rng = np.random.default_rng([seed, first_seq + i])
        specs.append(
            b.BatchSpec(
                batch_id=b.batch_id(start, (first_seq + i) % 10000),
                cell=CELLS[i % 2],
                start=start,
                recipe_id=recipe_id,
                campaign=Campaign.MFG,
                levers=jittered(recipe, rng),
                seed=int(batch_rng.integers(2**31)),
                traits=b.draw_traits(batch_rng),
            )
        )
    return specs


def _pc_indices(n: int) -> list[int]:
    """Two contiguous PC blocks, a fifth and three fifths of the way through history."""
    size = round(n * PC_FRACTION / 2)
    blocks = []
    for frac in (0.2, 0.6):
        start = int(n * frac)
        blocks.extend(range(start, min(n, start + size)))
    return blocks


def _assign_faults(n: int, rng: np.random.Generator) -> dict[int, tuple[b.FaultSpec, ...]]:
    count = round(n * FAULT_FRACTION)
    kinds = list(FAULT_WEIGHTS)
    weights = np.array([FAULT_WEIGHTS[k] for k in kinds])
    chosen = rng.choice(n, size=count, replace=False)
    # Deterministic, near-proportional allocation of types, then shuffled over batches.
    quota = np.floor(weights / weights.sum() * count).astype(int)
    for i in np.argsort(-(weights / weights.sum() * count - quota))[: count - quota.sum()]:
        quota[i] += 1
    types = [k for k, q in zip(kinds, quota, strict=True) for _ in range(q)]
    rng.shuffle(types)
    out: dict[int, tuple[b.FaultSpec, ...]] = {}
    for seq, kind in zip(sorted(int(x) for x in chosen), types, strict=True):
        params: dict[str, float | str] = {}
        if kind is FaultType.STUCK_SENSOR:
            params["tag"] = STUCK_TAGS[int(rng.integers(len(STUCK_TAGS)))]
        if kind is FaultType.FEED_PUMP_FAILURE:
            params["duration_h"] = 48.0
        onset = float(rng.uniform(2.0, 9.0 if kind is FaultType.CONTAMINATION else 11.0))
        out[seq] = (b.FaultSpec(kind, onset, params),)
    return out
