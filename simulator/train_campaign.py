"""Plan the historical API and OSD batches, and allocate API lots to tablet batches
(ADR-0019, ADR-0020). Pure.

As for the bioreactor (simulator/campaign.py), the history is designed so the levers can
be learned: manufacturing batches on the recipe in effect, with run-to-run jitter; two
blocks of process-characterisation (PC) batches from a Latin hypercube across the PARs;
about 15% of batches carry one fault, planned against an operation's start.

Batch ids are assigned later, across every process, in start-time order (ADR-0018).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from common.models import Campaign, FaultType, Operation
from simulator import recipes
from simulator.campaign import latin_hypercube
from simulator.train import LotUse, TrainFaultSpec, TrainSpec

FIRST_CELL = {"api": "RX-201", "osd": "BL-301"}
CYCLE = {"api": timedelta(days=2.5), "osd": timedelta(days=1.5)}
PC_FRACTION = 0.3
FAULT_FRACTION = 0.15
PROCESS_SEED = {"api": 101, "osd": 102}
RELEASE_DELAY = timedelta(days=21)  # QC release and shipping, Tuas to Freiburg
API_PER_BATCH_KG = 200.0
FREIBURG_REORDER_KG = 3 * API_PER_BATCH_KG  # below this, the next released Tuas lot ships

# When each fault strikes: (operation, earliest, latest hours into it), and its weight.
FAULT_PLAN: dict[str, dict[FaultType, tuple[Operation, float, float, float]]] = {
    "api": {
        FaultType.JACKET_FOULING: (Operation.CRYSTALLIZATION, 0.5, 3.0, 1.0),
        FaultType.DOSING_METER_DRIFT: (Operation.REACTION, 0.05, 0.4, 1.0),
        FaultType.AGITATOR_DEGRADATION: (Operation.REACTION, 0.5, 2.5, 1.0),
        FaultType.FILTER_BLINDING: (Operation.FILTRATION, 0.1, 0.6, 1.0),
        FaultType.VACUUM_LEAK: (Operation.DRYING, 0.5, 4.0, 1.0),
        FaultType.STUCK_SENSOR: (Operation.CRYSTALLIZATION, 0.5, 4.0, 0.8),
    },
    "osd": {
        FaultType.ROLL_FORCE_DRIFT: (Operation.COMPACTION, 0.3, 1.5, 1.0),
        FaultType.PUNCH_STICKING: (Operation.COMPRESSION, 0.3, 1.5, 1.0),
        FaultType.HOPPER_BRIDGING: (Operation.COMPRESSION, 0.5, 2.0, 1.0),
        FaultType.HVAC_HUMIDITY: (Operation.COMPACTION, 0.2, 2.0, 0.8),
        FaultType.STUCK_SENSOR: (Operation.COMPRESSION, 0.5, 2.0, 0.8),
    },
}
# Sensors a stuck-sensor fault may hit: none that a control loop acts on, so the fault
# stays a data-quality problem rather than a process upset.
STUCK_TAGS: dict[str, tuple[str, ...]] = {
    "api": ("jacket", "power", "pressure", "level", "chord"),
    "osd": ("ejection_force", "precomp_force", "hardness", "room_rh"),
}


def plan(process: str, n: int, end: datetime, seed: int) -> list[TrainSpec]:
    """`n` batches of a train process ending the day before `end`, in start order."""
    if n < 1:
        return []
    rng = np.random.default_rng([seed, PROCESS_SEED[process]])
    end_day = datetime(end.year, end.month, end.day, tzinfo=UTC) - timedelta(days=1)
    first = end_day - CYCLE[process] * n
    names = recipes.LEVERS_OF[process]
    pc = _pc_indices(n)
    lhs = latin_hypercube(max(len(pc), 1), len(names), rng)
    faults = _assign_faults(process, n, rng)
    specs = []
    for i in range(n):
        start = first + CYCLE[process] * i
        recipe = recipes.current(start.date(), process)
        if i in pc:
            u = lhs[pc.index(i)]
            levers = recipe.levers_model(
                **{k: recipe.par[k][0] + (recipe.par[k][1] - recipe.par[k][0]) * float(u[j])
                   for j, k in enumerate(names)}
            )  # fmt: skip
            campaign = Campaign.PC
        else:
            levers = jittered(recipe, rng)
            campaign = Campaign.MFG
        specs.append(_spec(process, i, start, recipe, campaign, levers, seed, faults.get(i, ())))
    return specs


def manufacturing(
    process: str, recipe_id: str, n: int, first_start: datetime, seed: int
) -> list[TrainSpec]:
    """`n` fault-free manufacturing batches of one recipe (test harness, experiments)."""
    rng = np.random.default_rng([seed, PROCESS_SEED[process], 2])
    recipe = recipes.get(recipe_id)
    return [
        _spec(process, 9000 + i, first_start + CYCLE[process] * i, recipe, Campaign.MFG,
              jittered(recipe, rng), seed, ())
        for i in range(n)
    ]  # fmt: skip


def jittered(recipe: recipes.Recipe, rng: np.random.Generator):
    values = {
        k: getattr(recipe.nominal, k) + float(rng.normal(0.0, recipe.jitter[k]))
        for k in recipe.levers_model.model_fields
    }
    return recipe.clip(recipe.levers_model(**values))


def draw_traits(process: str, rng: np.random.Generator) -> dict[str, float]:
    if process == "api":
        return {
            "reactivity": float(rng.normal(1.0, 0.04)),
            "nucleation": float(np.exp(rng.normal(0.0, 0.15))),
            "cloth": float(rng.normal(1.0, 0.05)),
        }
    return {"tabletability": float(rng.normal(1.0, 0.03)), "flow": float(rng.normal(1.0, 0.04))}


def _spec(process, i, start, recipe, campaign, levers, seed, faults) -> TrainSpec:
    batch_rng = np.random.default_rng([seed, PROCESS_SEED[process], i])
    return TrainSpec(
        batch_id=f"B{start.year}-{i % 10000:04d}",  # provisional; renumbered globally
        cell=FIRST_CELL[process],
        start=start,
        recipe_id=recipe.id,
        campaign=campaign,
        levers=levers,
        seed=int(batch_rng.integers(2**31)),
        traits=draw_traits(process, batch_rng),
        faults=faults,
    )


def _pc_indices(n: int) -> list[int]:
    """Two contiguous PC blocks, a fifth and three fifths of the way through history."""
    size = round(n * PC_FRACTION / 2)
    blocks: list[int] = []
    for frac in (0.2, 0.6):
        start = int(n * frac)
        blocks.extend(range(start, min(n, start + size)))
    return blocks


def _assign_faults(
    process: str, n: int, rng: np.random.Generator
) -> dict[int, tuple[TrainFaultSpec, ...]]:
    table = FAULT_PLAN[process]
    kinds = list(table)
    weights = np.array([table[k][3] for k in kinds])
    count = round(n * FAULT_FRACTION)
    chosen = sorted(int(x) for x in rng.choice(n, size=min(count, n), replace=False))
    types = rng.choice(len(kinds), size=len(chosen), p=weights / weights.sum())
    out: dict[int, tuple[TrainFaultSpec, ...]] = {}
    for i, t in zip(chosen, types, strict=True):
        kind = kinds[int(t)]
        op, lo, hi, _ = table[kind]
        params: dict[str, float | str] = {}
        if kind is FaultType.STUCK_SENSOR:
            tags = STUCK_TAGS[process]
            params["tag"] = tags[int(rng.integers(len(tags)))]
        out[i] = (TrainFaultSpec(kind, op, float(rng.uniform(lo, hi)), params),)
    return out


# --- genealogy (ADR-0020) ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Lot:
    """An API lot as Tuas released it: what Freiburg's allocation needs to know."""

    lot: str
    material: str
    quantity_kg: float
    released: datetime
    properties: dict[str, float]


def allocate(tablets: list[TrainSpec], lots: list[Lot]) -> tuple[list[TrainSpec], list[Lot]]:
    """Give each tablet batch its API lots, first in, first out, and return the specs
    with `lots` filled and the stock left at Freiburg afterwards.

    Tuas also supplies other sites: a released lot ships to Freiburg only while
    Freiburg holds less than three batches' worth. A tablet batch that finds too little
    released stock draws on the next lots due (history is planned so this is rare).
    """
    events = sorted(
        [(lot.released, 0, lot) for lot in lots] + [(s.start, 1, s) for s in tablets],
        key=lambda e: (e[0], e[1]),
    )
    stock: list[list] = []  # [lot, kg left]
    pending = [lot for lot in sorted(lots, key=lambda x: x.released)]
    shipped: set[str] = set()
    out: dict[str, TrainSpec] = {}
    for _, kind, item in events:
        if kind == 0:
            if item.lot in shipped:  # expedited earlier
                continue
            pending.remove(item)
            if sum(kg for _, kg in stock) < FREIBURG_REORDER_KG:
                stock.append([item, item.quantity_kg])
                shipped.add(item.lot)
            continue
        need, uses = API_PER_BATCH_KG, []
        while need > 1e-9:
            if not stock:
                if not pending:
                    break
                nxt = pending.pop(0)  # short of stock: expedite the next lot due
                stock.append([nxt, nxt.quantity_kg])
                shipped.add(nxt.lot)
            lot, kg = stock[0]
            take = min(kg, need)
            uses.append(LotUse(lot.lot, lot.material, round(take, 3), dict(lot.properties)))
            need -= take
            stock[0][1] -= take
            if stock[0][1] <= 1e-9:
                stock.pop(0)
        out[item.batch_id] = dataclasses.replace(item, lots=tuple(uses))
    # What Freiburg holds, including lots whose release date is still to come.
    left = [dataclasses.replace(lot, quantity_kg=round(kg, 3)) for lot, kg in stock if kg > 1e-9]
    return [out.get(s.batch_id, s) for s in tablets], left
