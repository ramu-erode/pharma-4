"""In-process fault harness for the API and OSD trains (ADR-0019, ADR-0021). Simulates
clean manufacturing batches and one batch per fault type through engine -> edge core ->
features, fits one anomaly model per scored equipment class, and scores. No broker, no
databases. Results are cached per test session.

A fault case names the equipment class that must catch it, when it strikes (hours into
an operation), and how detection is judged:
"before_breach": the first alert must open before the true process leaves its action
limit (`breach_minute`); a number: the first alert must open within that many hours.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache

from ai.anomaly import train
from ai.anomaly.model import AnomalyModel
from ai.offline import Collected, from_engine_units, to_input
from ai.profiles import profile_for
from common.models import Campaign, FaultType, Operation
from common.plant import get_plant
from common.settings import Settings
from edge.core import TagMap
from simulator import recipes
from simulator import train_campaign as tc
from simulator.train import TrainFaultSpec, TrainSpec

SETTINGS = Settings()
N_TRAIN = 40
N_HOLDOUT = 8
RECIPE = {"api": "asa-v2", "osd": "tab-v2"}


@dataclass(frozen=True)
class Case:
    process: str
    cls: str
    operation: Operation
    after_h: float
    params: dict
    criterion: str | float


CASES: dict[FaultType, Case] = {
    FaultType.JACKET_FOULING: Case(
        "api", "reactor", Operation.CRYSTALLIZATION, 1.0, {}, "before_breach"
    ),
    # A meter drift must be caught while the anhydride charge can still be corrected.
    FaultType.DOSING_METER_DRIFT: Case("api", "reactor", Operation.REACTION, 0.1, {}, 3.0),
    FaultType.AGITATOR_DEGRADATION: Case("api", "reactor", Operation.REACTION, 1.5, {}, 1.0),
    FaultType.FILTER_BLINDING: Case("api", "filter_dryer", Operation.FILTRATION, 0.3, {}, 1.0),
    FaultType.VACUUM_LEAK: Case("api", "filter_dryer", Operation.DRYING, 1.0, {}, "before_breach"),
    FaultType.ROLL_FORCE_DRIFT: Case(
        "osd", "roller_compactor", Operation.COMPACTION, 0.5, {}, "before_breach"
    ),
    FaultType.PUNCH_STICKING: Case(
        "osd", "tablet_press", Operation.COMPRESSION, 0.5, {}, "before_breach"
    ),
    FaultType.HOPPER_BRIDGING: Case("osd", "tablet_press", Operation.COMPRESSION, 1.0, {}, 1.0),
    FaultType.HVAC_HUMIDITY: Case(
        "osd", "roller_compactor", Operation.COMPACTION, 0.5, {}, "before_breach"
    ),
}
# Stuck sensors, one per process, each on an analog signal no loop controls.
STUCK: dict[str, Case] = {
    "api": Case("api", "reactor", Operation.CRYSTALLIZATION, 2.0, {"tag": "jacket"}, 1.0),
    "osd": Case("osd", "tablet_press", Operation.COMPRESSION, 1.0, {"tag": "ejection_force"}, 1.0),
}
STUCK_KEY = {"api": "rules-stuck_jacket_temperature", "osd": "rules-stuck_ejection_force"}


def cell_of(cls: str) -> str:
    return get_plant().cells(cls=cls)[0]


def _clean(spec: TrainSpec) -> tuple[str, dict[str, Collected]]:
    return spec.batch_id, from_engine_units(spec, SETTINGS, TagMap.load())


def _fault(args: tuple[str, FaultType, Case]) -> tuple[str, dict[str, Collected]]:
    key, kind, case = args
    spec = TrainSpec(
        batch_id="B2026-0900",
        cell=tc.FIRST_CELL[case.process],
        start=datetime(2026, 10, 1, tzinfo=UTC),
        recipe_id=RECIPE[case.process],
        campaign=Campaign.MFG,
        levers=recipes.get(RECIPE[case.process]).nominal,
        seed=100 + list(FaultType).index(kind),
        faults=(TrainFaultSpec(kind, case.operation, case.after_h, case.params),),
    )
    return key, from_engine_units(spec, SETTINGS, TagMap.load())


@dataclass
class TrainHarness:
    models: dict[str, AnomalyModel]  # by equipment class
    holdout: dict[str, list[train.BatchInput]]
    faults: dict[str, tuple[Case, Collected, train.BatchInput]]  # by case key


@lru_cache(maxsize=1)
def harness() -> TrainHarness:
    first = datetime(2025, 1, 1, tzinfo=UTC)
    specs = [
        s
        for p in ("api", "osd")
        for s in tc.manufacturing(p, RECIPE[p], N_TRAIN + N_HOLDOUT, first, 7)
    ]
    jobs = [(k.value, k, c) for k, c in CASES.items()]
    jobs += [(f"stuck_{p}", FaultType.STUCK_SENSOR, c) for p, c in STUCK.items()]
    with ProcessPoolExecutor() as pool:
        clean = list(pool.map(_clean, specs))
        faulty = dict(pool.map(_fault, jobs))
    models, holdout = {}, {}
    for cls in sorted({c.cls for c in (*CASES.values(), *STUCK.values())}):
        cell, profile = cell_of(cls), profile_for(cls)
        inputs = [to_input(b, units[cell], profile=profile) for b, units in clean if cell in units]
        models[cls] = train.fit(inputs[:N_TRAIN], seed=1, profile=profile)
        holdout[cls] = inputs[N_TRAIN:]
    faults = {}
    for key, _, case in jobs:
        c = faulty[key][cell_of(case.cls)]
        faults[key] = (case, c, to_input("B2026-0900", c, profile=profile_for(case.cls)))
    return TrainHarness(models, holdout, faults)


def onset_minute(c: Collected) -> float:
    return (c.labels[0].onset - c.series.origin).total_seconds() / 60


def breach_minute(kind: FaultType, c: Collected) -> float | None:
    """When the *true* process first left its action limit after onset."""
    onset = onset_minute(c)
    for t in c.truth:
        minute = (t.t - c.series.origin).total_seconds() / 60
        if minute < onset:
            continue
        st, sp = t.state, t.sp
        match kind:
            case FaultType.JACKET_FOULING:
                if abs(st["rx"].temp - sp["temperature"]) > 2.0:
                    return minute
            case FaultType.VACUUM_LEAK:
                if st["fd"].vacuum > 120.0:
                    return minute
            case FaultType.ROLL_FORCE_DRIFT:
                if sp["roll_force"] - st["rc"].force > 0.5:
                    return minute
            case FaultType.PUNCH_STICKING:
                if st["tp"].ejection > 550.0:
                    return minute
            case FaultType.HVAC_HUMIDITY:
                if st["rh"] > 40.0:
                    return minute
    return None
