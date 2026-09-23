"""In-process fault harness (plan: Increment 4). Simulates clean manufacturing batches and
one batch per fault type through engine -> edge core -> features, fits the anomaly model,
and scores. No broker, no databases. Results are cached per test session."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache

from ai.anomaly import train
from ai.anomaly.model import AnomalyModel
from ai.offline import Collected, from_engine, to_input
from common.models import Campaign, FaultType
from common.settings import Settings
from edge.core import TagMap
from simulator import batches as b
from simulator import campaign, recipes

SETTINGS = Settings()
N_TRAIN = 40
N_HOLDOUT = 8

# Onset (batch day) and params per fault, and how the detection is judged:
# "before_breach": the first alert must open before the true process leaves spec;
# a number: the first alert must open within that many hours of `reference`.
CASES: dict[FaultType, tuple[float, dict, str | float]] = {
    FaultType.PH_PROBE_DRIFT: (2.3, {}, "before_breach"),
    FaultType.DO_SPARGER_FOULING: (3.5, {}, "before_breach"),
    FaultType.CONTAMINATION: (6.0, {}, "before_breach"),
    FaultType.TEMP_CONTROL_LOSS: (7.3, {}, 1.5),
    FaultType.STUCK_SENSOR: (8.2, {"tag": "pressure"}, 1.0),
    FaultType.FEED_PUMP_FAILURE: (6.5, {}, 3.0),  # measured from the first missed bolus
}


def _tag_map() -> TagMap:
    return TagMap.load(SETTINGS.site, SETTINGS.area, SETTINGS.line)


def _clean(spec: b.BatchSpec) -> train.BatchInput:
    return to_input(spec.batch_id, from_engine(spec, SETTINGS, _tag_map()))


def _fault(kind: FaultType) -> tuple[FaultType, Collected, train.BatchInput]:
    day, params, _ = CASES[kind]
    spec = b.BatchSpec(
        batch_id="B2026-0900",
        cell="BR-101",
        start=datetime(2026, 10, 1, tzinfo=UTC),
        recipe_id="v3",
        campaign=Campaign.MFG,
        levers=recipes.get("v3").nominal,
        seed=100 + list(FaultType).index(kind),
        faults=(b.FaultSpec(kind, day, params),),
    )
    c = from_engine(spec, SETTINGS, _tag_map())
    return kind, c, to_input(spec.batch_id, c)


@dataclass
class Harness:
    model: AnomalyModel
    holdout: list[train.BatchInput]
    faults: dict[FaultType, tuple[Collected, train.BatchInput]]


@lru_cache(maxsize=1)
def harness() -> Harness:
    specs = campaign.manufacturing("v3", N_TRAIN + N_HOLDOUT, datetime(2025, 1, 1, tzinfo=UTC), 7)
    with ProcessPoolExecutor() as pool:
        clean = list(pool.map(_clean, specs))
        faults = {k: (c, bi) for k, c, bi in pool.map(_fault, list(CASES))}
    model = train.fit(clean[:N_TRAIN], seed=1)
    return Harness(model, clean[N_TRAIN:], faults)


def onset_minute(kind: FaultType) -> float:
    return b.t_of_day(CASES[kind][0]) * 60


def breach_minute(kind: FaultType, c: Collected) -> float | None:
    """When the *true* process first left its action limits after onset."""
    onset = onset_minute(kind)
    for t in c.truth:
        minute = (t.t - c.series.origin).total_seconds() / 60
        if minute < onset:
            continue
        s, sp = t.state, t.sp
        ph_out = s.ph < sp["ph"] - 0.1 or s.ph > sp["ph"] + 0.1
        if kind in (FaultType.PH_PROBE_DRIFT, FaultType.CONTAMINATION) and ph_out:
            return minute
        if kind in (FaultType.DO_SPARGER_FOULING, FaultType.CONTAMINATION) and s.do < 20:
            return minute
    return None


def reference_minute(kind: FaultType) -> float:
    """What a max-delay criterion is measured from."""
    onset = onset_minute(kind)
    if kind is FaultType.FEED_PUMP_FAILURE:
        day = CASES[kind][0]
        return b.t_of_day(float(int(day) + 1)) * 60  # the next scheduled bolus
    return onset
