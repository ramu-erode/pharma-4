"""The processes the simulator runs, and how to make a run of each (ADR-0018, ADR-0019).

The bioreactor runs a whole batch on one unit (`BatchRun`, `BatchSpec`); the API and OSD
processes run a batch through their train (`TrainRun` subclasses, `TrainSpec`). Callers
that handle every process (the live runner, backfill, the harness) come through here.
"""

from __future__ import annotations

from common.models import FaultType
from simulator import recipes
from simulator.api.engine import ApiRun
from simulator.batches import BatchSpec
from simulator.engine import BatchRun
from simulator.osd.engine import OsdRun
from simulator.train import TrainRun, TrainSpec

Run = BatchRun | TrainRun
Spec = BatchSpec | TrainSpec

TRAIN_RUNS: dict[str, type[TrainRun]] = {"api": ApiRun, "osd": OsdRun}

FAULTS: dict[str, tuple[FaultType, ...]] = {
    "bioreactor": (
        FaultType.PH_PROBE_DRIFT,
        FaultType.DO_SPARGER_FOULING,
        FaultType.TEMP_CONTROL_LOSS,
        FaultType.FEED_PUMP_FAILURE,
        FaultType.STUCK_SENSOR,
        FaultType.CONTAMINATION,
    ),
    "api": tuple(ApiRun.fault_defaults),
    "osd": tuple(OsdRun.fault_defaults),
}

# The process's yield target, its unit, and the lab result that reports it.
TARGET: dict[str, tuple[str, str]] = {
    "bioreactor": ("titer", "g/L"),
    "api": ("yield", "%"),
    "osd": ("yield", "%"),
}


def process_of(spec: Spec) -> str:
    return recipes.get(spec.recipe_id).process if isinstance(spec, TrainSpec) else "bioreactor"


def make_run(spec: Spec, **kwargs: object) -> Run:
    """A run for any process's spec. Keyword arguments go to the run's constructor."""
    if isinstance(spec, TrainSpec):
        return TRAIN_RUNS[process_of(spec)](spec, **kwargs)
    return BatchRun(spec, **kwargs)
