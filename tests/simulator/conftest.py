from __future__ import annotations

from datetime import UTC, datetime

import pytest

from common.models import Campaign, FaultType
from simulator import batches as b
from simulator import recipes
from simulator.engine import BatchRun

START = datetime(2026, 9, 24, tzinfo=UTC)


def make_spec(
    faults: tuple[b.FaultSpec, ...] = (),
    seed: int = 7,
    holds: tuple[b.Hold, ...] | None = (),
    **lever_overrides: float,
) -> b.BatchSpec:
    levers = recipes.get("v3").nominal.model_copy(update=lever_overrides)
    return b.BatchSpec(
        batch_id="B2026-0200",
        cell="BR-101",
        start=START,
        recipe_id="v3",
        campaign=Campaign.MFG,
        levers=levers,
        seed=seed,
        faults=faults,
        holds=holds,
    )


def fault_run(kind: FaultType, onset_day: float, **params: float | str) -> BatchRun:
    return BatchRun(make_spec(faults=(b.FaultSpec(kind, onset_day, params),)), record_truth=True)


@pytest.fixture
def spec() -> b.BatchSpec:
    return make_spec()
