"""Ground truth for evaluating the optimizer (ADR-0015): the true final titer of a batch
under a given set of levers. Available because the plant is simulated (ADR-0008);
nothing in the live path may call this.
"""

from __future__ import annotations

from datetime import UTC, datetime

from common.models import BatchStatus, Campaign, Levers, Operation
from simulator import batches as b
from simulator.engine import BatchRun
from simulator.process import Traits

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def simulate_remaining(run: BatchRun, levers: Levers) -> float:
    """True final titer if `run` continued from now with `levers`.

    Levers whose window has passed keep their actual effect: the shift day cannot move
    once the shift has started, whatever `levers` says. Faults already in the run carry
    on. Returns 0.0 for a batch that ends ABORTED.
    """
    clone = run.snapshot()
    clone.quiet = True
    shift_open = clone.operation in (Operation.SETUP, Operation.INOCULATION, Operation.GROWTH)
    if not shift_open or levers.shift_day <= clone.batch_day:
        levers = levers.model_copy(update={"shift_day": clone.levers.shift_day})
    clone.levers = levers
    clone.run_to_end()
    return 0.0 if clone.status is BatchStatus.ABORTED else clone.state.titer


def final_titer(levers: Levers, traits: Traits | None = None, seed: int = 0) -> float:
    """True final titer of a fault-free batch run start to finish with `levers`."""
    spec = b.BatchSpec(
        batch_id="B2026-9999",
        cell="BR-101",
        start=_EPOCH,
        recipe_id="truth",
        campaign=Campaign.PC,
        levers=levers,
        seed=seed,
        traits=traits or Traits(),
        holds=(),
    )
    run = BatchRun(spec, quiet=True)
    run.run_to_end()
    return run.state.titer
