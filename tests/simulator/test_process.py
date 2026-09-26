"""The process model gives the optimizer real, interior optima (ADR-0008, Q10)."""

from __future__ import annotations

from datetime import date

import pytest

from simulator import recipes, truth
from simulator.engine import BatchRun, LabOut
from tests.simulator.conftest import make_spec

NOMINAL = recipes.get("v3").nominal


def titer(**overrides: float) -> float:
    return truth.final_titer(NOMINAL.model_copy(update=overrides))


def test_nominal_batch_is_plausible():
    run = BatchRun(make_spec())
    msgs = run.run_to_end()
    lab = {}
    for m in msgs:
        if isinstance(m, LabOut):
            lab.setdefault(m.name, []).append(m.value)
    assert 15 < max(lab["vcd"]) < 30
    assert 3.5 < lab["titer"][-1] < 6.5
    assert lab["viability"][-1] > 70
    assert min(lab["glucose"]) > 0.5
    assert 1800 < run.state.vol < 2100


@pytest.mark.slow
@pytest.mark.parametrize(
    ("lever", "low", "best", "high"),
    [
        ("shift_day", 4.0, 5.0, 7.0),
        ("prod_temp", 32.0, 33.5, 35.0),
        ("ph_sp", 6.9, 7.0, 7.1),
        ("feed_mult", 0.8, 1.1, 1.2),
    ],
)
def test_levers_have_interior_optima(lever, low, best, high):
    t_best = titer(**{lever: best})
    assert t_best > titer(**{lever: low})
    assert t_best > titer(**{lever: high}) - 0.05  # feed plateaus near the top of the PAR


@pytest.mark.slow
def test_v3_leaves_room_for_the_optimizer():
    assert titer(prod_temp=33.5, feed_mult=1.1) > titer() + 0.05


def test_recipes_share_pars_and_nominals_sit_inside():
    all_r = recipes.load_all()
    assert {r.id for r in recipes.of_process("bioreactor")} == {"v1", "v2", "v3"}
    for process in ("bioreactor", "api", "osd"):
        versions = recipes.of_process(process)
        assert len(versions) >= 2
        assert all(r.par == versions[0].par for r in versions), process
    for r in all_r.values():
        assert r.clip(r.nominal) == r.nominal
    assert recipes.current(date(2023, 6, 1)).id == "v1"
    assert recipes.current(date(2026, 9, 1)).id == "v3"
    assert recipes.current(date(2026, 9, 1), "api").id == "asa-v2"
    assert recipes.current(date(2026, 9, 1), "osd").id == "tab-v2"


def test_truth_counterfactual_respects_frozen_shift():
    run = BatchRun(make_spec())
    run.advance(24.0 * 7)  # well past the day-5 shift
    later_shift = NOMINAL.model_copy(update={"shift_day": 6.5})
    assert truth.simulate_remaining(run, later_shift) == pytest.approx(
        truth.simulate_remaining(run, NOMINAL), rel=1e-9
    )
