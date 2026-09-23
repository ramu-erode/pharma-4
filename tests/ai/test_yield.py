"""Yield prediction and advice (ADR-0015), without trained artefacts."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ai.yield_ import optimize, rsm
from ai.yield_.features import FEATURES, training_days, with_levers
from ai.yield_.model import measured
from ai.yield_.rsm import LEVERS, ResponseSurface, fit_surface
from simulator import recipes

PAR = recipes.get("v3").par
NOMINAL = recipes.get("v3").nominal


def true_titer(L: np.ndarray) -> np.ndarray:
    """A known surface: best at prod_temp 33.5, feed_mult 1.1; the other levers are flat."""
    L = np.atleast_2d(L)
    t, f = L[:, LEVERS.index("prod_temp")], L[:, LEVERS.index("feed_mult")]
    return 5.0 - 0.5 * (t - 33.5) ** 2 - 4 * (f - 1.1) ** 2


def doe(n: int = 60, noise: float = 0.05, seed: int = 0) -> ResponseSurface:
    rng = np.random.default_rng(seed)
    L = np.stack([rng.uniform(*PAR[k], n) for k in LEVERS], axis=1)
    y = true_titer(L) + rng.normal(0, noise, n)
    return fit_surface(L, y, PAR, seed)


class FakeModel:
    """The in-batch titer model only supplies the displayed current prediction."""

    def __init__(self, surface: ResponseSurface) -> None:
        self.surface = surface

    def member_predictions(self, X):
        return np.full((20, len(np.atleast_2d(X))), 4.5)


SURFACE = doe()


def row() -> np.ndarray:
    return np.zeros(len(FEATURES))


def test_shift_day_is_frozen_once_the_batch_has_shifted():
    assert "shift_day" in optimize.open_levers(3.0, "Growth", NOMINAL)
    assert "shift_day" not in optimize.open_levers(6.0, "Production", NOMINAL)
    assert "shift_day" not in optimize.open_levers(5.1, "TempShift", NOMINAL)


def test_gate_closes_when_the_ensemble_disagrees():
    recommended = {k: 0.0 for k in LEVERS} | {"prod_temp": 1.0}
    base = {
        "current": {k: 0.0 for k in LEVERS},
        "recommended": recommended,
        "frozen": set(),
        "predicted_current": np.zeros(10),
        "predicted_recommended": np.zeros(10),
    }
    straddle = optimize.Advice(**base, gain=np.linspace(-0.3, 0.5, 10))
    clear = optimize.Advice(**base, gain=np.linspace(0.15, 0.4, 10))
    tiny = optimize.Advice(**base, gain=np.linspace(0.01, 0.05, 10))
    assert not straddle.passes_gate
    assert clear.passes_gate
    assert not tiny.passes_gate  # agreed, but below the practical minimum


def test_gate_needs_an_actual_change():
    same = {k: 1.0 for k in LEVERS}
    advice = optimize.Advice(same, same, set(), np.zeros(10), np.ones(10), np.ones(10))
    assert not advice.passes_gate


def test_surface_recovers_a_known_optimum():
    grid = np.array([[5.0, t, 7.0, 40.0, f] for t in np.linspace(32, 35, 31) for f in (1.1,)])
    best = grid[SURFACE.predict(grid).mean(axis=0).argmax()]
    assert abs(best[1] - 33.5) < 0.3


def test_recommendation_moves_toward_the_optimum_and_the_gain_is_honest():
    levers = NOMINAL.model_copy(update={"prod_temp": 32.6, "feed_mult": 0.95})
    advice = optimize.recommend(FakeModel(SURFACE), row(), levers, 2.0, "Growth", PAR, seed=1)
    assert advice.recommended["prod_temp"] > 32.8 and advice.recommended["feed_mult"] > 1.0
    assert advice.passes_gate
    now = np.array([getattr(levers, k) for k in LEVERS])
    moved = optimize.effective(advice.current, advice.recommended, 2.0)
    true_gain = float(true_titer(moved)[0] - true_titer(now)[0])
    assert abs(np.median(advice.gain) - true_gain) < 0.1


def test_no_advice_at_the_optimum():
    levers = NOMINAL.model_copy(update={"prod_temp": 33.5, "feed_mult": 1.1})
    advice = optimize.recommend(FakeModel(SURFACE), row(), levers, 4.0, "Growth", PAR, seed=1)
    assert not advice.passes_gate


def test_too_few_runs_is_refused():
    with pytest.raises(ValueError, match="DoE runs"):
        doe(n=12)


def test_a_noisy_experiment_keeps_the_gate_shut_near_the_optimum():
    noisy = doe(n=40, noise=0.8, seed=3)
    levers = NOMINAL.model_copy(update={"prod_temp": 33.3, "feed_mult": 1.08})
    advice = optimize.recommend(FakeModel(noisy), row(), levers, 4.0, "Growth", PAR, seed=1)
    assert not advice.passes_gate


def test_exposure_shrinks_late_changes():
    assert rsm.exposure("ph_sp", 0.0, 5.0) == 1.0
    assert rsm.exposure("ph_sp", 7.0, 5.0) == 0.5
    assert rsm.exposure("prod_temp", 3.0, 5.0) == 1.0  # before the shift: all ahead
    assert rsm.exposure("prod_temp", 9.5, 5.0) == 0.5
    assert rsm.exposure("feed_mult", 13.0, 5.0) == 0.0
    assert rsm.exposure("shift_day", 3.0, 5.0) == 1.0


@settings(max_examples=15, deadline=None)
@given(
    st.floats(PAR["prod_temp"][0], PAR["prod_temp"][1]),
    st.floats(PAR["feed_mult"][0], PAR["feed_mult"][1]),
    st.floats(3.0, 12.0),
)
def test_recommendations_stay_inside_the_pars_and_the_trust_region(temp, feed, day):
    levers = NOMINAL.model_copy(update={"prod_temp": temp, "feed_mult": feed})
    op = "Growth" if day < levers.shift_day else "Production"
    advice = optimize.recommend(FakeModel(SURFACE), row(), levers, day, op, PAR, seed=1, trials=20)
    for k, v in advice.recommended.items():
        lo, hi = PAR[k]
        assert lo - 1e-9 <= v <= hi + 1e-9
        assert abs(v - advice.current[k]) <= optimize.TRUST * (hi - lo) + 1e-9
    if op == "Production":
        assert advice.recommended["shift_day"] == levers.shift_day
    else:
        assert advice.recommended["shift_day"] >= min(
            day + optimize.SHIFT_LEAD_DAYS, levers.shift_day
        )


def test_no_label_or_truth_can_reach_the_features():
    forbidden = ("fault", "label", "truth", "injection", "onset")
    assert not any(word in f for f in FEATURES for word in forbidden)


def test_training_days_stop_before_harvest():
    assert training_days(14.0)[0] == 3.0 and training_days(14.0)[-1] == 12.0
    assert training_days(10.2)[-1] <= 9.7


def test_measured_titer_is_zero_before_the_first_sample():
    x = with_levers(row(), NOMINAL.model_dump())
    x[FEATURES.index("titer_so_far")] = np.nan
    assert measured(x)[0] == 0.0
    x[FEATURES.index("titer_so_far")] = 2.5
    assert measured(x)[0] == 2.5
