"""Yield prediction and advice (ADR-0014), without trained artefacts."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ai.yield_ import optimize
from ai.yield_.features import FEATURES, LEVERS, training_days, with_levers
from ai.yield_.model import measured
from simulator import recipes

PAR = recipes.get("v3").par
NOMINAL = recipes.get("v3").nominal


class Peaked:
    """A stand-in booster: titer gain peaks at prod_temp 33.5 and feed_mult 1.1."""

    def __init__(self, bias: float = 0.0) -> None:
        self.bias = bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        t = X[:, FEATURES.index("prod_temp")]
        f = X[:, FEATURES.index("feed_mult")]
        return 5.0 - (t - 33.5) ** 2 - 4 * (f - 1.1) ** 2 + self.bias


class FakeModel:
    def __init__(self, members) -> None:
        self.members = members

    def member_predictions(self, X):
        return np.stack([b.predict(X) for b in self.members]) + measured(X)

    def predict(self, X):
        return self.member_predictions(X).mean(axis=0)


def row(**levers: float) -> np.ndarray:
    x = np.zeros(len(FEATURES))
    x[FEATURES.index("titer_so_far")] = np.nan
    return with_levers(x, {**NOMINAL.model_dump(), **levers})


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


def test_recommendation_moves_toward_the_optimum():
    model = FakeModel([Peaked(b) for b in np.linspace(-0.05, 0.05, 20)])
    levers = NOMINAL.model_copy(update={"prod_temp": 33.0, "feed_mult": 1.0})
    x = row(prod_temp=33.0, feed_mult=1.0)
    advice = optimize.recommend(model, x, levers, 4.0, "Growth", PAR, seed=1, trials=60)
    assert advice.recommended["prod_temp"] > 33.2 and advice.recommended["feed_mult"] > 1.04
    assert np.median(advice.gain) > 0


@settings(max_examples=15, deadline=None)
@given(
    st.floats(PAR["prod_temp"][0], PAR["prod_temp"][1]),
    st.floats(PAR["feed_mult"][0], PAR["feed_mult"][1]),
    st.floats(3.0, 12.0),
)
def test_recommendations_stay_inside_the_pars_and_the_trust_region(temp, feed, day):
    model = FakeModel([Peaked(0.0)] * 4)
    levers = NOMINAL.model_copy(update={"prod_temp": temp, "feed_mult": feed})
    op = "Growth" if day < levers.shift_day else "Production"
    x = row(prod_temp=temp, feed_mult=feed)
    advice = optimize.recommend(model, x, levers, day, op, PAR, seed=1, trials=20)
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
    x = row()
    assert measured(x)[0] == 0.0
    x[FEATURES.index("titer_so_far")] = 2.5
    assert measured(x)[0] == 2.5
