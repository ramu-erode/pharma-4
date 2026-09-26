"""Setpoint recommendation (ADR-0015). Pure: no MQTT.

What a lever change is worth comes from the response surface fitted to the designed
experiment (`rsm.py`), not from the in-batch titer model: fitted to all of history, that
model overstated lever effects about threefold (measured against the simulator's truth).

- Only open levers are searched (Optuna TPE), inside the PARs and a trust region around
  the current values: the quadratic is only trusted locally.
- A change made mid-batch acts only for the rest of the lever's window, so it is scored
  as a proportionally smaller whole-batch change (`rsm.exposure`).
- The bootstrap surfaces are split: half drive the search, the other half judge the
  winner (against the optimizer's curse). The judges' paired gains must have P10 > 0 and
  median >= MIN_GAIN, and some lever must actually move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import optuna
from pydantic import BaseModel

from ai.yield_.model import YieldModel
from ai.yield_.rsm import LEVERS, ResponseSurface, exposure
from common.models import Levers

if TYPE_CHECKING:
    from ai.yield_.profiles import At

MIN_GAIN = 0.1  # g/L: smallest gain worth a person's attention (the bioreactor's)
TRIALS = 150
TRUST = 0.25  # max move per recommendation, as a fraction of the PAR width
SHIFT_LEAD_DAYS = 0.25  # a shift can be moved to no sooner than 6 h from now
MIN_CHANGE = {"shift_day": 0.1, "prod_temp": 0.1, "ph_sp": 0.01, "do_sp": 1.0, "feed_mult": 0.02}

optuna.logging.set_verbosity(optuna.logging.WARNING)


def open_levers(day: float, operation: str | None, levers: Levers) -> set[str]:
    """Bioreactor levers that can still change the outcome at batch day `day`."""
    out = {"prod_temp", "ph_sp", "do_sp", "feed_mult"}
    if operation in ("Setup", "Inoculation", "Growth"):  # not shifted yet
        out.add("shift_day")
    return out


@dataclass
class Advice:
    current: dict[str, float]
    recommended: dict[str, float]
    frozen: set[str]
    predicted_current: np.ndarray  # in-batch model, per ensemble member
    predicted_recommended: np.ndarray  # the current prediction plus the judged gains
    gain: np.ndarray  # paired, per judging surface
    min_change: dict[str, float] = field(default_factory=lambda: MIN_CHANGE)
    min_gain: float = MIN_GAIN

    @property
    def passes_gate(self) -> bool:
        changed = any(
            abs(self.recommended[k] - self.current[k]) >= self.min_change[k]
            for k in self.recommended
        )
        return (
            changed
            and float(np.quantile(self.gain, 0.1)) > 0
            and float(np.median(self.gain)) >= self.min_gain
        )


def effective(current: dict[str, float], candidate: dict[str, float], day: float) -> np.ndarray:
    """The whole-batch bioreactor lever vector equivalent to changing to `candidate` on
    `day`."""
    shift = current["shift_day"]
    return np.array(
        [
            current[k] + exposure(k, day, shift) * (candidate.get(k, current[k]) - current[k])
            for k in LEVERS
        ]
    )


def _effective(
    weights: dict[str, float], names: tuple[str, ...], current: dict, candidate: dict
) -> np.ndarray:
    return np.array(
        [current[k] + weights[k] * (candidate.get(k, current[k]) - current[k]) for k in names]
    )


def gains(
    surface: ResponseSurface,
    current: dict[str, float],
    candidate: dict[str, float],
    weights: dict[str, float],
    which: slice,
) -> np.ndarray:
    """Paired gain of moving to `candidate` now, one per bootstrap surface. `weights`
    says how much of each lever's whole-batch effect a change can still have."""
    names = surface.levers
    now = np.array([current[k] for k in names])
    pred = surface.predict(np.stack([now, _effective(weights, names, current, candidate)]), which)
    return pred[:, 1] - pred[:, 0]


def recommend(
    model: YieldModel,
    x_now: np.ndarray,
    levers: BaseModel,
    day: float,
    operation: str | None,
    par: dict[str, tuple[float, float]],
    seed: int,
    trials: int = TRIALS,
    at: At | None = None,  # where a train batch is (ADR-0021)
) -> Advice:
    from ai.yield_.profiles import PROFILES, At  # it imports this module

    surface = model.surface
    if surface is None:
        raise ValueError("the yield model has no DoE response surface; retrain it")
    profile = PROFILES[getattr(model, "process", "bioreactor")]
    at = at or At(day, operation)
    names = surface.levers
    current = {k: float(getattr(levers, k)) for k in names}
    weights = {k: profile.exposure(k, at, levers) for k in names}
    bounds = {}
    for k in names:
        if weights[k] <= 0.0:
            continue
        lo, hi = par[k]
        step = TRUST * (hi - lo)
        bounds[k] = (max(lo, current[k] - step), min(hi, current[k] + step))
    if "shift_day" in bounds:
        lo, hi = bounds["shift_day"]
        bounds["shift_day"] = (max(lo, day + SHIFT_LEAD_DAYS), hi)
        if bounds["shift_day"][0] >= hi:
            del bounds["shift_day"]

    half = len(surface.coef) // 2
    search, judge = slice(0, half), slice(half, None)

    def objective(trial: optuna.Trial) -> float:
        cand = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in bounds.items()}
        return float(gains(surface, current, cand, weights, search).mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({k: min(max(current[k], lo), hi) for k, (lo, hi) in bounds.items()})
    study.optimize(objective, n_trials=trials)
    best = {**current, **study.best_params}
    judged = gains(surface, current, best, weights, judge)
    now = model.member_predictions(np.atleast_2d(x_now))[:, 0]
    return Advice(
        current, best, set(names) - set(bounds), now, now.mean() + judged, judged,
        profile.min_change, profile.min_gain,
    )  # fmt: skip
