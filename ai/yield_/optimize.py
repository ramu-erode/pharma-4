"""Setpoint recommendation (ADR-0014). Pure: no MQTT.

Only the levers whose window is still open are searched (Optuna TPE, inside the recipe's
PARs); the observed trajectory stays fixed. A recommendation is made only if the
ensemble agrees the gain is real: the paired gain (each member's recommended minus its
current prediction, so shared error cancels) must have P10 > 0 and median >= MIN_GAIN.

The ensemble is split: half the members drive the search, the other half judge the
winner. Picking the best of many candidates with the same models that then score it
inflates the gain (the optimizer's curse); measured against the simulator's ground truth,
that inflation opened the gate for gains that did not exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna

from ai.yield_.features import LEVERS, with_levers
from ai.yield_.model import YieldModel, measured
from common.models import Levers

MIN_GAIN = 0.1  # g/L: smallest gain worth a person's attention
TRIALS = 150
TRUST = 0.25  # max move per recommendation, as a fraction of the PAR width
SHIFT_LEAD_DAYS = 0.25  # a shift can be moved to no sooner than 6 h from now
MIN_CHANGE = {"shift_day": 0.1, "prod_temp": 0.1, "ph_sp": 0.01, "do_sp": 1.0, "feed_mult": 0.02}

optuna.logging.set_verbosity(optuna.logging.WARNING)


def open_levers(day: float, operation: str | None, levers: Levers) -> set[str]:
    """Levers that can still change the outcome at batch day `day`."""
    out = {"prod_temp", "ph_sp", "do_sp", "feed_mult"}
    if operation in ("Setup", "Inoculation", "Growth"):  # not shifted yet
        out.add("shift_day")
    return out


@dataclass
class Advice:
    current: dict[str, float]
    recommended: dict[str, float]
    frozen: set[str]
    predicted_current: np.ndarray  # member predictions
    predicted_recommended: np.ndarray
    gain: np.ndarray  # paired, per member

    @property
    def passes_gate(self) -> bool:
        changed = any(
            abs(self.recommended[k] - self.current[k]) >= MIN_CHANGE[k] for k in self.recommended
        )
        return (
            changed
            and float(np.quantile(self.gain, 0.1)) > 0
            and float(np.median(self.gain)) >= MIN_GAIN
        )


def recommend(
    model: YieldModel,
    x_now: np.ndarray,
    levers: Levers,
    day: float,
    operation: str | None,
    par: dict[str, tuple[float, float]],
    seed: int,
    trials: int = TRIALS,
) -> Advice:
    x_now = np.atleast_2d(x_now)
    current = {k: float(getattr(levers, k)) for k in LEVERS}
    open_ = open_levers(day, operation, levers)
    # Trust region: move each lever at most TRUST of its PAR width from where it is, so the
    # search stays near inputs the model has seen (it holds the trajectory fixed, ADR-0014).
    bounds = {}
    for k in open_:
        lo, hi = par[k]
        step = TRUST * (hi - lo)
        bounds[k] = (max(lo, current[k] - step), min(hi, current[k] + step))
    if "shift_day" in bounds:
        lo, hi = bounds["shift_day"]
        bounds["shift_day"] = (max(lo, day + SHIFT_LEAD_DAYS), hi)
        if bounds["shift_day"][0] >= hi:
            del bounds["shift_day"]

    half = len(model.members) // 2
    search, judge = model.members[:half], model.members[half:]

    def objective(trial: optuna.Trial) -> float:
        cand = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in bounds.items()}
        x = with_levers(x_now, cand)
        return float(np.mean([b.predict(x)[0] for b in search]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({k: min(max(current[k], lo), hi) for k, (lo, hi) in bounds.items()})
    study.optimize(objective, n_trials=trials)
    best = {**current, **study.best_params}
    x_best = with_levers(x_now, study.best_params)
    now = np.array([b.predict(x_now)[0] for b in judge]) + measured(x_now)[0]
    rec = np.array([b.predict(x_best)[0] for b in judge]) + measured(x_best)[0]
    return Advice(current, best, set(LEVERS) - set(bounds), now, rec, rec - now)
