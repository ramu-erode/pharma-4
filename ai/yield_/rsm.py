"""Lever effects from the designed experiment (ADR-0015).

A quadratic response surface in the five levers, coded to [-1, 1] over the PARs (the
usual DoE coding), fitted to the final titer of the clean process-characterisation runs:
COMPLETE batches with no rules-layer alert (a process deviation excludes a DoE run, as
it would in a QbD study; no fault labels are read). Huber loss down-weights what the
deviation screen misses. Bootstrap refits over the runs give the uncertainty.

Because the lever settings were randomised by design, these effects are not confounded
with recipe history the way a model fitted to all batches is. A global quadratic still
misfits the true surface far from the data, so it is only trusted for local moves
(`optimize.TRUST`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import HuberRegressor

LEVERS = ("shift_day", "prod_temp", "ph_sp", "do_sp", "feed_mult")
BOOTSTRAPS = 200
MIN_RUNS = 25  # the quadratic in 5 levers has 21 terms
ALPHA = 1e-2  # ridge penalty; heavier flattens the surface (measured: gate opened wrongly)
HARVEST_DAY, FEED_FIRST, FEED_LAST = 14.0, 3.0, 13.0


def quadratic(coded: np.ndarray) -> np.ndarray:
    """Intercept, linear, squared and two-way interaction terms."""
    c = np.atleast_2d(coded)
    k = c.shape[1]
    cols = [np.ones(len(c))] + [c[:, i] for i in range(k)] + [c[:, i] ** 2 for i in range(k)]
    cols += [c[:, i] * c[:, j] for i, j in itertools.combinations(range(k), 2)]
    return np.stack(cols, axis=1)


@dataclass
class ResponseSurface:
    par: dict[str, tuple[float, float]]
    coef: np.ndarray  # (BOOTSTRAPS, n_terms): one surface per bootstrap refit
    full: np.ndarray  # (n_terms,): the fit on every run
    runs: int

    def code(self, levers: np.ndarray) -> np.ndarray:
        lo = np.array([self.par[k][0] for k in LEVERS])
        hi = np.array([self.par[k][1] for k in LEVERS])
        return (np.atleast_2d(levers) - (lo + hi) / 2) / ((hi - lo) / 2)

    def predict(self, levers: np.ndarray, which: slice = slice(None)) -> np.ndarray:
        """Shape (n_bootstraps, n_points): final titer per bootstrap surface."""
        return self.coef[which] @ quadratic(self.code(levers)).T


def fit_surface(
    levers: np.ndarray, titer: np.ndarray, par: dict[str, tuple[float, float]], seed: int
) -> ResponseSurface:
    if len(titer) < MIN_RUNS:
        # Fewer runs than about the number of quadratic terms and every bootstrap refit
        # overfits the same way: they agree, confidently, on a wrong gain.
        raise ValueError(f"{len(titer)} DoE runs; a response surface needs at least {MIN_RUNS}")
    empty = ResponseSurface(par, np.zeros((1, 1)), np.zeros(1), len(titer))
    X = quadratic(empty.code(levers))

    def one(rows: np.ndarray) -> np.ndarray:
        m = HuberRegressor(alpha=ALPHA, max_iter=5000, fit_intercept=False)
        return m.fit(X[rows], titer[rows]).coef_

    rng = np.random.default_rng(seed)
    n = len(titer)
    coef = np.stack([one(rng.integers(0, n, n)) for _ in range(BOOTSTRAPS)])
    return ResponseSurface(par, coef, one(np.arange(n)), n)


def exposure(lever: str, day: float, shift_day: float) -> float:
    """How much of a lever's whole-batch effect a change made on `day` can still have:
    the remaining fraction of the window in which the lever acts."""
    if lever == "shift_day":
        return 1.0  # only open before the shift, when all of it still lies ahead
    if lever == "prod_temp":
        start = max(day, shift_day)
        return float(np.clip((HARVEST_DAY - start) / (HARVEST_DAY - shift_day), 0.0, 1.0))
    if lever == "feed_mult":
        return float(np.clip((FEED_LAST - max(day, FEED_FIRST)) / (FEED_LAST - FEED_FIRST), 0, 1))
    return float(np.clip((HARVEST_DAY - day) / HARVEST_DAY, 0.0, 1.0))  # ph_sp, do_sp
