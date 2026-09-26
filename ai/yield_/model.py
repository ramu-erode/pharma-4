"""Titer models (ADR-0015): a 20-member bootstrap LightGBM ensemble (point estimate and
paired gains), P10/P50/P90 quantile models (the displayed band), a PLS baseline (the
bioprocess standard, for comparison) and TreeSHAP importances from LightGBM itself.

Every model predicts the *remaining gain*: final titer minus the titer measured so far
(zero before titer is measured). What has been measured is then not the model's problem,
and one model serves all batch days. Measured on held-out batches: 13-21% lower RMSE
than predicting final titer directly, and the error narrows with batch day.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from ai.yield_.features import FEATURES
from ai.yield_.rsm import ResponseSurface

_TITER = FEATURES.index("titer_so_far")


def measured(X: np.ndarray, column: int | None = _TITER) -> np.ndarray:
    """What is already measured of the target in each row: the bioreactor's titer so far
    (0 before the first sample). Processes whose target is only known at the end have
    no such column and predict the final value directly (ADR-0021)."""
    X = np.atleast_2d(X)
    if column is None:
        return np.zeros(len(X))
    return np.nan_to_num(X[:, column])


MEMBERS = 20
QUANTILES = (0.1, 0.5, 0.9)
PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
}
ROUNDS = 300


@dataclass
class YieldModel:
    version: str
    members: list[lgb.Booster]
    quantiles: dict[float, lgb.Booster]
    pls: PLSRegression
    pls_fill: np.ndarray  # column means for PLS, which cannot take NaN
    importance: dict[str, float]  # mean |SHAP| per feature
    metrics: dict = field(default_factory=dict)
    surface: ResponseSurface | None = None  # lever effects from the DoE (ADR-0015)
    band_scale: float = 1.0  # stretches the P10/P90 half-widths to their nominal coverage
    features: tuple[str, ...] = FEATURES
    process: str = "bioreactor"
    measured_column: int | None = _TITER

    def measured(self, X: np.ndarray) -> np.ndarray:
        return measured(X, self.measured_column)

    def member_predictions(self, X: np.ndarray) -> np.ndarray:
        """Shape (MEMBERS, n): each member's prediction of the target."""
        X = np.atleast_2d(X)
        return np.stack([b.predict(X) for b in self.members]) + self.measured(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.member_predictions(X).mean(axis=0)

    def band(self, X: np.ndarray) -> dict[float, np.ndarray]:
        X = np.atleast_2d(X)
        out = {q: b.predict(X) + self.measured(X) for q, b in self.quantiles.items()}
        mid = out[0.5]
        lo = mid - self.band_scale * np.maximum(mid - out[0.1], 0.0)  # quantile models can
        hi = mid + self.band_scale * np.maximum(out[0.9] - mid, 0.0)  # cross; keep ordered
        return {0.1: lo, 0.5: mid, 0.9: hi}

    def pls_predict(self, X: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(X), self.pls_fill, X)
        return self.pls.predict(filled).ravel() + self.measured(X)


def _booster(
    X: np.ndarray, y: np.ndarray, seed: int, features: tuple[str, ...] = FEATURES, **extra
) -> lgb.Booster:
    params = {**PARAMS, "seed": seed, **extra}
    return lgb.train(params, lgb.Dataset(X, y, feature_name=list(features)), ROUNDS)


def fit(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    version: str,
    features: tuple[str, ...] = FEATURES,
    process: str = "bioreactor",
    measured_column: int | None = _TITER,
) -> YieldModel:
    """`groups` = batch index per row; bootstrap resamples whole batches."""
    rng = np.random.default_rng(seed)
    batches = np.unique(groups)
    y = y - measured(X, measured_column)  # remaining gain (all of it, with no column)
    members = []
    for i in range(MEMBERS):
        pick = rng.choice(batches, size=len(batches), replace=True)
        rows = np.concatenate([np.nonzero(groups == b)[0] for b in pick])
        members.append(_booster(X[rows], y[rows], seed + i, features))
    # Coarser trees for the quantiles: fine ones fit the training rows' own noise and give
    # a band far narrower than the real spread (measured: 44% P10-P90 coverage).
    coarse = {"num_leaves": 7, "min_data_in_leaf": 80, "learning_rate": 0.03}
    quantiles = {
        q: _booster(X, y, seed, features, objective="quantile", alpha=q, **coarse)
        for q in QUANTILES
    }
    fill = np.nanmean(X, axis=0)
    pls = PLSRegression(n_components=5).fit(np.where(np.isnan(X), fill, X), y)
    contrib = quantiles[0.5].predict(X, pred_contrib=True)[:, : len(features)]
    importance = dict(zip(features, np.abs(contrib).mean(axis=0).tolist(), strict=True))
    return YieldModel(
        version, members, quantiles, pls, fill, importance,
        features=features, process=process, measured_column=measured_column,
    )  # fmt: skip


MODEL_REVISION = "y3"  # bump when the modelling changes (y3: DoE response surface, ADR-0015)


def version_for(
    batch_ids: list[str], seed: int, process: str = "bioreactor", features=FEATURES
) -> str:
    tag = "" if process == "bioreactor" else f"{process}:"  # the bioreactor's is unchanged
    h = hashlib.sha256(f"{MODEL_REVISION}:{tag}{seed}:{','.join(features)}".encode())
    for b in sorted(batch_ids):
        h.update(b.encode())
    return "yield-" + h.hexdigest()[:12]


def calibrate_band(model: YieldModel, X: np.ndarray, y: np.ndarray, target: float = 0.8) -> float:
    """Split-conformal: the factor on the band's half-widths that makes `target` of these
    (held-out) outcomes fall inside it."""
    b = model.band(X)
    mid = b[0.5]
    below = np.maximum(mid - b[0.1], 1e-6)
    above = np.maximum(b[0.9] - mid, 1e-6)
    need = np.where(y < mid, (mid - y) / below, (y - mid) / above)
    return float(max(1.0, np.quantile(need, target)))
