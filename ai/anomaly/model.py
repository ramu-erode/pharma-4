"""The fitted anomaly model: batch-evolution alignment, PCA (T²/SPE) and Isolation Forest.

All scoring functions take matrices (one row per window), so offline evaluation scores a
whole batch at once and the live service scores a 1-row matrix with the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from ai.features import FEATURES

BIN_H = 1.0  # age bin width (hours since inoculation)
MIN_BATCHES = 8  # widen a bin's pool until it spans at least this many batches
MAX_POOL = 12  # ...but never beyond this many bins each side
MIN_STD = 1e-3
PCA_VARIANCE = 0.90


@dataclass
class AgeScaler:
    """Per operation, per batch-age bin: mean and std of every feature over clean batches
    (the standard batch-evolution approach; architecture: anomaly detection)."""

    mean: dict[str, np.ndarray] = field(default_factory=dict)  # op -> (n_bins, n_features)
    std: dict[str, np.ndarray] = field(default_factory=dict)
    first_bin: dict[str, int] = field(default_factory=dict)
    coverage: dict[str, np.ndarray] = field(default_factory=dict)  # op -> batches per bin

    def fit(self, op: str, ages_h: np.ndarray, X: np.ndarray, groups: np.ndarray) -> None:
        """`groups` says which batch each row came from. Near operation boundaries few
        batches reach a given age, so a bin pools its neighbours until it spans
        MIN_BATCHES batches (the daily feed bolus keeps most bins unpooled)."""
        bins = np.floor(ages_h / BIN_H).astype(int)
        lo, hi = int(bins.min()), int(bins.max())
        n = hi - lo + 1
        mean = np.zeros((n, X.shape[1]))
        std = np.zeros((n, X.shape[1]))
        floor = np.maximum(0.1 * X.std(axis=0), MIN_STD)
        coverage = np.zeros(n, dtype=int)
        for i in range(n):
            b = lo + i
            coverage[i] = len(np.unique(groups[bins == b]))
            for radius in range(MAX_POOL + 1):
                sel = np.abs(bins - b) <= radius
                if len(np.unique(groups[sel])) >= MIN_BATCHES:
                    break
            mean[i] = X[sel].mean(axis=0)
            std[i] = np.maximum(X[sel].std(axis=0), floor)
        self.mean[op], self.std[op], self.first_bin[op] = mean, std, lo
        self.coverage[op] = coverage

    def covered(self, op: str, ages_h: np.ndarray) -> np.ndarray:
        """Whether at least MIN_BATCHES training batches reached each age in this operation.
        Outside that (the ragged ends of an operation) there is no reference population,
        so the statistical layers do not score."""
        bins = np.floor(ages_h / BIN_H).astype(int) - self.first_bin[op]
        inside = (bins >= 0) & (bins < len(self.coverage[op]))
        out = np.zeros(len(ages_h), dtype=bool)
        out[inside] = self.coverage[op][bins[inside]] >= MIN_BATCHES
        return out

    def transform(self, op: str, ages_h: np.ndarray, X: np.ndarray) -> np.ndarray:
        mean, std = self.mean[op], self.std[op]
        idx = np.clip(np.floor(ages_h / BIN_H).astype(int) - self.first_bin[op], 0, len(mean) - 1)
        return (X - mean[idx]) / std[idx]


@dataclass
class OpModels:
    pca: PCA
    eigen: np.ndarray  # variance of each retained component
    iforest: IsolationForest

    def t2(self, Z: np.ndarray) -> np.ndarray:
        scores = self.pca.transform(Z)
        return (scores**2 / self.eigen).sum(axis=1)

    def residual(self, Z: np.ndarray) -> np.ndarray:
        return Z - self.pca.inverse_transform(self.pca.transform(Z))

    def spe(self, Z: np.ndarray) -> np.ndarray:
        return (self.residual(Z) ** 2).sum(axis=1)

    def iforest_score(self, Z: np.ndarray) -> np.ndarray:
        return -self.iforest.score_samples(Z)


def fit_op_models(Z: np.ndarray, seed: int) -> OpModels:
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full").fit(Z)
    iforest = IsolationForest(n_estimators=200, random_state=seed).fit(Z)
    return OpModels(pca=pca, eigen=pca.explained_variance_, iforest=iforest)


@dataclass
class AnomalyModel:
    version: str
    scaler: AgeScaler
    ops: dict[str, OpModels]
    thresholds: dict[str, float]  # channel -> threshold
    trained_on: list[str]
    features: tuple[str, ...] = FEATURES
