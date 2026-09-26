"""Fit and calibrate the anomaly model on clean batches (plan task 4.8).

Thresholds are cross-fitted: the clean batches are split into K folds; a model fitted
on K-1 folds scores the held-out fold, so every clean batch yields out-of-sample scores.
A channel's threshold is the THRESHOLD_QUANTILE of each batch's maximum *sustained*
score (the lower of two consecutive windows, which is what open-after-two reacts to).
A per-window percentile would fire dozens of times per batch. The final model is then
fitted on all clean batches.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from ai.anomaly.detector import AlertChange, Detector, Limit, Window, score_channels
from ai.anomaly.model import AgeScaler, AnomalyModel, fit_op_models
from ai.profiles import BIOREACTOR, AnomalyProfile

FOLDS = 5
THRESHOLD_QUANTILE = 0.97
FEATURE_VERSION = "f1"


@dataclass
class BatchInput:
    batch_id: str
    windows: list[Window]
    limits: dict[str, Limit]
    meta: dict = field(default_factory=dict)


@dataclass
class BatchResult:
    batch_id: str
    changes: list[AlertChange]
    index: np.ndarray
    raw: list[dict[str, float]]


def run(model: AnomalyModel, batch: BatchInput) -> BatchResult:
    """Run the detector over a whole batch, window by window."""
    ch = score_channels(model, batch.windows)
    det = Detector(model, batch.batch_id, batch.limits)
    changes, index, raw = [], np.zeros(len(batch.windows)), []
    for i, w in enumerate(batch.windows):
        idx, ch_changes, r = det.step(w, ch, i)
        index[i] = idx
        changes += ch_changes
        raw.append(r)
    return BatchResult(batch.batch_id, changes, index, raw)


def fit(clean: list[BatchInput], seed: int, profile: AnomalyProfile = BIOREACTOR) -> AnomalyModel:
    if len(clean) < 2 * FOLDS:
        raise ValueError(f"need at least {2 * FOLDS} clean batches to fit and calibrate")
    folds = np.random.default_rng(seed).permutation(len(clean)) % FOLDS
    maxima: dict[str, list[float]] = {}
    for k in range(FOLDS):
        partial = _fit_models(
            [b for b, f in zip(clean, folds, strict=True) if f != k], seed, profile
        )
        held_out = [b for b, f in zip(clean, folds, strict=True) if f == k]
        for c, values in sustained_maxima(partial, held_out).items():
            maxima.setdefault(c, []).extend(values)
    model = _fit_models(clean, seed, profile)
    model.thresholds = {c: float(np.quantile(v, THRESHOLD_QUANTILE)) for c, v in maxima.items()}
    return model


def _fit_models(batches: list[BatchInput], seed: int, profile: AnomalyProfile) -> AnomalyModel:
    scaler, ops = AgeScaler(), {}
    for op in profile.scored_ops:
        rows = [
            (i, w)
            for i, b in enumerate(batches)
            for w in b.windows
            if w.operation == op and not w.settling
        ]
        if not rows:
            continue
        X = np.stack([w.x for _, w in rows])
        ages = np.array([w.age_h for _, w in rows])
        scaler.fit(op, ages, X, groups=np.array([i for i, _ in rows]))
        ops[op] = fit_op_models(scaler.transform(op, ages, X), seed)
    return AnomalyModel(
        version=version_of(batches, seed, profile),
        scaler=scaler,
        ops=ops,
        thresholds={},
        trained_on=sorted(b.batch_id for b in batches),
        features=profile.features,
        cls=profile.cls,
    )


def sustained_maxima(model: AnomalyModel, batches: list[BatchInput]) -> dict[str, list[float]]:
    """Per channel, each batch's highest score held for two consecutive windows."""
    maxima: dict[str, list[float]] = {}
    for b in batches:
        result = run(model, b)  # no thresholds: channels are computed, nothing fires
        per_channel: dict[str, float] = {}
        prev: dict[str, float] = {}
        for r in result.raw:
            for c, v in r.items():
                if c in prev:
                    per_channel[c] = max(per_channel.get(c, 0.0), min(v, prev[c]))
            prev = r
        for c, v in per_channel.items():
            maxima.setdefault(c, []).append(v)
    return maxima


def version_of(batches: list[BatchInput], seed: int, profile: AnomalyProfile = BIOREACTOR) -> str:
    return version_for([b.batch_id for b in batches], seed, profile)


def version_for(batch_ids: list[str], seed: int, profile: AnomalyProfile = BIOREACTOR) -> str:
    """A model's identity: equipment class, feature set, seed and the exact training
    batches. Bootstrap retrains only when this changes."""
    h = hashlib.sha256()
    cls = "" if profile is BIOREACTOR else f"{profile.cls}:"  # the bioreactor's is unchanged
    h.update(f"{FEATURE_VERSION}:{cls}{seed}:{','.join(profile.features)}".encode())
    for batch_id in sorted(batch_ids):
        h.update(batch_id.encode())
    return "anomaly-" + h.hexdigest()[:12]
