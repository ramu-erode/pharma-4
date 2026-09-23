"""Train the titer models from TimescaleDB history (plan task 5.1-5.2).

Rows: every non-aborted batch, every half day from day 3 to 12 (or until its harvest).
Target: the batch's final lab titer. "Alerts so far" come from running the anomaly model
over each historical batch offline; with train/serve parity that is exactly what the live
service would have raised. Fault labels are never read.

Evaluation holds out 20% of batches (grouped, never rows of a training batch), reports
RMSE and MAPE by batch day for the ensemble, the P50 model and the PLS baseline, and the
P10-P90 coverage; then the final model is fitted on every batch.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import psycopg

from ai import store
from ai.anomaly import train as anomaly_train
from ai.context import controlled_by_config
from ai.offline import Collected, from_history, to_input
from ai.yield_.features import YieldInput, features_at, training_days
from ai.yield_.model import YieldModel, calibrate_band, fit, version_for
from common import pools
from common.settings import Settings
from historian.migrate import connect

log = logging.getLogger("train.yield")
HOLDOUT = 0.2

BATCHES_SQL = """
SELECT s.batch_id FROM uns_events s
JOIN uns_events e ON e.batch_id = s.batch_id AND e.payload -> 'v' ->> 'kind' = 'BATCH_END'
WHERE s.payload -> 'v' ->> 'kind' = 'BATCH_START'
  AND e.payload -> 'v' ->> 'status' <> 'ABORTED'
ORDER BY s.ts
"""


def alert_minutes(anomaly_model, batch_id: str, c: Collected) -> list[float]:
    if anomaly_model is None:
        return []
    result = anomaly_train.run(anomaly_model, to_input(batch_id, c, controlled_by_config()))
    return [
        float(ch.end)
        for ch in result.changes
        if ch.state == "OPEN" and not ch.key.startswith("rules-quality")
    ]


def harvest_day(c: Collected) -> float | None:
    starts = {name: start for name, start, _ in c.ctx.operations}
    if "Harvest" in starts and c.ctx.inoculation_min is not None:
        return (starts["Harvest"] - c.ctx.inoculation_min) / 1440.0
    return None


def batch_rows(c: Collected, alerts: list[float]) -> tuple[np.ndarray, float] | None:
    titers = c.lab.get("titer", [])
    if not titers or c.levers is None:
        return None
    inp = YieldInput(c.series, c.ctx, c.lab, c.levers, alerts)
    X = np.stack([features_at(inp, d) for d in training_days(harvest_day(c))])
    return X, titers[-1][1]


_settings: Settings | None = None
_anomaly = None


def _init(settings: Settings) -> None:
    global _settings, _anomaly
    _settings = settings
    _anomaly = store.load(settings.models_dir, "anomaly")


def _extract(batch_id: str) -> tuple[str, np.ndarray, float] | None:
    assert _settings is not None
    with connect(_settings.postgres_dsn) as conn:
        c = from_history(conn, batch_id)
    rows = batch_rows(c, alert_minutes(_anomaly, batch_id, c))
    return None if rows is None else (batch_id, rows[0], rows[1])


def dataset(
    conn: psycopg.Connection, settings: Settings
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    ids = [r[0] for r in conn.execute(BATCHES_SQL)]
    workers = max(1, (os.cpu_count() or 2) - 1)
    with pools.pool(_init, (settings,), workers) as pool:
        results = [r for r in pool.map(_extract, ids) if r is not None]
    X = np.concatenate([r[1] for r in results])
    y = np.concatenate([np.full(len(r[1]), r[2]) for r in results])
    groups = np.concatenate([np.full(len(r[1]), i) for i, r in enumerate(results)])
    return X, y, groups, [r[0] for r in results]


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    batches = np.unique(groups)
    test = set(rng.choice(batches, size=max(1, int(len(batches) * HOLDOUT)), replace=False))
    is_test = np.isin(groups, list(test))
    model = fit(X[~is_test], y[~is_test], groups[~is_test], seed, "holdout")
    Xt, yt = X[is_test], y[is_test]
    preds = {
        "ensemble": model.predict(Xt),
        "p50": model.band(Xt)[0.5],
        "pls": model.pls_predict(Xt),
    }
    raw = model.band(Xt)
    raw_coverage = float(np.mean((yt >= raw[0.1]) & (yt <= raw[0.9])))
    model.band_scale = calibrate_band(model, Xt, yt)
    band = model.band(Xt)
    days = Xt[:, 0]
    by_day = {}
    for d in sorted(set(days.tolist())):
        sel = days == d
        by_day[f"{d:g}"] = {
            name: {
                "rmse": float(np.sqrt(np.mean((p[sel] - yt[sel]) ** 2))),
                "mape": float(np.mean(np.abs(p[sel] - yt[sel]) / yt[sel]) * 100),
            }
            for name, p in preds.items()
        }
    return {
        "holdout_batches": len(test),
        "by_day": by_day,
        "rmse": {n: float(np.sqrt(np.mean((p - yt) ** 2))) for n, p in preds.items()},
        "p10_p90_coverage_raw": raw_coverage,
        "p10_p90_coverage": float(np.mean((yt >= band[0.1]) & (yt <= band[0.9]))),
        "band_scale": model.band_scale,
    }


def train_yield(conn: psycopg.Connection, settings: Settings, force: bool = False) -> bool:
    ids = [r[0] for r in conn.execute(BATCHES_SQL)]
    anomaly = store.manifest(settings.models_dir, "anomaly") or {}
    version = version_for([*ids, anomaly.get("version", "none")], settings.seed)
    existing = store.manifest(settings.models_dir, "yield")
    if existing and existing.get("version") == version and not force:
        log.info("yield model %s is up to date", version)
        return False
    t0 = time.perf_counter()
    X, y, groups, batch_ids = dataset(conn, settings)
    log.info(
        "dataset: %d rows from %d batches (%.0f s)",
        len(y),
        len(batch_ids),
        time.perf_counter() - t0,
    )
    metrics = evaluate(X, y, groups, settings.seed)
    model: YieldModel = fit(X, y, groups, settings.seed, version)
    model.metrics = metrics
    model.band_scale = metrics["band_scale"]
    store.save(
        settings.models_dir,
        "yield",
        model,
        {
            "version": version,
            "trained_on": batch_ids,
            "rows": len(y),
            "metrics": metrics,
            "importance": model.importance,
            "anomaly_model": anomaly.get("version"),
        },
    )
    log.info(
        "yield model %s: holdout RMSE ensemble %.3f, P50 %.3f, PLS %.3f g/L; "
        "P10-P90 coverage %.0f%% raw, band scale %.2f (%.0f s)",
        version, metrics["rmse"]["ensemble"], metrics["rmse"]["p50"], metrics["rmse"]["pls"],
        100 * metrics["p10_p90_coverage_raw"], metrics["band_scale"], time.perf_counter() - t0,
    )  # fmt: skip
    return True
