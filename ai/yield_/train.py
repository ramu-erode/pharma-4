"""Train the yield models from TimescaleDB history (plan task 5.1-5.2, ADR-0021).

One model per process: the bioreactor's titer (g/L), the API's and the tablets' yield
(%). Rows: every non-aborted batch, at the profile's cadence (the bioreactor every half
day from day 3 to 12; the API hourly from Reaction to the end of Drying; tablets every
30 minutes through Compaction and Compression). Target: the batch's final lab result.
"Alerts so far" come from running the anomaly models over each historical batch
offline; with train/serve parity that is exactly what the live service would have
raised. Fault labels are never read.

The lever effects the optimizer uses come from a separate response surface fitted to
the clean process-characterisation runs of the process (ADR-0015, `rsm.py`).

Evaluation holds out 20% of batches (grouped, never rows of a training batch), reports
RMSE and MAPE by progress (batch day, or hours for a train process) for the ensemble,
the P50 model and the PLS baseline, and the P10-P90 coverage; then the final model is
fitted on every batch.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import psycopg

from ai import store
from ai.anomaly import train as anomaly_train
from ai.anomaly.evaluate import unit_of
from ai.offline import Collected, from_history, to_input
from ai.yield_ import profiles as yp
from ai.yield_.features import YieldInput, features_at, training_days
from ai.yield_.model import YieldModel, calibrate_band, fit, version_for
from ai.yield_.rsm import MIN_RUNS, ResponseSurface, fit_surface
from common import pools
from common.models import BatchStatus
from common.plant import get_plant
from common.settings import Settings
from historian.migrate import connect
from simulator import recipes

log = logging.getLogger("train.yield")
HOLDOUT = 0.2
MIN_BATCHES = 20

BATCHES_SQL = """
SELECT s.batch_id FROM uns_events s
JOIN uns_events e ON e.batch_id = s.batch_id AND e.payload -> 'v' ->> 'kind' = 'BATCH_END'
WHERE s.payload -> 'v' ->> 'kind' = 'BATCH_START'
  AND e.payload -> 'v' ->> 'status' <> 'ABORTED'
  AND coalesce(s.payload -> 'v' ->> 'process', 'bioreactor') = %s
ORDER BY s.ts
"""


def alerts(anomaly_model, batch_id: str, c: Collected) -> tuple[list[float], bool]:
    """Alert open times (for "alerts so far") and whether the rules layer flagged a
    process deviation (which excludes a DoE run from the response surface)."""
    if anomaly_model is None:
        return [], False
    result = anomaly_train.run(anomaly_model, to_input(batch_id, c, profile=anomaly_model.profile))
    opens = [
        ch for ch in result.changes if ch.state == "OPEN" and not ch.key.startswith("rules-quality")
    ]
    return [float(ch.end) for ch in opens], any(ch.key.startswith("rules-") for ch in opens)


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


def train_rows(process: str, inp: yp.TrainInput) -> tuple[np.ndarray, float] | None:
    target = inp.lab.get(yp.PROFILES[process].target, [])
    days = yp.train_days(process, inp)
    if not target or inp.levers is None or not days:
        return None
    X = np.stack([yp.TRAIN_FEATURES[process](inp, d) for d in days])
    return X, target[-1][1]


_settings: Settings | None = None
_anomaly: dict[str, object] = {}


def _init(settings: Settings) -> None:
    global _settings, _anomaly
    _settings = settings
    _anomaly = {}
    for u in get_plant().units.values():
        model = store.load(settings.models_dir, store.anomaly_name(u.cls))
        if model is not None:
            _anomaly[u.cls] = model


@dataclass
class Extracted:
    batch_id: str
    X: np.ndarray
    target: float
    levers: list[float]
    doe_run: bool  # a clean process-characterisation run (ADR-0015)


def _campaign(conn: psycopg.Connection, batch_id: str) -> str:
    return conn.execute(
        "SELECT payload -> 'v' ->> 'campaign' FROM uns_events WHERE batch_id = %s "
        "AND payload -> 'v' ->> 'kind' = 'BATCH_START'",
        (batch_id,),
    ).fetchone()[0]


def _extract(args: tuple[str, str]) -> Extracted | None:
    assert _settings is not None
    batch_id, process = args
    names = yp.PROFILES[process].levers
    with connect(_settings.postgres_dsn) as conn:
        campaign = _campaign(conn, batch_id)
        if process == "bioreactor":
            c = from_history(conn, batch_id)
            minutes, deviation = alerts(_anomaly.get("bioreactor"), batch_id, c)
            rows = batch_rows(c, minutes)
            levers, status = c.levers, c.status
        else:
            col = yp.train_from_history(conn, batch_id)
            minutes, deviation = [], False
            for cls in sorted({get_plant().unit(cell).cls for cell in get_plant().cells(process)}):
                if cls not in _anomaly:
                    continue
                c = from_history(conn, batch_id, unit_of(cls))
                offset = (c.series.origin - col.origin).total_seconds() / 60.0
                found, dev = alerts(_anomaly[cls], batch_id, c)
                minutes += [t + offset for t in found]
                deviation = deviation or dev
            col.inp.alert_minutes = sorted(minutes)
            rows = train_rows(process, col.inp)
            levers, status = col.inp.levers, col.status
    if rows is None:
        return None
    clean = campaign == "PC" and status is BatchStatus.COMPLETE and not deviation
    return Extracted(batch_id, rows[0], rows[1], [getattr(levers, k) for k in names], clean)


def dataset(conn: psycopg.Connection, settings: Settings, process: str = "bioreactor"):
    ids = [r[0] for r in conn.execute(BATCHES_SQL, (process,))]
    workers = max(1, (os.cpu_count() or 2) - 1)
    with pools.pool(_init, (settings,), workers) as pool:
        return [r for r in pool.map(_extract, [(b, process) for b in ids]) if r is not None]


def arrays(results: list[Extracted]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.concatenate([r.X for r in results])
    y = np.concatenate([np.full(len(r.X), r.target) for r in results])
    groups = np.concatenate([np.full(len(r.X), i) for i, r in enumerate(results)])
    return X, y, groups


def doe_surface(
    results: list[Extracted], seed: int, process: str = "bioreactor"
) -> ResponseSurface:
    runs = [r for r in results if r.doe_run]
    par = recipes.of_process(process)[-1].par  # the PARs are the same for every version
    return fit_surface(
        np.array([r.levers for r in runs]),
        np.array([r.target for r in runs]),
        par,
        seed,
        yp.PROFILES[process].levers,
        yp.PROFILES[process].surface_transform,
    )


def _fit(X, y, groups, seed, version, process) -> YieldModel:
    profile = yp.PROFILES[process]
    column = profile.features.index(profile.measured) if profile.measured else None
    return fit(X, y, groups, seed, version, profile.features, process, column)


def evaluate(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, process: str = "bioreactor"
) -> dict:
    rng = np.random.default_rng(seed)
    batches = np.unique(groups)
    test = set(rng.choice(batches, size=max(1, int(len(batches) * HOLDOUT)), replace=False))
    is_test = np.isin(groups, list(test))
    model = _fit(X[~is_test], y[~is_test], groups[~is_test], seed, "holdout", process)
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
    progress = Xt[:, 0]  # batch day for the bioreactor, hours for a train process
    if process != "bioreactor":
        progress = np.floor(progress)
    by_day = {}
    for d in sorted(set(progress.tolist())):
        sel = progress == d
        by_day[f"{d:g}"] = {
            name: {
                "rmse": float(np.sqrt(np.mean((p[sel] - yt[sel]) ** 2))),
                "mape": float(np.mean(np.abs(p[sel] - yt[sel]) / yt[sel]) * 100),
            }
            for name, p in preds.items()
        }
    return {
        "holdout_batches": len(test),
        "progress": "day" if process == "bioreactor" else "hour",
        "by_day": by_day,
        "rmse": {n: float(np.sqrt(np.mean((p - yt) ** 2))) for n, p in preds.items()},
        "p10_p90_coverage_raw": raw_coverage,
        "p10_p90_coverage": float(np.mean((yt >= band[0.1]) & (yt <= band[0.9]))),
        "band_scale": model.band_scale,
    }


def train_yield(
    conn: psycopg.Connection,
    settings: Settings,
    force: bool = False,
    process: str = "bioreactor",
) -> bool:
    """Fit and save one process's yield model. Returns False if an up-to-date model
    exists, or history is too small to train one."""
    profile = yp.PROFILES[process]
    name = store.yield_name(process)
    ids = [r[0] for r in conn.execute(BATCHES_SQL, (process,))]
    if len(ids) < MIN_BATCHES:
        log.warning("%s: %d batches, too few for a yield model; skipped", process, len(ids))
        return False
    classes = sorted({u.cls for u in get_plant().units.values() if u.process == process})
    anomaly = [
        (store.manifest(settings.models_dir, store.anomaly_name(c)) or {}).get("version", "none")
        for c in classes
    ]
    version = version_for([*ids, *anomaly], settings.seed, process, profile.features)
    existing = store.manifest(settings.models_dir, name)
    if existing and existing.get("version") == version and not force:
        log.info("%s model %s is up to date", name, version)
        return False
    t0 = time.perf_counter()
    results = dataset(conn, settings, process)
    runs = sum(r.doe_run for r in results)
    if runs < MIN_RUNS:
        log.warning("%s: %d clean DoE runs, too few for a response surface; skipped", process, runs)
        return False
    X, y, groups = arrays(results)
    batch_ids = [r.batch_id for r in results]
    log.info(
        "%s dataset: %d rows from %d batches (%.0f s)",
        process,
        len(y),
        len(batch_ids),
        time.perf_counter() - t0,
    )
    metrics = evaluate(X, y, groups, settings.seed, process)
    model = _fit(X, y, groups, settings.seed, version, process)
    model.metrics = metrics
    model.band_scale = metrics["band_scale"]
    model.surface = doe_surface(results, settings.seed, process)
    metrics["doe_runs"] = model.surface.runs
    store.save(
        settings.models_dir,
        name,
        model,
        {
            "version": version,
            "process": process,
            "target": profile.target,
            "unit": profile.unit,
            "trained_on": batch_ids,
            "rows": len(y),
            "metrics": metrics,
            "importance": model.importance,
            "anomaly_models": anomaly,
        },
    )
    log.info(
        "%s model %s: holdout RMSE ensemble %.3f, P50 %.3f, PLS %.3f %s; "
        "P10-P90 coverage %.0f%% raw, band scale %.2f (%.0f s)",
        name, version, metrics["rmse"]["ensemble"], metrics["rmse"]["p50"], metrics["rmse"]["pls"],
        profile.unit, 100 * metrics["p10_p90_coverage_raw"], metrics["band_scale"],
        time.perf_counter() - t0,
    )  # fmt: skip
    return True


def train_yield_all(conn: psycopg.Connection, settings: Settings, force: bool = False) -> bool:
    retrained = False
    for process in yp.PROFILES:
        retrained = train_yield(conn, settings, force, process) or retrained
    return retrained
