"""Train the AI models from TimescaleDB history (ADR-0011: training reads only TimescaleDB).

    python -m ai.train_all [--force]

Anomaly model: fitted on clean manufacturing batches of the current recipe, the way
MSPC models describe the process as it now runs. Characterisation batches are excluded:
their setpoints are deliberately spread across the PARs. Retrains only when the set of
training batches (hence the model version) changes, unless --force.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

import psycopg

from ai import store
from ai.anomaly import train
from ai.context import controlled_by_config
from ai.offline import from_history, to_input
from common.settings import Settings, get_settings
from historian.migrate import connect

log = logging.getLogger("train")
MIN_TRAINING_BATCHES = 20

BATCHES_SQL = """
SELECT s.batch_id,
       s.payload -> 'v' ->> 'recipe'   AS recipe,
       s.payload -> 'v' ->> 'campaign' AS campaign,
       e.payload -> 'v' ->> 'status'   AS status,
       s.ts                            AS start,
       EXISTS (SELECT 1 FROM fault_labels f WHERE f.batch_id = s.batch_id) AS faulty
FROM uns_events s
JOIN uns_events e ON e.batch_id = s.batch_id AND e.payload -> 'v' ->> 'kind' = 'BATCH_END'
WHERE s.payload -> 'v' ->> 'kind' = 'BATCH_START'
ORDER BY s.ts
"""


def anomaly_training_batches(conn: psycopg.Connection) -> tuple[list[str], str]:
    rows = conn.execute(BATCHES_SQL).fetchall()
    clean = [r for r in rows if r[2] == "MFG" and r[3] == "COMPLETE" and not r[5]]
    if not clean:
        raise RuntimeError("no clean manufacturing batches in history")
    current = clean[-1][1]
    ids = [r[0] for r in clean if r[1] == current]
    if len(ids) < MIN_TRAINING_BATCHES:
        log.warning("only %d clean %s batches; using every recipe version", len(ids), current)
        ids, current = [r[0] for r in clean], "all"
    return ids, current


_settings: Settings | None = None


def _init(settings: Settings) -> None:
    global _settings
    _settings = settings


def _extract(batch_id: str) -> train.BatchInput:
    assert _settings is not None
    with connect(_settings.postgres_dsn) as conn:
        return to_input(batch_id, from_history(conn, batch_id), controlled_by_config())


def train_anomaly(conn: psycopg.Connection, settings: Settings, force: bool = False) -> bool:
    """Fit and save the anomaly model. Returns False if an up-to-date model exists."""
    ids, recipe = anomaly_training_batches(conn)
    version = train.version_for(ids, settings.seed)
    existing = store.manifest(settings.models_dir, "anomaly")
    if existing and existing.get("version") == version and not force:
        log.info("anomaly model %s is up to date", version)
        return False
    t0 = time.perf_counter()
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(settings,)) as pool:
        batches = list(pool.map(_extract, ids))
    log.info("extracted %d batches in %.0f s", len(batches), time.perf_counter() - t0)
    model = train.fit(batches, settings.seed)
    store.save(
        settings.models_dir,
        "anomaly",
        model,
        {
            "version": model.version,
            "recipe": recipe,
            "trained_on": model.trained_on,
            "thresholds": model.thresholds,
            "features": list(model.features),
        },
    )
    log.info("anomaly model %s trained on %d batches in %.0f s", model.version,
             len(ids), time.perf_counter() - t0)  # fmt: skip
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m ai.train_all")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    settings = get_settings()
    with connect(settings.postgres_dsn) as conn:
        train_anomaly(conn, settings, force=args.force)


if __name__ == "__main__":
    main()
