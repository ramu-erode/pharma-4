"""Train the AI models from TimescaleDB history (ADR-0011: training reads only TimescaleDB).

The yield models are trained after the anomaly models: their "alerts so far" feature
comes from running the anomaly models over history.

    python -m ai.train_all [--force]

Anomaly models, one per equipment class (ADR-0021): fitted on clean manufacturing
batches of the process's current recipe, the way MSPC models describe the process as it
now runs. Characterisation batches are excluded: their setpoints are deliberately
spread across the PARs. A class retrains only when its set of training batches (hence
its model version) changes, unless --force. A class whose history is too small is
skipped with a warning, so a small dev backfill still boots.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import psycopg

from ai import store
from ai.anomaly import train
from ai.anomaly.evaluate import unit_of
from ai.offline import from_history, to_input
from ai.profiles import PROFILES, profile_for
from ai.yield_.train import train_yield_all
from common import pools
from common.plant import get_plant
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
       EXISTS (SELECT 1 FROM fault_labels f WHERE f.batch_id = s.batch_id) AS faulty,
       coalesce(s.payload -> 'v' ->> 'process', 'bioreactor') AS process
FROM uns_events s
JOIN uns_events e ON e.batch_id = s.batch_id AND e.payload -> 'v' ->> 'kind' = 'BATCH_END'
WHERE s.payload -> 'v' ->> 'kind' = 'BATCH_START'
ORDER BY s.ts
"""


def process_of_class(cls: str) -> str:
    plant = get_plant()
    return plant.unit(plant.cells(cls=cls)[0]).process


def anomaly_training_batches(
    conn: psycopg.Connection, process: str = "bioreactor"
) -> tuple[list[str], str]:
    rows = [r for r in conn.execute(BATCHES_SQL).fetchall() if r[6] == process]
    clean = [r for r in rows if r[2] == "MFG" and r[3] == "COMPLETE" and not r[5]]
    if not clean:
        return [], "none"
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


def _extract(args: tuple[str, str]) -> train.BatchInput:
    assert _settings is not None
    batch_id, cls = args
    with connect(_settings.postgres_dsn) as conn:
        c = from_history(conn, batch_id, unit_of(cls))
    return to_input(batch_id, c, profile=profile_for(cls))


def train_anomaly(
    conn: psycopg.Connection, settings: Settings, force: bool = False, cls: str = "bioreactor"
) -> bool:
    """Fit and save one class's anomaly model. Returns False if an up-to-date model
    exists, or history is too small to train one."""
    profile = profile_for(cls)
    name = store.anomaly_name(cls)
    ids, recipe = anomaly_training_batches(conn, process_of_class(cls))
    if len(ids) < 2 * train.FOLDS:
        log.warning("%s: %d clean batches, too few for an anomaly model; skipped", cls, len(ids))
        return False
    version = train.version_for(ids, settings.seed, profile)
    existing = store.manifest(settings.models_dir, name)
    if existing and existing.get("version") == version and not force:
        log.info("%s model %s is up to date", name, version)
        return False
    t0 = time.perf_counter()
    workers = max(1, (os.cpu_count() or 2) - 1)
    with pools.pool(_init, (settings,), workers) as pool:
        batches = list(pool.map(_extract, [(b, cls) for b in ids]))
    log.info("%s: extracted %d batches in %.0f s", cls, len(batches), time.perf_counter() - t0)
    model = train.fit(batches, settings.seed, profile)
    store.save(
        settings.models_dir,
        name,
        model,
        {
            "version": model.version,
            "equipment_class": cls,
            "recipe": recipe,
            "trained_on": model.trained_on,
            "thresholds": model.thresholds,
            "features": list(model.features),
        },
    )
    log.info("%s model %s trained on %d batches in %.0f s", name, model.version,
             len(ids), time.perf_counter() - t0)  # fmt: skip
    return True


def train_anomaly_all(conn: psycopg.Connection, settings: Settings, force: bool = False) -> bool:
    """Every class's anomaly model; True if any was retrained."""
    retrained = False
    for cls in PROFILES:
        retrained = train_anomaly(conn, settings, force, cls) or retrained
    return retrained


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m ai.train_all")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    settings = get_settings()
    with connect(settings.postgres_dsn) as conn:
        train_anomaly_all(conn, settings, force=args.force)
        train_yield_all(conn, settings, force=args.force)


if __name__ == "__main__":
    main()
