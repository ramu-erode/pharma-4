"""One-shot, idempotent first start (ADR-0010). Takes a clean checkout to demo-ready:

1. apply TimescaleDB migrations
2. apply the Neo4j schema and load configuration (plant, bindings, tags, recipes)
3. backfill the historical batches, unless already complete
4. replay into the graph any batch the historian has and the graph lacks, and project
   its attribution (so a Neo4j reset heals itself)
5. train the AI models (anomaly, then yield) if their training data changed
6. write the models' evaluation reports (the dashboard shows them) when missing or stale

    python -m bootstrap.run

Running it again is a no-op. Services that need its results declare
`depends_on: bootstrap: condition: service_completed_successfully`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ai import train_all
from ai.anomaly import evaluate as anomaly_evaluate
from ai.yield_ import evaluate as yield_evaluate
from ai.yield_.train import train_yield
from common.settings import get_settings
from graph import db as graph_db
from graph import load, replay
from historian.migrate import connect, migrate
from simulator import backfill

log = logging.getLogger("bootstrap")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    t0 = time.perf_counter()
    with connect(settings.postgres_dsn, wait_s=120) as conn, graph_db.connect(settings) as driver:
        log.info("step 1/6: TimescaleDB migrations")
        migrate(conn)
        log.info("step 2/6: graph schema and configuration")
        graph_db.run_each(driver, load.schema_statements())
        graph_db.run_all(
            driver, load.config_statements(settings.site, settings.area, settings.line)
        )
        log.info("step 3/6: backfill (%d batches)", settings.backfill_batches)
        backfill.run(conn, settings)
        missing = replay.missing_batches(conn, driver)
        log.info("step 4/6: graph replay (%d batches missing)", len(missing))
        if missing:
            replay.replay(conn, driver, missing)
        log.info("step 5/6: model training")
        retrained = train_all.train_anomaly(conn, settings)
        retrained = train_yield(conn, settings) or retrained
        models = Path(settings.models_dir)
        reports = (models / "anomaly_eval.json", models / "yield_eval.json")
        if retrained or not all(r.exists() for r in reports):
            log.info("step 6/6: evaluation reports")
            anomaly_evaluate.evaluate(settings)
            yield_evaluate.evaluate(settings, per_recipe=4, day=4.0)
        else:
            log.info("step 6/6: evaluation reports are current")
    log.info("bootstrap complete in %.0f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
