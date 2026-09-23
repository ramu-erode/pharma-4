"""One-shot, idempotent first start (ADR-0010). Takes a clean checkout to demo-ready:

1. apply TimescaleDB migrations
2. apply the Neo4j schema and load configuration (plant, bindings, tags, recipes)
3. backfill the historical batches, unless already complete
4. replay into the graph any batch the historian has and the graph lacks, and project
   its attribution (so a Neo4j reset heals itself)
5. train the AI models if their training data changed (increment 5 adds the yield model)

    python -m bootstrap.run

Running it again is a no-op. Services that need its results declare
`depends_on: bootstrap: condition: service_completed_successfully`.
"""

from __future__ import annotations

import logging
import time

from ai import train_all
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
        log.info("step 1/5: TimescaleDB migrations")
        migrate(conn)
        log.info("step 2/5: graph schema and configuration")
        graph_db.run_each(driver, load.schema_statements())
        graph_db.run_all(
            driver, load.config_statements(settings.site, settings.area, settings.line)
        )
        log.info("step 3/5: backfill (%d batches)", settings.backfill_batches)
        backfill.run(conn, settings)
        missing = replay.missing_batches(conn, driver)
        log.info("step 4/5: graph replay (%d batches missing)", len(missing))
        if missing:
            replay.replay(conn, driver, missing)
        log.info("step 5/5: model training")
        train_all.train_anomaly(conn, settings)
    log.info("bootstrap complete in %.0f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
