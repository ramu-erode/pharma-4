"""One-shot, idempotent first start (ADR-0010). Takes a clean checkout to demo-ready:

1. apply TimescaleDB migrations
2. backfill the historical batches, unless already complete
   (increments 3-5 add: graph schema and config, attribution projection, training)

    python -m bootstrap.run

Running it again is a no-op. Services that need its results declare
`depends_on: bootstrap: condition: service_completed_successfully`.
"""

from __future__ import annotations

import logging
import time

from common.settings import get_settings
from historian.migrate import connect, migrate
from simulator import backfill

log = logging.getLogger("bootstrap")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    t0 = time.perf_counter()
    with connect(settings.postgres_dsn, wait_s=120) as conn:
        log.info("step 1/2: migrations")
        migrate(conn)
        log.info("step 2/2: backfill (%d batches)", settings.backfill_batches)
        backfill.run(conn, settings)
    log.info("bootstrap complete in %.0f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
