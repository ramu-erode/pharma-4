"""graph-sync service: UNS context into Neo4j, attribution into TimescaleDB.

    python -m graph.sync

Messages are handled in arrival order by one worker thread. After any batch-structure
event the batch's attribution is re-projected (ADR-0011), so `tag_attribution` follows
the live batch within a message or two.
"""

from __future__ import annotations

import logging
import queue
import signal
import threading

from neo4j.exceptions import Neo4jError
from psycopg import Error as PgError
from pydantic import BaseModel

from common import uns
from common.models import Src
from common.mqtt import connect
from common.settings import get_settings
from common.uns import TopicClass
from graph import core, db, project
from historian.migrate import connect as pg_connect

SERVICE = "graph-sync"
log = logging.getLogger(SERVICE)

SUBSCRIPTIONS = [
    uns.SUB_META_TAGS,
    uns.sub_class(TopicClass.EVENTS),
    uns.sub_class(TopicClass.LAB),
    f"{uns.ENTERPRISE}/+/+/+/+/ai/anomaly/alert/+",
    f"{uns.ENTERPRISE}/+/+/+/+/ai/yield/recommendation",
    uns.SUB_SIM_FAULTS,
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    driver = db.connect(settings)
    pg = pg_connect(settings.postgres_dsn)
    inbox: queue.Queue[tuple[str, BaseModel | None]] = queue.Queue()
    stop = threading.Event()

    def work() -> None:
        handled = 0
        while not stop.is_set() or not inbox.empty():
            try:
                topic, payload = inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                db.run_all(driver, core.handle(topic, payload))
                batch = core.projection_trigger(topic, payload)
                if batch:
                    with driver.session() as session:
                        project.write(pg, batch, project.project(session, batch))
                handled += 1
                if handled % 1000 == 0:
                    log.info("%d messages synced", handled)
            except (Neo4jError, PgError):
                log.exception("failed to sync %s", topic)

    worker = threading.Thread(target=work, name="sync")
    worker.start()
    client = connect(SERVICE, Src.GRAPH_SYNC, settings)
    for pattern in SUBSCRIPTIONS:
        client.subscribe(pattern, lambda t, p: inbox.put((t, p)))
    log.info("syncing %s", ", ".join(SUBSCRIPTIONS))

    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    client.close()
    worker.join(timeout=10)
    driver.close()
    pg.close()


if __name__ == "__main__":
    main()
