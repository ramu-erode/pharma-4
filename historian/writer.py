"""Historian service: everything in the UNS (and the fault labels) into TimescaleDB.

    python -m historian.writer

Messages are routed on the network thread and written by a flusher thread every second
or every FLUSH_ROWS rows. A failed write is retried with the same rows after reconnecting,
so a database restart loses nothing that is still in memory.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import signal
import threading
import time

import psycopg
from pydantic import BaseModel

from common import uns
from common.models import Src
from common.mqtt import connect
from common.settings import get_settings
from historian.core import Batch, Row, route, write
from historian.migrate import connect as db_connect

SERVICE = "historian"
FLUSH_S = 1.0
FLUSH_ROWS = 5000
log = logging.getLogger(SERVICE)


class Historian:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.rows: queue.Queue[Row] = queue.Queue()
        self.conn: psycopg.Connection | None = None
        self.written = 0

    def on_message(self, topic: str, payload: BaseModel | None) -> None:
        row = route(topic, payload)
        if row is not None:
            self.rows.put(row)

    def flush_forever(self, stop: threading.Event) -> None:
        pending = Batch()
        last = time.monotonic()
        while not stop.is_set() or len(pending) or not self.rows.empty():
            with contextlib.suppress(queue.Empty):
                pending.add(self.rows.get(timeout=0.1))
            due = time.monotonic() - last >= FLUSH_S or len(pending) >= FLUSH_ROWS
            if pending and (due or stop.is_set()):
                if self._write(pending):
                    self.written += len(pending)
                    pending = Batch()
                last = time.monotonic()

    def _write(self, batch: Batch) -> bool:
        try:
            if self.conn is None or self.conn.closed:
                self.conn = db_connect(self.dsn)
            write(self.conn, batch)
            return True
        except psycopg.Error:
            log.exception("write of %d rows failed; will retry", len(batch))
            if self.conn is not None:
                self.conn.close()
            self.conn = None
            time.sleep(1.0)
            return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    historian = Historian(settings.postgres_dsn)
    stop = threading.Event()
    flusher = threading.Thread(target=historian.flush_forever, args=(stop,), name="flusher")
    flusher.start()
    client = connect(SERVICE, Src.HISTORIAN, settings)
    client.subscribe(uns.SUB_UNS_ALL, historian.on_message)
    client.subscribe(uns.SUB_SIM_FAULTS, historian.on_message)
    log.info("writing to %s:%s", settings.postgres_host, settings.postgres_port)

    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while not stop.wait(30):
        log.info("%d rows written, %d queued", historian.written, historian.rows.qsize())
    client.close()
    flusher.join(timeout=10)


if __name__ == "__main__":
    main()
