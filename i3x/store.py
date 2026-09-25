"""History from TimescaleDB: `tag_values` for scalars, `uns_events` for structured payloads."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

import psycopg

from historian.migrate import connect


class PgHistory:
    """One connection shared by the request threads, serialised by a lock.

    A request that finds the connection broken reconnects once and retries.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._lock = threading.Lock()
        self._conn: psycopg.Connection | None = None

    def _query(self, sql: str, params: tuple) -> list[tuple]:
        with self._lock:
            for attempt in (1, 2):
                if self._conn is None or self._conn.closed:
                    self._conn = connect(self._dsn, wait_s=10)
                try:
                    return self._conn.execute(sql, params).fetchall()
                except psycopg.OperationalError:
                    self._conn = None
                    if attempt == 2:
                        raise
            raise AssertionError("unreachable")

    def numeric(
        self, topic: str, start: datetime, end: datetime, limit: int
    ) -> list[tuple[datetime, float, str]]:
        return self._query(
            "SELECT ts, value, quality FROM tag_values "
            "WHERE topic = %s AND ts >= %s AND ts <= %s ORDER BY ts LIMIT %s",
            (topic, start, end, limit),
        )

    def structured(
        self, topic: str, start: datetime, end: datetime, limit: int
    ) -> list[tuple[datetime, dict[str, Any]]]:
        return self._query(
            "SELECT ts, payload FROM uns_events "
            "WHERE topic = %s AND ts >= %s AND ts <= %s ORDER BY ts, seq LIMIT %s",
            (topic, start, end, limit),
        )
