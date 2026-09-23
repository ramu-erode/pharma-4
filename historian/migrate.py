"""Apply historian/migrations/*.sql in order, once each. Idempotent.

    python -m historian.migrate

Statements run one at a time in autocommit mode, because TimescaleDB refuses some DDL
(continuous aggregates) inside a transaction block. Migration files therefore must not
contain semicolons except as statement terminators.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import psycopg

from common.settings import get_settings

MIGRATIONS = Path(__file__).with_name("migrations")
log = logging.getLogger("migrate")


def statements(sql: str) -> list[str]:
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def connect(dsn: str, wait_s: float = 60.0) -> psycopg.Connection:
    """Connect, retrying while the database starts."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return psycopg.connect(dsn, autocommit=True)
        except psycopg.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1.0)


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply pending migrations; return the versions applied."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
    applied = []
    for path in sorted(MIGRATIONS.glob("*.sql")):
        version = path.stem
        if version in done:
            continue
        for stmt in statements(path.read_text()):
            conn.execute(stmt)
        conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
        applied.append(version)
        log.info("applied %s", version)
    return applied


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    with connect(get_settings().postgres_dsn) as conn:
        applied = migrate(conn)
    log.info("%d migration(s) applied", len(applied))


if __name__ == "__main__":
    main()
