"""Backfill and bootstrap against a real TimescaleDB, in a throwaway database.
Needs the compose stack's timescaledb (`pytest -m compose`)."""

from __future__ import annotations

import os

import psycopg
import pytest

from common.settings import get_settings
from historian.migrate import connect, migrate
from simulator import backfill

pytestmark = pytest.mark.compose


@pytest.fixture
def test_db():
    base = get_settings()
    name = f"pharma_test_{os.getpid()}"
    try:
        admin = psycopg.connect(base.postgres_dsn, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("TimescaleDB is not running")
    admin.execute(f'CREATE DATABASE "{name}"')
    settings = base.model_copy(update={"postgres_db": name, "backfill_batches": 3})
    try:
        yield settings
    finally:
        admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.close()


def counts(conn) -> dict[str, int]:
    return {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in ("tag_values", "uns_events", "fault_labels")
    }


def test_backfill_writes_history_and_second_run_is_a_noop(test_db):
    with connect(test_db.postgres_dsn) as conn:
        migrate(conn)
        results = backfill.run(conn, test_db)
        assert results is not None and len(results) == 3
        first = counts(conn)
        assert first["tag_values"] == sum(r.tag_rows for r in results)
        assert first["uns_events"] > 3 * 50
        ids = {r[0] for r in conn.execute("SELECT DISTINCT batch_id FROM tag_values")}
        assert ids == {r.batch_id for r in results}

        assert backfill.run(conn, test_db) is None  # complete: skipped
        assert counts(conn) == first
        assert migrate(conn) == []


def test_interrupted_backfill_is_redone_not_duplicated(test_db):
    with connect(test_db.postgres_dsn) as conn:
        migrate(conn)
        backfill.run(conn, test_db)
        first = counts(conn)
        # simulate a crash before completion was recorded
        conn.execute("DELETE FROM bootstrap_state WHERE key = 'backfill_complete'")
        backfill.run(conn, test_db)
        assert counts(conn) == first
