"""Graph sync against the running Neo4j and TimescaleDB (`pytest -m compose`).

Uses the backfilled history: replays one batch again and checks nothing changes."""

from __future__ import annotations

import pytest

from common.settings import get_settings
from graph import db, project, replay
from historian.migrate import connect as pg_connect

pytestmark = pytest.mark.compose


@pytest.fixture(scope="module")
def stores():
    settings = get_settings()
    try:
        conn = pg_connect(settings.postgres_dsn, wait_s=2)
        driver = db.connect(settings, wait_s=2)
    except Exception:
        pytest.skip("stack is not running")
    batch = conn.execute(
        "SELECT batch_id FROM uns_events WHERE payload -> 'v' ->> 'kind' = 'BATCH_END' "
        "ORDER BY ts LIMIT 1"
    ).fetchone()
    if batch is None:
        pytest.skip("no backfilled history")
    yield conn, driver, batch[0]
    driver.close()
    conn.close()


def snapshot(conn, driver, batch_id):
    nodes, _, _ = driver.execute_query(
        "MATCH (b:Batch {id: $b})-[*1..3]->(n) RETURN count(DISTINCT n) AS n", b=batch_id
    )
    rows = conn.execute(
        "SELECT count(*) FROM tag_attribution WHERE batch_id = %s", (batch_id,)
    ).fetchone()[0]
    return nodes[0]["n"], rows


def test_replay_is_idempotent(stores):
    conn, driver, batch_id = stores
    before = snapshot(conn, driver, batch_id)
    replay.replay(conn, driver, [batch_id])
    assert snapshot(conn, driver, batch_id) == before


def test_graph_holds_the_batch_structure(stores):
    _, driver, batch_id = stores
    records, _, _ = driver.execute_query(
        "MATCH (b:Batch {id: $b})-[:HAS_OPERATION]->(o)-[:HAS_PHASE]->(pi) "
        "RETURN count(DISTINCT o) AS ops, count(pi) AS phases, b.status AS status",
        b=batch_id,
    )
    r = records[0]
    assert r["ops"] == 6 and r["phases"] == 19


def test_ph_is_controlled_by_ph_ctrl_and_monitored_by_feed_add_during_growth(stores):
    _, driver, batch_id = stores
    with driver.session() as s:
        rows = project.project(s, batch_id)
    growth_ph = {
        (r.phase, r.role) for r in rows if r.operation == "Growth" and r.topic.endswith("/pv/ph")
    }
    assert {("PH_CTRL", "control"), ("FEED_ADD", "monitor")} <= growth_ph


def test_aborted_batches_have_no_titer(stores):
    _, driver, _ = stores
    records, _, _ = driver.execute_query(
        "MATCH (b:Batch {status: 'ABORTED'})-[:RESULTED_IN]->(o) RETURN o.titer AS t, "
        "o.disposition AS d"
    )
    for r in records:
        assert r["t"] is None and r["d"] == "REJECTED"
