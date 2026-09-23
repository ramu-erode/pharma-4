"""Rebuild the graph from the historian: the UNS messages graph-sync would have seen.

`uns_events` keeps every structured message as its full payload (in arrival order),
lab results are in `tag_values`, ground-truth labels in `fault_labels`. Replaying them
through the graph-sync core gives the same graph as the live service, so the graph can
be rebuilt at any time (bootstrap after backfill, or after a Neo4j reset).

    python -m graph.replay [--all]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from collections.abc import Iterator

import psycopg
from neo4j import Driver
from pydantic import BaseModel

from common import models as m
from common import uns
from common.settings import get_settings
from graph import core, db, project
from historian.migrate import connect as pg_connect

log = logging.getLogger("graph.replay")

Message = tuple[str, BaseModel]


def history(
    conn: psycopg.Connection, batch_ids: list[str] | None = None
) -> dict[str, list[Message]]:
    """Per batch, the graph-relevant messages in arrival order."""
    where, params = ("WHERE batch_id = ANY(%s)", (batch_ids,)) if batch_ids else ("", ())
    out: dict[str, list[tuple[object, int, Message]]] = defaultdict(list)
    for ts, seq, topic, batch, payload in conn.execute(
        f"SELECT ts, seq, topic, batch_id, payload FROM uns_events {where}", params
    ):
        decoded = m.decode(topic, json.dumps(payload).encode())
        if decoded is not None and batch:
            out[batch].append((ts, 0, seq, (topic, decoded)))
    lab_where = "AND batch_id = ANY(%s)" if batch_ids else ""
    for ts, topic, batch, value, quality in conn.execute(
        "SELECT ts, topic, batch_id, value, quality FROM tag_values "
        f"WHERE topic LIKE %s AND batch_id IS NOT NULL {lab_where}",
        (f"{uns.ENTERPRISE}/%/lab/%", *params),
    ):
        p = m.ScalarPayload(v=value, ts=ts, unit=None, q=quality, batch=batch, src=m.Src.SIM)
        out[batch].append((ts, 1, 0, (topic, p)))
    for fid, cell, batch, fault, onset, end, fparams in conn.execute(
        f'SELECT id, cell, batch_id, fault, onset, "end", params FROM fault_labels {where}',
        params,
    ):
        label = m.FaultLabel(
            id=fid, fault=fault, cell=cell, batch_id=batch, onset=onset, end=end, params=fparams
        )
        payload = m.FaultLabelPayload(v=label, ts=end or onset, unit=None, batch=batch, src="sim")
        out[batch].append((end or onset, 2, 0, (uns.sim_faults(cell), payload)))
    return {b: [msg for *_, msg in sorted(v, key=lambda x: x[:3])] for b, v in sorted(out.items())}


def statements(messages: list[Message]) -> Iterator[core.Stmt]:
    for topic, payload in messages:
        yield from core.handle(topic, payload)


def replay(conn: psycopg.Connection, driver: Driver, batch_ids: list[str] | None = None) -> int:
    """Replay batches into the graph and re-project their attribution. Returns the count."""
    t0 = time.perf_counter()
    batches = history(conn, batch_ids)
    with driver.session() as session:
        for i, (batch_id, messages) in enumerate(batches.items(), start=1):
            db.run_all(driver, statements(messages))
            project.write(conn, batch_id, project.project(session, batch_id))
            if i % 50 == 0 or i == len(batches):
                log.info(
                    "replayed %d/%d batches (%.0f s)", i, len(batches), time.perf_counter() - t0
                )
    return len(batches)


def missing_batches(conn: psycopg.Connection, driver: Driver) -> list[str]:
    """Batches the historian knows (by their start event) that the graph lacks."""
    known = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT batch_id FROM uns_events "
            "WHERE payload -> 'v' ->> 'kind' = 'BATCH_START'"
        )
    }
    records, _, _ = driver.execute_query(
        "MATCH (b:Batch) WHERE b.end IS NOT NULL RETURN b.id AS id"
    )
    in_graph = {r["id"] for r in records}
    return sorted(known - in_graph)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m graph.replay")
    p.add_argument("--all", action="store_true", help="replay every batch, not only missing ones")
    args = p.parse_args()
    settings = get_settings()
    with pg_connect(settings.postgres_dsn) as conn, db.connect(settings) as driver:
        ids = None if args.all else missing_batches(conn, driver)
        if ids == []:
            log.info("graph is up to date")
            return
        log.info("replaying %s batches", "all" if ids is None else len(ids))
        replay(conn, driver, ids)


if __name__ == "__main__":
    main()
