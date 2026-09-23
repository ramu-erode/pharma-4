"""Generate the historical batches straight into the stores (ADR-0010).

    python -m simulator.backfill [--batches N] [--force]

The one sanctioned bypass of the broker. Each batch runs the same code as the live path,
in-process: engine -> edge core (map, enrich, deadband) -> historian core (COPY).
Batches are independent, so they run in parallel, one database connection per worker.

The graph is then rebuilt from what was written (graph.replay), exactly as it would be
after a Neo4j reset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg

from common.models import BatchStatus
from common.settings import Settings, get_settings
from common.uns import UnitPath
from edge.core import Deadband, Mapped, TagMap, map_and_enrich
from historian import core as hist
from historian.migrate import connect, migrate
from simulator import campaign
from simulator.batches import BatchSpec
from simulator.engine import BatchRun, RawOut
from simulator.wire import to_wire

log = logging.getLogger("backfill")


@dataclass(frozen=True, slots=True)
class BatchResult:
    batch_id: str
    status: BatchStatus
    true_titer: float
    tag_rows: int
    seconds: float


# --- per-process state -------------------------------------------------------------------

_conn: psycopg.Connection | None = None
_settings: Settings | None = None
_tag_map: TagMap | None = None


def _init_worker(settings: Settings) -> None:
    global _conn, _settings, _tag_map
    _settings = settings
    _tag_map = TagMap.load(settings.site, settings.area, settings.line)
    _conn = connect(settings.postgres_dsn)


def simulate(
    spec: BatchSpec, settings: Settings, tag_map: TagMap
) -> tuple[hist.Batch, BatchResult]:
    """Run one batch through engine, edge core and historian routing. Pure (no I/O)."""
    t0 = time.perf_counter()
    unit = UnitPath(settings.site, settings.area, settings.line, spec.cell)
    run = BatchRun(
        spec, dt_s=settings.integration_step_s, publish_period_s=settings.publish_period_s
    )
    deadband = Deadband(settings.deadband_floor_s, settings.publish_period_s)
    batch_of_cell: dict[str, str | None] = {spec.cell: None}
    rows = hist.Batch()
    for msg in run.run_to_end():
        if isinstance(msg, RawOut):
            m = map_and_enrich(msg.tag, msg.value, msg.t, msg.q, tag_map, batch_of_cell)
            if isinstance(m, Mapped) and deadband.offer(m):
                rows.tags.append(hist.TagRow(m.ts, m.topic, m.batch, m.value, m.q.value))
            continue
        wire = to_wire(unit, spec.batch_id, msg)
        if wire is None:
            continue
        topic, payload = wire
        if topic.endswith("/state/batch"):  # what the live adapter's batch cache would see
            batch_of_cell[spec.cell] = payload.v
        row = hist.route(topic, payload)
        if row is not None:
            rows.add(row)
    result = BatchResult(
        batch_id=spec.batch_id,
        status=run.status,
        true_titer=run.state.titer,
        tag_rows=len(rows.tags),
        seconds=time.perf_counter() - t0,
    )
    return rows, result


def _run_one(spec: BatchSpec) -> BatchResult:
    assert _conn is not None and _settings is not None and _tag_map is not None
    rows, result = simulate(spec, _settings, _tag_map)
    hist.write(_conn, rows)
    return result


# --- orchestration -------------------------------------------------------------------------


def get_state(conn: psycopg.Connection, key: str) -> object | None:
    row = conn.execute("SELECT value FROM bootstrap_state WHERE key = %s", (key,)).fetchone()
    return row[0] if row else None


def set_state(conn: psycopg.Connection, key: str, value: object) -> None:
    conn.execute(
        "INSERT INTO bootstrap_state (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
        (key, json.dumps(value)),
    )


def run(
    conn: psycopg.Connection, settings: Settings, force: bool = False
) -> list[BatchResult] | None:
    """Backfill unless already complete. Returns the per-batch results, or None if skipped."""
    if get_state(conn, "backfill_complete") and not force:
        log.info("backfill already complete; skipping")
        return None
    end_iso = get_state(conn, "backfill_end")
    if end_iso is None or force:
        end_iso = datetime.now(UTC).isoformat()
        set_state(conn, "backfill_end", end_iso)
    end = datetime.fromisoformat(end_iso)
    plan = campaign.plan(settings.backfill_batches, end, settings.seed)
    ids = [s.batch_id for s in plan.specs]
    _delete_history(conn, end)  # a previous attempt may have written some of it

    workers = max(1, (os.cpu_count() or 2) - 1)
    log.info("backfilling %d batches with %d workers", len(ids), workers)
    t0 = time.perf_counter()
    results: list[BatchResult] = []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(settings,)) as pool:
        for i, r in enumerate(pool.map(_run_one, plan.specs), start=1):
            results.append(r)
            if i % 20 == 0 or i == len(ids):
                log.info("%d/%d batches (%.0f s)", i, len(ids), time.perf_counter() - t0)
    _finish_storage(conn)
    set_state(conn, "backfill_complete", True)
    log.info(
        "backfill done in %.0f s: %d tag rows, statuses %s",
        time.perf_counter() - t0,
        sum(r.tag_rows for r in results),
        _count(r.status.value for r in results),
    )
    return results


def _delete_history(conn: psycopg.Connection, end: datetime) -> None:
    """Backfill owns everything before its end instant; live data only ever comes after
    first boot. Whole chunks are dropped (no decompression); the rest deleted by time."""
    conn.execute("SELECT drop_chunks('tag_values', older_than => %s::timestamptz)", (end,))
    conn.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
    conn.execute("DELETE FROM tag_values WHERE ts < %s", (end,))
    conn.execute("DELETE FROM uns_events WHERE ts < %s", (end,))
    conn.execute("DELETE FROM fault_labels WHERE onset < %s", (end,))
    conn.execute("DELETE FROM tag_attribution WHERE t_start < %s", (end,))


def _finish_storage(conn: psycopg.Connection) -> None:
    """Compress historical chunks now rather than waiting for the policy. The chart
    aggregate is not refreshed: it serves history in real-time mode (migration 002)."""
    t0 = time.perf_counter()
    conn.execute(
        "SELECT compress_chunk(c, if_not_compressed => true) "
        "FROM show_chunks('tag_values', older_than => INTERVAL '7 days') c"
    )
    log.info("compressed history in %.0f s", time.perf_counter() - t0)


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m simulator.backfill")
    p.add_argument("--batches", type=int, help="override BACKFILL_BATCHES")
    p.add_argument("--force", action="store_true", help="redo even if complete")
    args = p.parse_args()
    settings = get_settings()
    if args.batches:
        settings = settings.model_copy(update={"backfill_batches": args.batches})
    with connect(settings.postgres_dsn) as conn:
        migrate(conn)
        run(conn, settings, force=args.force)


if __name__ == "__main__":
    main()
