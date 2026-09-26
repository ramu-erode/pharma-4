"""Generate the historical batches straight into the stores (ADR-0010, ADR-0018).

    python -m simulator.backfill [--batches N] [--force]

The one sanctioned bypass of the broker. Each batch runs the same code as the live path,
in-process: engine -> edge core (map, enrich, deadband) -> historian core (COPY).
Batches are independent, so they run in parallel, one database connection per worker.
Every process's history is planned first, then numbered in start-time order across the
enterprise, as the ERP would. Tablet batches run last, because the API lots they consume
(ADR-0020) must exist first; the Freiburg stock left at the end is kept as the live
simulator's opening stock.

The graph is then rebuilt from what was written (graph.replay), exactly as it would be
after a Neo4j reset.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg

from common import models as m
from common import pools, uns
from common.models import BatchStatus
from common.plant import get_plant
from common.settings import Settings, get_settings
from edge.core import Deadband, Mapped, TagMap, map_and_enrich
from historian import core as hist
from historian.migrate import connect, migrate
from simulator import batches as b
from simulator import campaign
from simulator import train_campaign as tc
from simulator.messages import RawOut
from simulator.processes import Spec, make_run, process_of
from simulator.train import TrainRun
from simulator.wire import to_wire

log = logging.getLogger("backfill")


@dataclass(frozen=True, slots=True)
class BatchResult:
    batch_id: str
    process: str
    status: BatchStatus
    target: float  # the true yield target: titer (g/L) or yield (%)
    tag_rows: int
    seconds: float
    lot: tc.Lot | None = None  # an API lot for Freiburg (ADR-0020)


# --- per-process state -------------------------------------------------------------------

_conn: psycopg.Connection | None = None
_settings: Settings | None = None
_tag_map: TagMap | None = None


def _init_worker(settings: Settings) -> None:
    global _conn, _settings, _tag_map
    _settings = settings
    _tag_map = TagMap.load()
    _conn = connect(settings.postgres_dsn)


def simulate(spec: Spec, settings: Settings, tag_map: TagMap) -> tuple[hist.Batch, BatchResult]:
    """Run one batch through engine, edge core and historian routing. Pure (no I/O)."""
    t0 = time.perf_counter()
    unit = get_plant().path(spec.cell)
    run = make_run(
        spec, dt_s=settings.integration_step_s, publish_period_s=settings.publish_period_s
    )
    deadband = Deadband(settings.deadband_floor_s, settings.publish_period_s)
    cells = run.cells if isinstance(run, TrainRun) else (spec.cell,)
    batch_of_cell: dict[str, str | None] = dict.fromkeys(cells)
    rows = hist.Batch()
    for msg in run.run_to_end():
        if isinstance(msg, RawOut):
            mp = map_and_enrich(msg.tag, msg.value, msg.t, msg.q, tag_map, batch_of_cell)
            if isinstance(mp, Mapped) and deadband.offer(mp):
                rows.tags.append(hist.TagRow(mp.ts, mp.topic, mp.batch, mp.value, mp.q.value))
            continue
        wire = to_wire(unit, spec.batch_id, msg)
        if wire is None:
            continue
        topic, payload = wire
        if topic.endswith("/state/batch"):  # what the live adapter's batch cache would see
            batch_of_cell[uns.parse(topic).unit.cell] = payload.v
        row = hist.route(topic, payload)
        if row is not None:
            rows.add(row)
    process = process_of(spec)
    result = BatchResult(
        batch_id=spec.batch_id,
        process=process,
        status=run.status,
        target=run.target() if isinstance(run, TrainRun) else run.state.titer,
        tag_rows=len(rows.tags),
        seconds=time.perf_counter() - t0,
        lot=_lot(run) if process == "api" else None,
    )
    return rows, result


def _lot(run: TrainRun) -> tc.Lot | None:
    """A released API lot, with the true properties the tablet process depends on."""
    if run.disposition != "ACCEPTED":
        return None
    return tc.Lot(
        lot=run.spec.batch_id,
        material=run.material,
        quantity_kg=round(run.coa["product_kg"], 3),
        released=run.ts() + tc.RELEASE_DELAY,
        properties={
            "d50_um": run.coa["d50"],
            "free_sa_pct": run.coa["free_sa"],
            "assay_pct": run.coa["assay"],
        },
    )


def _run_one(spec: Spec) -> BatchResult:
    assert _conn is not None and _settings is not None and _tag_map is not None
    rows, result = simulate(spec, _settings, _tag_map)
    hist.write(_conn, rows)
    return result


def plan_history(settings: Settings, end: datetime) -> list[Spec]:
    """Every process's history, numbered B<year>-<seq> across the enterprise in start
    order (ADR-0018). Each batch keeps the seed its own process's planner gave it."""
    specs: list[Spec] = [
        *campaign.plan(settings.backfill_batches, end, settings.seed).specs,
        *tc.plan("api", settings.backfill_api_batches, end, settings.seed),
        *tc.plan("osd", settings.backfill_osd_batches, end, settings.seed),
    ]
    specs.sort(key=lambda s: (s.start, s.cell))
    return [
        dataclasses.replace(s, batch_id=b.batch_id(s.start, seq)) for seq, s in enumerate(specs)
    ]


def opening_stock(lots: list[tc.Lot]) -> m.Inventory:
    return m.Inventory(
        site="freiburg",
        lots=[
            m.LotStock(
                lot=lot.lot,
                material=lot.material,
                quantity_kg=lot.quantity_kg,
                released=lot.released,
                properties=lot.properties,
            )
            for lot in lots
        ],
    )


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
    specs = plan_history(settings, end)
    _delete_history(conn, end)  # a previous attempt may have written some of it

    workers = max(1, (os.cpu_count() or 2) - 1)
    log.info("backfilling %d batches with %d workers", len(specs), workers)
    t0 = time.perf_counter()
    first = [s for s in specs if process_of(s) != "osd"]
    tablets = [s for s in specs if process_of(s) == "osd"]
    results: list[BatchResult] = []
    with pools.pool(_init_worker, (settings,), workers) as pool:
        results += _progress(pool.map(_run_one, first), len(specs), t0)
        lots = [r.lot for r in results if r.lot is not None]
        tablets, stock = tc.allocate(tablets, lots)
        results += _progress(pool.map(_run_one, tablets), len(specs), t0, len(results))
    set_state(conn, "opening_stock", opening_stock(stock).model_dump(mode="json"))
    _finish_storage(conn)
    set_state(conn, "backfill_complete", True)
    log.info(
        "backfill done in %.0f s: %d tag rows, statuses %s",
        time.perf_counter() - t0,
        sum(r.tag_rows for r in results),
        _count(f"{r.process}:{r.status.value}" for r in results),
    )
    return results


def _progress(results, total: int, t0: float, done: int = 0) -> list[BatchResult]:
    out = []
    for i, r in enumerate(results, start=done + 1):
        out.append(r)
        if i % 40 == 0 or i == total:
            log.info("%d/%d batches (%.0f s)", i, total, time.perf_counter() - t0)
    return out


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
    p.add_argument("--batches", type=int, help="override BACKFILL_BATCHES (bioreactor)")
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
