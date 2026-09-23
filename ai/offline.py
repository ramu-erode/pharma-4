"""Turn a batch into detector input, from the simulator (harness) or TimescaleDB (training).

Both paths produce the same thing the live service builds from the UNS: the published
(deadbanded) values per signal and the batch context from `events/batch`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import psycopg
from pydantic import BaseModel

from ai.anomaly.detector import Limit
from ai.anomaly.train import BatchInput
from ai.anomaly.windows import batch_windows
from ai.context import Context, controlled_by_config
from ai.features import Series, signal_of_topic
from common import models as m
from common.levers import actual_levers
from common.settings import Settings
from common.uns import UnitPath
from edge.core import Deadband, Mapped, TagMap, map_and_enrich
from simulator import recipes
from simulator.batches import BatchSpec
from simulator.engine import BatchRun, RawOut, TruthOut
from simulator.wire import to_wire


@dataclass
class Collected:
    series: Series
    ctx: Context
    recipe_id: str
    truth: list[TruthOut] = field(default_factory=list)
    labels: list[m.FaultLabel] = field(default_factory=list)
    status: m.BatchStatus | None = None
    lab: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    levers: m.Levers | None = None  # actual: planned + operator changes (common.levers)


def limits_for(recipe_id: str) -> dict[str, Limit]:
    """Action limits from the recipe (live, the service reads the same from the graph)."""
    return {
        name: Limit(lim.sp_relative, lim.action[0], lim.action[1])
        for name, lim in recipes.get(recipe_id).limits.items()
    }


class Collector:
    """Accumulates UNS messages of one batch into series + context."""

    def __init__(self, origin: datetime) -> None:
        self.series = Series(origin)
        self.ctx = Context()
        self.recipe_id = "v3"
        self.labels: list[m.FaultLabel] = []
        self.status: m.BatchStatus | None = None
        self.lab: dict[str, list[tuple[float, float]]] = {}
        self.planned: m.Levers | None = None
        self.operator_events: list[m.OperatorEvent] = []

    @property
    def levers(self) -> m.Levers | None:
        return actual_levers(self.planned, self.operator_events) if self.planned else None

    def collected(self, truth: list[TruthOut] | None = None) -> Collected:
        return Collected(
            self.series, self.ctx, self.recipe_id, truth or [], self.labels, self.status,
            self.lab, self.levers,
        )  # fmt: skip

    def minute(self, ts: datetime) -> float:
        return (ts - self.series.origin).total_seconds() / 60.0

    def value(self, topic: str, ts: datetime, value: float, q: str) -> None:
        sig = signal_of_topic(topic)
        if sig is not None:
            self.series.add(sig, ts, value, q != "GOOD")
        elif "/lab/" in topic:
            self.lab.setdefault(topic.rsplit("/", 1)[1], []).append((self.minute(ts), value))

    def message(self, topic: str, payload: BaseModel) -> None:
        if isinstance(payload, m.ScalarPayload):
            self.value(topic, payload.ts, payload.v, payload.q.value)
        elif isinstance(payload, m.BatchEventPayload):
            ev, t = payload.v, self.minute(payload.ts)
            match ev:
                case m.BatchStarted():
                    self.recipe_id = ev.recipe
                    self.planned = ev.planned_levers
                case m.OperationChanged():
                    self.ctx.open_operation(ev.current.value, t)
                case m.PhaseChanged(state=m.PhaseState.HELD):
                    self.ctx.hold(ev.phase.value, t)
                case m.PhaseChanged():
                    self.ctx.release(ev.phase.value, t)
                case m.BatchEnded():
                    self.status = ev.status
        elif isinstance(payload, m.OperatorEventPayload):
            self.operator_events.append(payload.v)
        elif isinstance(payload, m.FaultLabelPayload):
            self.labels = [x for x in self.labels if x.id != payload.v.id] + [payload.v]


def from_engine(spec: BatchSpec, settings: Settings, tag_map: TagMap) -> Collected:
    """Simulate a batch in-process and collect what the UNS would carry (the harness)."""
    unit = UnitPath(settings.site, settings.area, settings.line, spec.cell)
    run = BatchRun(spec, publish_period_s=settings.publish_period_s, record_truth=True)
    deadband = Deadband(settings.deadband_floor_s, settings.publish_period_s)
    batch_of_cell: dict[str, str | None] = {spec.cell: None}
    col = Collector(spec.start)
    truth = []
    for msg in run.run_to_end():
        if isinstance(msg, TruthOut):
            truth.append(msg)
            continue
        if isinstance(msg, RawOut):
            mp = map_and_enrich(msg.tag, msg.value, msg.t, msg.q, tag_map, batch_of_cell)
            if isinstance(mp, Mapped) and deadband.offer(mp):
                col.value(mp.topic, mp.ts, mp.value, mp.q.value)
            continue
        wire = to_wire(unit, spec.batch_id, msg)
        if wire is None:
            continue
        topic, payload = wire
        if topic.endswith("/state/batch"):
            batch_of_cell[spec.cell] = payload.v
        col.message(topic, payload)
    return col.collected(truth)


def from_history(conn: psycopg.Connection, batch_id: str) -> Collected:
    """A batch as the historian recorded it."""
    events = conn.execute(
        "SELECT ts, topic, payload FROM uns_events WHERE batch_id = %s ORDER BY ts, seq",
        (batch_id,),
    ).fetchall()
    start = next(
        ts
        for ts, _, p in events
        if isinstance(p["v"], dict) and p["v"].get("kind") == "BATCH_START"
    )
    col = Collector(start)
    for _, topic, p in events:
        payload = m.decode(topic, json.dumps(p).encode())
        if payload is not None:
            col.message(topic, payload)
    for ts, topic, value, q in conn.execute(
        "SELECT ts, topic, value, quality FROM tag_values WHERE batch_id = %s "
        "AND (topic LIKE %s OR topic LIKE %s OR topic LIKE %s) ORDER BY ts",
        (batch_id, "%/pv/%", "%/sp/%", "%/lab/%"),
    ):
        col.value(topic, ts, value, q)
    for fid, cell, fault, onset, end, params in conn.execute(
        'SELECT id, cell, fault, onset, "end", params FROM fault_labels WHERE batch_id = %s',
        (batch_id,),
    ):
        col.labels.append(
            m.FaultLabel(
                id=fid, fault=fault, cell=cell, batch_id=batch_id, onset=onset, end=end,
                params=params,
            )
        )  # fmt: skip
    return col.collected()


def to_input(
    batch_id: str, c: Collected, controlled: dict[str, set[str]] | None = None
) -> BatchInput:
    windows = batch_windows(c.series, c.ctx, controlled or controlled_by_config())
    return BatchInput(batch_id, windows, limits_for(c.recipe_id))
