"""Turn a batch into detector input, from the simulator (harness) or TimescaleDB (training).

Both paths produce the same thing the live service builds from the UNS: the published
(deadbanded) values per signal and the batch context from `events/batch`, for one unit.
A bioreactor batch lives on one unit; a train batch (ADR-0018) is collected per unit,
from when it arrived there, with that unit's equipment-class profile (ADR-0021).
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
from ai.features import Series
from ai.profiles import BIOREACTOR, AnomalyProfile, profile_for
from common import models as m
from common.levers import actual_levers
from common.plant import get_plant
from common.settings import Settings
from common.uns import UnitPath
from edge.core import Deadband, Mapped, TagMap, map_and_enrich
from simulator import recipes
from simulator.batches import BatchSpec
from simulator.messages import RawOut, TruthOut
from simulator.processes import Spec, make_run
from simulator.train import TrainRun
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
    levers: BaseModel | None = None  # actual: planned + operator changes (common.levers)
    cell: str | None = None


def limits_for(recipe_id: str) -> dict[str, Limit]:
    """Action limits from the recipe (live, the service reads the same from the graph)."""
    return {
        name: Limit(lim.sp_relative, lim.action[0], lim.action[1], lim.operations)
        for name, lim in recipes.get(recipe_id).limits.items()
    }


class Collector:
    """Accumulates UNS messages of one batch on one unit into series + context."""

    def __init__(
        self, origin: datetime, profile: AnomalyProfile = BIOREACTOR, recipe_id: str = "v3"
    ) -> None:
        self.profile = profile
        self.series = profile.series(origin)
        self.ctx = Context()
        self.recipe_id = recipe_id
        self.labels: list[m.FaultLabel] = []
        self.status: m.BatchStatus | None = None
        self.lab: dict[str, list[tuple[float, float]]] = {}
        self.planned: BaseModel | None = None
        self.operator_events: list[m.OperatorEvent] = []
        self.cell: str | None = None

    @property
    def levers(self) -> BaseModel | None:
        return actual_levers(self.planned, self.operator_events) if self.planned else None

    def collected(self, truth: list[TruthOut] | None = None) -> Collected:
        return Collected(
            self.series, self.ctx, self.recipe_id, truth or [], self.labels, self.status,
            self.lab, self.levers, self.cell,
        )  # fmt: skip

    def minute(self, ts: datetime) -> float:
        return (ts - self.series.origin).total_seconds() / 60.0

    def value(self, topic: str, ts: datetime, value: float, q: str) -> None:
        sig = self.profile.signal_of(topic)
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


def profile_of_cell(cell: str) -> AnomalyProfile | None:
    try:
        return profile_for(get_plant().unit(cell).cls)
    except KeyError:
        return None


def from_engine_units(spec: Spec, settings: Settings, tag_map: TagMap) -> dict[str, Collected]:
    """Simulate a batch in-process and collect what the UNS would carry, per scored unit
    (the harness). A unit's collection starts when the batch arrives on it."""
    plant = get_plant()
    start_unit = plant.path(spec.cell)
    run = make_run(spec, publish_period_s=settings.publish_period_s, record_truth=True)
    cells = run.cells if isinstance(run, TrainRun) else (spec.cell,)
    deadband = Deadband(settings.deadband_floor_s, settings.publish_period_s)
    batch_of_cell: dict[str, str | None] = dict.fromkeys(cells)
    cols: dict[str, Collector] = {}
    truth: dict[str, list[TruthOut]] = {}
    planned = spec.levers

    def collector(cell: str, t: datetime) -> Collector | None:
        if cell not in cols:
            profile = profile_of_cell(cell)
            if profile is None:
                return None
            col = Collector(t, profile, spec.recipe_id)
            col.planned, col.cell = planned, cell
            cols[cell] = col
        return cols[cell]

    for msg in run.run_to_end():
        cell = msg.cell or spec.cell
        if isinstance(msg, TruthOut):
            truth.setdefault(cell, []).append(msg)
            continue
        if isinstance(msg, RawOut):
            mp = map_and_enrich(msg.tag, msg.value, msg.t, msg.q, tag_map, batch_of_cell)
            if isinstance(mp, Mapped) and deadband.offer(mp):
                col = cols.get(mp.entry.unit_path.cell)
                if col is not None and (mp.batch == spec.batch_id or len(cells) == 1):
                    col.value(mp.topic, mp.ts, mp.value, mp.q.value)
            continue
        wire = to_wire(start_unit, spec.batch_id, msg)
        if wire is None:
            continue
        topic, payload = wire
        if topic.endswith("/state/batch"):
            batch_of_cell[cell] = payload.v
        col = collector(cell, msg.t)
        if col is not None:
            col.message(topic, payload)
    if isinstance(run, TrainRun):
        for col in cols.values():  # the end is published on the last unit only
            col.status = run.status
    return {c: col.collected(truth.get(c, [])) for c, col in cols.items()}


def from_engine(spec: BatchSpec, settings: Settings, tag_map: TagMap) -> Collected:
    """A bioreactor batch simulated in-process (the harness)."""
    return from_engine_units(spec, settings, tag_map)[spec.cell]


def from_history(conn: psycopg.Connection, batch_id: str, cell: str | None = None) -> Collected:
    """A batch as the historian recorded it: on its only unit (`cell` None), or its slice
    on one unit of its train, from when it arrived there."""
    events = conn.execute(
        "SELECT ts, topic, payload FROM uns_events WHERE batch_id = %s ORDER BY ts, seq",
        (batch_id,),
    ).fetchall()
    start_ev = next(
        (ts, topic, p)
        for ts, topic, p in events
        if isinstance(p["v"], dict) and p["v"].get("kind") == "BATCH_START"
    )
    start_cell = UnitPath(*start_ev[1].split("/")[1:5]).cell
    cell = cell or start_cell
    prefix = get_plant().path(cell).prefix + "/"
    mine = [(ts, topic, p) for ts, topic, p in events if topic.startswith(prefix)]
    origin = start_ev[0] if cell == start_cell else mine[0][0]
    profile = profile_of_cell(cell) or BIOREACTOR
    col = Collector(origin, profile, start_ev[2]["v"]["recipe"])
    col.cell = cell
    col.planned = m.decode(start_ev[1], json.dumps(start_ev[2]).encode()).v.planned_levers
    for _, topic, p in events:
        payload = m.decode(topic, json.dumps(p).encode())
        if payload is None:
            continue
        if topic.startswith(prefix) or isinstance(payload, m.OperatorEventPayload):
            col.message(topic, payload)
        elif isinstance(payload.v, m.BatchEnded):
            col.status = payload.v.status
    for ts, topic, value, q in conn.execute(
        "SELECT ts, topic, value, quality FROM tag_values WHERE batch_id = %s "
        "AND topic LIKE %s AND (topic LIKE %s OR topic LIKE %s OR topic LIKE %s) ORDER BY ts",
        (batch_id, prefix + "%", "%/pv/%", "%/sp/%", "%/lab/%"),
    ):
        col.value(topic, ts, value, q)
    for fid, fcell, fault, onset, end, params in conn.execute(
        'SELECT id, cell, fault, onset, "end", params FROM fault_labels '
        "WHERE batch_id = %s AND cell = %s",
        (batch_id, cell),
    ):
        col.labels.append(
            m.FaultLabel(
                id=fid, fault=fault, cell=fcell, batch_id=batch_id, onset=onset, end=end,
                params=params,
            )
        )  # fmt: skip
    return col.collected()


def to_input(
    batch_id: str,
    c: Collected,
    controlled: dict[str, set[str]] | None = None,
    profile: AnomalyProfile | None = None,
) -> BatchInput:
    profile = profile or (profile_of_cell(c.cell) if c.cell else None) or BIOREACTOR
    controlled = controlled if controlled is not None else controlled_by_config(profile.cls)
    windows = batch_windows(c.series, c.ctx, controlled, profile=profile)
    return BatchInput(batch_id, windows, limits_for(c.recipe_id))
