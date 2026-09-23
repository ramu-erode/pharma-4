"""Historian core: which table a message belongs in, and bulk writes. No MQTT.

- scalar values (pv, sp, lab, ai/anomaly/score)      -> tag_values (COPY)
- structured UNS messages (state, events, ai/*)       -> uns_events (jsonb)
- ground-truth labels (_sim/faults)                   -> fault_labels (upsert)
- _meta heartbeats and tag metadata                   -> not stored

Backfill writes through the same functions (ADR-0010).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

import psycopg
from pydantic import BaseModel

from common import models as m
from common import uns
from common.uns import TopicKind


@dataclass(slots=True)
class TagRow:
    ts: datetime
    topic: str
    batch_id: str | None
    value: float
    quality: str


@dataclass(slots=True)
class EventRow:
    ts: datetime
    topic: str
    batch_id: str | None
    payload: str  # JSON of the full six-key payload


@dataclass(slots=True)
class LabelRow:
    id: str
    cell: str
    batch_id: str
    fault: str
    onset: datetime
    end: datetime | None
    params: str  # JSON


Row = TagRow | EventRow | LabelRow


def route(topic: str, payload: BaseModel | None) -> Row | None:
    """The row a message becomes, or None if the historian does not keep it."""
    if payload is None:  # a retained-message clear: nothing happened in the process
        return None
    p = uns.parse(topic)
    if p.kind is TopicKind.SIM_FAULTS and isinstance(payload, m.FaultLabelPayload):
        lbl = payload.v
        return LabelRow(
            id=lbl.id,
            cell=lbl.cell,
            batch_id=lbl.batch_id,
            fault=lbl.fault.value,
            onset=lbl.onset,
            end=lbl.end,
            params=json.dumps(lbl.params),
        )
    if p.kind is not TopicKind.UNS or not isinstance(payload, m.Payload):
        return None
    if isinstance(payload, m.ScalarPayload):
        return TagRow(payload.ts, topic, payload.batch, payload.v, payload.q.value)
    return EventRow(payload.ts, topic, payload.batch, payload.model_dump_json())


@dataclass
class Batch:
    """Rows waiting to be written, grouped by table."""

    tags: list[TagRow] = field(default_factory=list)
    events: list[EventRow] = field(default_factory=list)
    labels: list[LabelRow] = field(default_factory=list)

    def add(self, row: Row) -> None:
        if isinstance(row, TagRow):
            self.tags.append(row)
        elif isinstance(row, EventRow):
            self.events.append(row)
        else:
            self.labels.append(row)

    def __len__(self) -> int:
        return len(self.tags) + len(self.events) + len(self.labels)


def write(conn: psycopg.Connection, batch: Batch) -> None:
    """Write everything in one transaction."""
    with conn.transaction():
        copy_tags(conn, batch.tags)
        insert_events(conn, batch.events)
        upsert_labels(conn, batch.labels)


def copy_tags(conn: psycopg.Connection, rows: Iterable[TagRow]) -> None:
    with (
        conn.cursor() as cur,
        cur.copy("COPY tag_values (ts, topic, batch_id, value, quality) FROM STDIN") as cp,
    ):
        for r in rows:
            cp.write_row((r.ts, r.topic, r.batch_id, r.value, r.quality))


def insert_events(conn: psycopg.Connection, rows: list[EventRow]) -> None:
    if rows:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO uns_events (ts, topic, batch_id, payload) VALUES (%s, %s, %s, %s)",
                [(r.ts, r.topic, r.batch_id, r.payload) for r in rows],
            )


def upsert_labels(conn: psycopg.Connection, rows: list[LabelRow]) -> None:
    if rows:
        with conn.cursor() as cur:
            cur.executemany(
                'INSERT INTO fault_labels (id, cell, batch_id, fault, onset, "end", params) '
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                'ON CONFLICT (id) DO UPDATE SET "end" = EXCLUDED."end"',
                [(r.id, r.cell, r.batch_id, r.fault, r.onset, r.end, r.params) for r in rows],
            )
