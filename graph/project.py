"""Attribution projection: the graph's answer to ADR-0006, flattened into TimescaleDB.

For tag X on unit U at time t:
  1. phase instances running on U at t (the batch's operations on U and their phases;
     a train batch runs on several units, ADR-0018),
  2. keep those whose phase class is BOUND_TO X, directly or through X's equipment
     module, with the binding's role (control / monitor),
  3. none left -> fall back to the operation (phase and role null).

Phase instances span their operation, so the answer changes only at operation, phase
and hold boundaries: one row per (tag, phase instance, RUNNING/HELD segment). Training
joins on `ts BETWEEN t_start AND t_end` (ADR-0011).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg

OPEN_END = datetime(9999, 12, 31, tzinfo=UTC)  # interval still open (live batch)

QUERY = """
MATCH (b:Batch {id: $batch})-[:RAN_ON]->(u:Equipment)
MATCH (b)-[:HAS_OPERATION]->(op:Operation)
WHERE op.cell = u.id OR op.cell IS NULL
MATCH (u)-[:HAS_EM]->(:EquipmentModule)-[:HAS_CM]->(:ControlModule)-[:HAS_TAG]->(t:Tag)
OPTIONAL MATCH (op)-[:HAS_PHASE]->(pi:PhaseInstance)-[:INSTANCE_OF]->(:PhaseClass)
               -[bt:BOUND_TO {unit: u.id}]->(m)
WHERE (m)-[:HAS_TAG]->(t) OR (m)-[:HAS_CM]->(:ControlModule)-[:HAS_TAG]->(t)
OPTIONAL MATCH (pi)-[:HAD_HOLD]->(h:Hold)
RETURN t.topic AS topic, op.name AS operation, op.start AS op_start, op.end AS op_end,
       pi.phase AS phase, bt.role AS role, pi.start AS pi_start, pi.end AS pi_end,
       [x IN collect(h) | [x.start, x.end]] AS holds
"""


@dataclass(frozen=True, slots=True)
class AttributionRow:
    topic: str
    batch_id: str
    operation: str
    phase: str | None
    role: str | None
    phase_state: str | None
    t_start: datetime
    t_end: datetime


def segments(
    start: datetime, end: datetime, holds: list[tuple[datetime, datetime | None]]
) -> list[tuple[str, datetime, datetime]]:
    """Split [start, end) into RUNNING and HELD pieces. An open hold runs to `end`."""
    out: list[tuple[str, datetime, datetime]] = []
    cursor = start
    for h_start, h_end in sorted(holds):
        h_start = max(h_start, cursor)
        h_stop = min(h_end or end, end)
        if h_stop <= h_start:
            continue
        if h_start > cursor:
            out.append(("RUNNING", cursor, h_start))
        out.append(("HELD", h_start, h_stop))
        cursor = h_stop
    if cursor < end:
        out.append(("RUNNING", cursor, end))
    return out


def _native(value: Any) -> datetime | None:
    if value is None:
        return None
    return value.to_native() if hasattr(value, "to_native") else value


def rows_from_records(batch_id: str, records: list[dict[str, Any]]) -> list[AttributionRow]:
    """Turn query records into attribution rows (pure; tested without Neo4j)."""
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        by_key.setdefault((r["topic"], r["operation"]), []).append(r)
    out: list[AttributionRow] = []
    for (topic, operation), recs in sorted(by_key.items()):
        op_start = _native(recs[0]["op_start"])
        op_end = _native(recs[0]["op_end"]) or OPEN_END
        bound = [r for r in recs if r["phase"] is not None]
        if not bound:
            out.append(
                AttributionRow(topic, batch_id, operation, None, None, None, op_start, op_end)
            )
            continue
        for r in bound:
            start = _native(r["pi_start"]) or op_start
            end = _native(r["pi_end"]) or op_end
            holds = [(_native(s), _native(e)) for s, e in r["holds"] if s is not None]
            for state, s, e in segments(start, end, holds):
                out.append(
                    AttributionRow(topic, batch_id, operation, r["phase"], r["role"], state, s, e)
                )
    return out


def project(session: Any, batch_id: str) -> list[AttributionRow]:
    records = [rec.data() for rec in session.run(QUERY, batch=batch_id)]
    return rows_from_records(batch_id, records)


def write(conn: psycopg.Connection, batch_id: str, rows: list[AttributionRow]) -> None:
    """Replace one batch's attribution rows."""
    with conn.transaction():
        conn.execute("DELETE FROM tag_attribution WHERE batch_id = %s", (batch_id,))
        with (
            conn.cursor() as cur,
            cur.copy(
                "COPY tag_attribution (topic, batch_id, operation, phase, role, phase_state, "
                "t_start, t_end) FROM STDIN"
            ) as cp,
        ):
            for r in rows:
                cp.write_row(
                    (r.topic, r.batch_id, r.operation, r.phase, r.role, r.phase_state,
                     r.t_start, r.t_end)
                )  # fmt: skip
