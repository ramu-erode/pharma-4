"""graph-sync core: UNS message -> Cypher statements. Pure; no driver.

Subscriptions (ADR-0004, ADR-0012, ADR-0015): `_meta/tags`, `events/*`, `lab/*`,
`ai/anomaly/alert/*`, `ai/yield/recommendation`, `_sim/faults`. Never `pv`/`sp`: raw
values stay out of the graph. Every statement MERGEs on a natural key, so replaying a
message changes nothing.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from common import models as m
from common import uns
from common.levers import LEVER_TAG
from common.uns import TopicClass, TopicKind

Stmt = tuple[str, dict[str, Any]]


def cm_of_raw_tag(raw_tag: str) -> str:
    """BR101.AIC-102.PV -> AIC-102."""
    return raw_tag.split(".")[1]


def tag_statement(topic: str, raw_tag: str, unit: str, kind: str, cell: str) -> Stmt:
    p = uns.parse(topic)
    return (
        "MERGE (t:Tag {topic: $topic}) "
        "SET t.name = $name, t.class = $cls, t.unit = $unit, t.kind = $kind, "
        "t.raw_tag = $raw_tag, t.cell = $cell "
        "WITH t MATCH (cm:ControlModule {id: $cm}) MERGE (cm)-[:HAS_TAG]->(t)",
        {
            "topic": topic,
            "name": p.name,
            "cls": p.cls.value if p.cls else None,
            "unit": unit,
            "kind": kind,
            "raw_tag": raw_tag,
            "cell": cell,
            "cm": f"{cell}/{cm_of_raw_tag(raw_tag)}",
        },
    )


def handle(topic: str, payload: BaseModel | None) -> list[Stmt]:
    """Statements for one message; empty for anything the graph does not keep."""
    if payload is None:
        return []
    p = uns.parse(topic)
    if p.kind is TopicKind.META_TAG and isinstance(payload, m.TagMetaPayload):
        t = payload.v
        return [tag_statement(t.topic, t.raw_tag, t.unit, t.kind.value, p.cell)]
    if p.kind is TopicKind.SIM_FAULTS and isinstance(payload, m.FaultLabelPayload):
        return _fault_label(payload.v)
    if p.kind is not TopicKind.UNS or not isinstance(payload, m.Payload):
        return []
    unit = p.unit
    match payload:
        case m.BatchEventPayload(v=m.BatchStarted() as ev):
            return _batch_started(ev, unit.cell, payload.ts)
        case m.BatchEventPayload(v=m.OperationChanged() as ev):
            return _operation_changed(ev, payload.ts)
        case m.BatchEventPayload(v=m.PhaseChanged() as ev):
            return _phase_changed(ev, payload.ts)
        case m.BatchEventPayload(v=m.BatchEnded() as ev):
            return _batch_ended(ev, payload.ts)
        case m.OperatorEventPayload(v=ev):
            return _operator(ev, unit, payload.batch, payload.ts)
        case m.AlertPayload(v=alert):
            return _alert(alert, unit, payload.batch)
        case m.RecommendationPayload(v=rec):
            return _recommendation(rec, payload.batch, payload.ts)
        case m.ScalarPayload() if p.cls is TopicClass.LAB and payload.batch:
            return _lab(p.name, payload.v, payload.batch, payload.ts)
    return []


def projection_trigger(topic: str, payload: BaseModel | None) -> str | None:
    """The batch whose attribution must be re-projected after this message, if any."""
    if isinstance(payload, m.BatchEventPayload) and not isinstance(payload.v, m.BatchStarted):
        return payload.v.batch_id
    return None


# --- batch structure (ADR-0011) ------------------------------------------------------------


def op_id(batch_id: str, operation: m.Operation) -> str:
    return f"{batch_id}/{operation.value}"


def phase_id(batch_id: str, operation: m.Operation, phase: m.PhaseClass) -> str:
    return f"{batch_id}/{operation.value}/{phase.value}"


def _batch_started(ev: m.BatchStarted, cell: str, ts: datetime) -> list[Stmt]:
    levers = ev.planned_levers.model_dump()
    return [
        (
            "MERGE (b:Batch {id: $id}) "
            "SET b.start = $ts, b.status = 'RUNNING', b.campaign = $campaign, "
            "b.recipe = $recipe, b.cell = $cell, b += $levers, b += $planned",
            {
                "id": ev.batch_id,
                "ts": ts,
                "campaign": ev.campaign.value,
                "recipe": ev.recipe,
                "cell": cell,
                "levers": levers,
                "planned": {f"planned_{k}": v for k, v in levers.items()},
            },
        ),
        (
            "MATCH (b:Batch {id: $id}), (u:Equipment {id: $cell}) MERGE (b)-[:RAN_ON]->(u)",
            {"id": ev.batch_id, "cell": cell},
        ),
        (
            "MATCH (b:Batch {id: $id}), (r:Recipe {id: $recipe}) MERGE (b)-[:FOLLOWS]->(r)",
            {"id": ev.batch_id, "recipe": ev.recipe},
        ),
    ]


def _operation_changed(ev: m.OperationChanged, ts: datetime) -> list[Stmt]:
    out: list[Stmt] = []
    if ev.previous is not m.Operation.IDLE:
        out.append(
            (
                "MATCH (o:Operation {id: $id}) SET o.end = $ts",
                {"id": op_id(ev.batch_id, ev.previous), "ts": ts},
            )
        )
    if ev.current is not m.Operation.IDLE:
        out.append(
            (
                "MATCH (b:Batch {id: $batch}) "
                "MERGE (o:Operation {id: $id}) SET o.name = $name, o.start = $ts, "
                "o.batch_id = $batch "
                "MERGE (b)-[:HAS_OPERATION]->(o)",
                {
                    "batch": ev.batch_id,
                    "id": op_id(ev.batch_id, ev.current),
                    "name": ev.current.value,
                    "ts": ts,
                },
            )
        )
    return out


def _phase_changed(ev: m.PhaseChanged, ts: datetime) -> list[Stmt]:
    pid = phase_id(ev.batch_id, ev.operation, ev.phase)
    close_hold = (
        "MATCH (pi:PhaseInstance {id: $id})-[:HAD_HOLD]->(h:Hold) WHERE h.end IS NULL "
        "SET h.end = $ts",
        {"id": pid, "ts": ts},
    )
    if ev.state is m.PhaseState.RUNNING:
        return [
            (
                "MATCH (pc:PhaseClass {name: $phase}) "
                "MERGE (o:Operation {id: $op}) "
                "MERGE (pi:PhaseInstance {id: $id}) "
                "SET pi.phase = $phase, pi.state = 'RUNNING', pi.batch_id = $batch, "
                "pi.start = coalesce(pi.start, $ts) "
                "MERGE (o)-[:HAS_PHASE]->(pi) MERGE (pi)-[:INSTANCE_OF]->(pc)",
                {
                    "op": op_id(ev.batch_id, ev.operation),
                    "phase": ev.phase.value,
                    "id": pid,
                    "batch": ev.batch_id,
                    "ts": ts,
                },
            ),
            close_hold,
        ]
    if ev.state is m.PhaseState.HELD:
        return [
            (
                "MATCH (pi:PhaseInstance {id: $id}) SET pi.state = 'HELD' "
                "MERGE (h:Hold {id: $hold}) SET h.start = $ts "
                "MERGE (pi)-[:HAD_HOLD]->(h)",
                {"id": pid, "hold": f"{pid}/{ts.isoformat()}", "ts": ts},
            )
        ]
    return [
        close_hold,
        (
            "MATCH (pi:PhaseInstance {id: $id}) SET pi.state = 'COMPLETE', pi.end = $ts",
            {"id": pid, "ts": ts},
        ),
    ]


def _batch_ended(ev: m.BatchEnded, ts: datetime) -> list[Stmt]:
    aborted = ev.status is m.BatchStatus.ABORTED
    return [
        (
            "MATCH (b:Batch {id: $id}) SET b.end = $ts, b.status = $status, b.end_reason = $reason "
            "MERGE (o:Outcome {batch_id: $id}) MERGE (b)-[:RESULTED_IN]->(o) "
            "SET o.status = $status, o.disposition = $disposition "
            "WITH b, o "
            "OPTIONAL MATCH (b)-[:HAS_OPERATION]->(i:Operation {name: 'Inoculation'}) "
            "OPTIONAL MATCH (b)-[:HAS_OPERATION]->(h:Operation {name: 'Harvest'}) "
            "SET o.harvest_day = CASE WHEN i IS NULL OR h IS NULL THEN null "
            "ELSE duration.inSeconds(i.start, h.start).seconds / 86400.0 END, "
            "o.titer = CASE WHEN $aborted THEN null ELSE o.titer END",
            {
                "id": ev.batch_id,
                "ts": ts,
                "status": ev.status.value,
                "reason": ev.reason,
                "disposition": "REJECTED" if aborted else "ACCEPTED",
                "aborted": aborted,
            },
        )
    ]


# --- lab outcome ---------------------------------------------------------------------------


def _lab(name: str, value: float, batch_id: str, ts: datetime) -> list[Stmt]:
    if name == "vcd":
        update = (
            "o.peak_vcd = CASE WHEN o.peak_vcd IS NULL OR $v > o.peak_vcd THEN $v "
            "ELSE o.peak_vcd END"
        )
    elif name in ("titer", "viability"):
        # A rejected (aborted) batch has no titer, whatever order the messages arrive in.
        rejected = "WHEN o.disposition = 'REJECTED' THEN null " if name == "titer" else ""
        update = (
            f"o.{name} = CASE {rejected}"
            f"WHEN o.{name}_ts IS NULL OR $ts >= o.{name}_ts THEN $v "
            f"ELSE o.{name} END, "
            f"o.{name}_ts = CASE WHEN o.{name}_ts IS NULL OR $ts >= o.{name}_ts THEN $ts "
            f"ELSE o.{name}_ts END"
        )
    else:
        return []  # glucose, lactate, ph_offline are time series, not outcomes
    return [
        (
            "MATCH (b:Batch {id: $batch}) "
            "MERGE (o:Outcome {batch_id: $batch}) MERGE (b)-[:RESULTED_IN]->(o) "
            f"SET {update}",
            {"batch": batch_id, "v": value, "ts": ts},
        )
    ]


# --- events ----------------------------------------------------------------------------------


def _operator(
    ev: m.OperatorEvent, unit: uns.UnitPath, batch_id: str | None, ts: datetime
) -> list[Stmt]:
    if batch_id is None:
        return []
    eid = f"{batch_id}/operator/{ts.isoformat()}/{ev.parameter}"
    out: list[Stmt] = [
        (
            "MATCH (b:Batch {id: $batch}) "
            "MERGE (e:Event {id: $id}) SET e.type = 'operator', e.ts = $ts, "
            "e.operator = $operator, e.parameter = $param, e.old_value = $old, "
            "e.new_value = $new, e.reason = $reason "
            "MERGE (b)-[:HAS_EVENT]->(e)",
            {
                "batch": batch_id,
                "id": eid,
                "ts": ts,
                "operator": ev.operator,
                "param": ev.parameter,
                "old": ev.old,
                "new": ev.new,
                "reason": ev.reason,
            },
        )
    ]
    if ev.parameter in LEVER_TAG:
        # Whitelisted lever name, so it is safe as a property key.
        out.append(
            (
                f"MATCH (b:Batch {{id: $batch}}) SET b.{ev.parameter} = $new",
                {"batch": batch_id, "new": ev.new},
            )
        )
        cls, name = LEVER_TAG[ev.parameter]
        out.append(_on_tag(eid, uns.pv(unit, name) if cls == "pv" else uns.sp(unit, name)))
    if ev.recommendation_id:
        out.append(
            (
                "MATCH (e:Event {id: $id}) MERGE (r:Recommendation {id: $rec}) "
                "MERGE (e)-[:ACTED_ON]->(r)",
                {"id": eid, "rec": ev.recommendation_id},
            )
        )
    return out


def _alert(alert: m.Alert, unit: uns.UnitPath, batch_id: str | None) -> list[Stmt]:
    out: list[Stmt] = [
        (
            "MERGE (e:Event {id: $id}) SET e.type = 'alert', e.key = $key, e.layer = $layer, "
            "e.state = $state, e.score = $score, e.threshold = $threshold, "
            "e.fault_class = $fault_class, e.ts = $opened, e.opened_at = $opened, "
            "e.cleared_at = $cleared, e.top_tags = $top",
            {
                "id": alert.id,
                "key": alert.key,
                "layer": alert.layer,
                "state": alert.state,
                "score": alert.score,
                "threshold": alert.threshold,
                "fault_class": alert.fault_class.value if alert.fault_class else None,
                "opened": alert.opened_at,
                "cleared": alert.cleared_at,
                "top": alert.top_tags,
            },
        )
    ]
    if batch_id:
        out.append(
            (
                "MATCH (b:Batch {id: $batch}), (e:Event {id: $id}) MERGE (b)-[:HAS_EVENT]->(e)",
                {"batch": batch_id, "id": alert.id},
            )
        )
    out += [_on_tag(alert.id, uns.pv(unit, tag)) for tag in alert.top_tags]
    return out


def _recommendation(rec: m.Recommendation, batch_id: str | None, ts: datetime) -> list[Stmt]:
    out: list[Stmt] = [
        (
            "MERGE (r:Recommendation {id: $id}) SET r.ts = $ts, r.model_version = $model, "
            "r.gain_p10 = $g10, r.gain_p50 = $g50, r.gain_p90 = $g90, r.levers = $levers",
            {
                "id": rec.id,
                "ts": ts,
                "model": rec.model_version,
                "g10": rec.gain.p10,
                "g50": rec.gain.p50,
                "g90": rec.gain.p90,
                "levers": json.dumps({k: v.model_dump() for k, v in rec.levers.items()}),
            },
        )
    ]
    if batch_id:
        out.append(
            (
                "MATCH (b:Batch {id: $batch}), (r:Recommendation {id: $id}) "
                "MERGE (b)-[:HAS_RECOMMENDATION]->(r)",
                {"batch": batch_id, "id": rec.id},
            )
        )
    return out


def _fault_label(label: m.FaultLabel) -> list[Stmt]:
    """Ground truth stays apart from Events (ADR-0012)."""
    return [
        (
            "MERGE (f:FaultInjection {id: $id}) SET f.fault = $fault, f.onset = $onset, "
            "f.end = $end, f.params = $params "
            "WITH f MATCH (b:Batch {id: $batch}) MERGE (b)-[:HAS_INJECTION]->(f)",
            {
                "id": label.id,
                "fault": label.fault.value,
                "onset": label.onset,
                "end": label.end,
                "params": json.dumps(label.params),
                "batch": label.batch_id,
            },
        )
    ]


def _on_tag(event_id: str, topic: str) -> Stmt:
    return (
        "MATCH (e:Event {id: $id}), (t:Tag {topic: $topic}) MERGE (e)-[:ON_TAG]->(t)",
        {"id": event_id, "topic": topic},
    )
