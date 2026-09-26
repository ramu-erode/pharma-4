"""The assistant's tools: one per i3X call, nothing else (ADR-0017).

Each tool returns compact JSON for the model. History is downsampled on this side,
because i3X 1.0 has no aggregation and a batch holds ~20k points per tag.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from assistant.i3x_client import I3xClient, I3xError

MAX_RESULT_CHARS = 40_000
MAX_IDS = 25
MAX_HISTORY_IDS = 8


class ToolError(ValueError):
    """Bad tool input, or an i3X error the model can act on."""


def _ids(field_description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": MAX_IDS,
        "description": field_description,
    }


TOOLS: list[dict[str, Any]] = [
    {
        "name": "describe_model",
        "description": (
            "The i3X information model: namespaces, every object type with its JSON "
            "schema (what an object's value contains), and every relationship type with "
            "its reverse. Call this first if you don't yet know which types exist."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "find_objects",
        "description": (
            "Search the address space. Filter by object type (e.g. BatchType, "
            "AnomalyAlertType, ProcessValueType, RotaryTabletPressType, MaterialLotType) "
            "and/or text in the elementId or display name (case-insensitive). root_only "
            "lists the browse roots. Returns elementId, displayName, type and parent for "
            "each match."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "type_element_id": {"type": "string", "description": "Object type elementId"},
                "text": {"type": "string", "description": "Substring of elementId or name"},
                "root_only": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 300, "default": 50},
            },
        },
    },
    {
        "name": "describe_objects",
        "description": (
            "Object records with metadata: description, type provenance, and every "
            "relationship as elementIds (HasChildren, HasComponent, RanOn, ConcernsTag, "
            "Controls, Monitors, ...). No values; use read_values for those."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"element_ids": _ids("Objects to describe")},
            "required": ["element_ids"],
        },
    },
    {
        "name": "get_related",
        "description": (
            "Objects related to the given ones, optionally through one relationship type "
            "only (e.g. HasChildren of a batch = its alerts, operator actions and "
            "recommendations; ConcernedBy of a tag = alerts that implicated it)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "element_ids": _ids("Objects to start from"),
                "relationship_type": {"type": "string", "description": "e.g. HasChildren"},
            },
            "required": ["element_ids"],
        },
    },
    {
        "name": "read_values",
        "description": (
            "Current value, quality and plant-time timestamp of objects. max_depth > 1 "
            "also returns component values (a unit with max_depth 0 returns its "
            "whole live state). A batch's value holds its recipe, levers, outcome and "
            "operation/phase timeline."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "element_ids": _ids("Objects to read"),
                "max_depth": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                    "default": 1,
                    "description": "0 = all components, 1 = the object only",
                },
            },
            "required": ["element_ids"],
        },
    },
    {
        "name": "read_history",
        "description": (
            "Historical values of data points (tags, lab results, state, AI outputs) "
            "between two plant-time instants (RFC 3339, UTC). Long numeric series come "
            "back as time buckets with min/mean/max; structured series as their changes. "
            "Use a batch's start and end from its value as the window."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "element_ids": {**_ids("Data points"), "maxItems": MAX_HISTORY_IDS},
                "start_time": {"type": "string", "description": "e.g. 2026-03-01T00:00:00Z"},
                "end_time": {"type": "string"},
                "max_points": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 1000,
                    "default": 150,
                    "description": "Most points or buckets returned per series",
                },
            },
            "required": ["element_ids", "start_time", "end_time"],
        },
    },
]


def run(client: I3xClient, name: str, args: dict[str, Any]) -> str:
    """Execute one tool call. Raises ToolError with a message the model can act on."""
    handler = _HANDLERS.get(name)
    if handler is None:
        raise ToolError(f"unknown tool {name!r}")
    try:
        result = handler(client, args)
    except I3xError as exc:
        raise ToolError(str(exc)) from exc
    text = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
    if len(text) > MAX_RESULT_CHARS:
        raise ToolError(
            f"result too large ({len(text)} characters). Narrow the request: fewer "
            "element_ids, a shorter time window, a lower max_points or max_depth 1."
        )
    return text


# --- handlers ---------------------------------------------------------------------------------


def _describe_model(client: I3xClient, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "objectTypes": [
            {"elementId": t["elementId"], "displayName": t["displayName"], "schema": t["schema"]}
            for t in client.object_types()
        ],
        "relationshipTypes": [
            {"elementId": r["elementId"], "reverseOf": r["reverseOf"]}
            for r in client.relationship_types()
        ],
    }


def _find_objects(client: I3xClient, args: dict[str, Any]) -> dict[str, Any]:
    text = str(args.get("text") or "").lower()
    limit = _int(args, "limit", 50, 1, 300)
    found = [
        o
        for o in client.objects(args.get("type_element_id"), bool(args.get("root_only")))
        if text in o["elementId"].lower() or text in o["displayName"].lower()
    ]
    return {
        "total": len(found),
        "shown": min(len(found), limit),
        "objects": [
            {
                "elementId": o["elementId"],
                "displayName": o["displayName"],
                "type": o["typeElementId"],
                "parentId": o["parentId"],
            }
            for o in found[:limit]
        ],
    }


def _describe_objects(client: I3xClient, args: dict[str, Any]) -> list[dict[str, Any]]:
    return [_unwrap(r) for r in client.list(_id_list(args, MAX_IDS), metadata=True)]


def _get_related(client: I3xClient, args: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in client.related(_id_list(args, MAX_IDS), args.get("relationship_type")):
        if not r["success"]:
            out.append(_unwrap(r))
            continue
        out.append(
            {
                "elementId": r["elementId"],
                "related": [
                    {
                        "relationship": e["sourceRelationship"],
                        "elementId": e["object"]["elementId"],
                        "displayName": e["object"]["displayName"],
                        "type": e["object"]["typeElementId"],
                    }
                    for e in r["result"]
                ],
            }
        )
    return out


def _read_values(client: I3xClient, args: dict[str, Any]) -> list[dict[str, Any]]:
    depth = _int(args, "max_depth", 1, 0, 4)
    return [_unwrap(r) for r in client.value(_id_list(args, MAX_IDS), depth)]


def _read_history(client: I3xClient, args: dict[str, Any]) -> dict[str, Any]:
    ids = _id_list(args, MAX_HISTORY_IDS)
    start, end = _time(args, "start_time"), _time(args, "end_time")
    if start >= end:
        raise ToolError("start_time must be before end_time")
    max_points = _int(args, "max_points", 150, 10, 1000)
    results, note = client.history(ids, start, end)
    series = []
    for r in results:
        if not r["success"]:
            series.append(_unwrap(r))
            continue
        series.append({"elementId": r["elementId"], **summarise(r["result"]["values"], max_points)})
    out: dict[str, Any] = {"series": series}
    if note:
        out["serverNote"] = note
    return out


_HANDLERS = {
    "describe_model": _describe_model,
    "find_objects": _find_objects,
    "describe_objects": _describe_objects,
    "get_related": _get_related,
    "read_values": _read_values,
    "read_history": _read_history,
}


# --- helpers ---------------------------------------------------------------------------------


def summarise(values: list[dict[str, Any]], max_points: int) -> dict[str, Any]:
    """Fit a series into `max_points`: numbers as time buckets, the rest as changes."""
    n = len(values)
    if n and all(isinstance(v["value"], int | float) for v in values if v["value"] is not None):
        numeric = [v for v in values if v["value"] is not None]
        bad = n - len(numeric)
        if len(numeric) <= max_points:
            out = {
                "points": n,
                "values": [[v["timestamp"], _round(v["value"])] for v in numeric],
            }
        else:
            out = {"points": n, "buckets": _buckets(numeric, max_points)}
        if bad:
            out["nullPoints"] = bad
        return out
    changes = [v for i, v in enumerate(values) if i == 0 or v["value"] != values[i - 1]["value"]]
    if len(changes) > max_points:
        step = len(changes) / max_points
        changes = [changes[int(i * step)] for i in range(max_points - 1)] + [changes[-1]]
        return {"points": n, "sampledChanges": [_compact(v) for v in changes]}
    return {"points": n, "changes": [_compact(v) for v in changes]}


def _buckets(values: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    times = [_time({"t": v["timestamp"]}, "t").timestamp() for v in values]
    t0, t1 = times[0], times[-1]
    width = (t1 - t0) / count or 1.0
    groups: dict[int, list[float]] = {}
    for t, v in zip(times, values, strict=True):
        groups.setdefault(min(int((t - t0) / width), count - 1), []).append(v["value"])
    return [
        {
            "start": datetime.fromtimestamp(t0 + i * width, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "min": _round(min(g)),
            "mean": _round(sum(g) / len(g)),
            "max": _round(max(g)),
            "n": len(g),
        }
        for i, g in sorted(groups.items())
    ]


def _compact(v: dict[str, Any]) -> dict[str, Any]:
    out = {"t": v["timestamp"], "value": v["value"]}
    if v["quality"] != "Good":
        out["quality"] = v["quality"]
    return out


def _round(x: float) -> float:
    return float(f"{x:.5g}")


def _unwrap(r: dict[str, Any]) -> dict[str, Any]:
    if r["success"]:
        return {"elementId": r["elementId"], **_as_dict(r["result"])}
    return {"elementId": r["elementId"], "error": r["responseDetail"]["detail"]}


def _as_dict(result: Any) -> dict[str, Any]:
    return result if isinstance(result, dict) else {"result": result}


def _id_list(args: dict[str, Any], limit: int) -> list[str]:
    ids = args.get("element_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        raise ToolError("element_ids must be a non-empty list of elementId strings")
    if len(ids) > limit:
        raise ToolError(f"at most {limit} element_ids per call")
    return ids


def _int(args: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    value = args.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
        raise ToolError(f"{key} must be an integer from {lo} to {hi}")
    return value


def _time(args: dict[str, Any], key: str) -> datetime:
    raw = args.get(key)
    try:
        t = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ToolError(f"{key} must be an RFC 3339 timestamp, got {raw!r}") from exc
    return t if t.tzinfo else t.replace(tzinfo=UTC)
