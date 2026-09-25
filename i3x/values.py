"""Values for the i3X façade: the broker cache, payload -> VQT, and value/history results.

A VQT is `{value, quality, timestamp}` (i3X 1.0). Timestamps are simulated plant time
(ADR-0009). Quality follows the spec's null rules: a null value is `Bad` or
`GoodNoData`, never `Good` or `Uncertain`.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel

from common import models as m
from common.models import format_ts, now_utc
from i3x.space import AddressSpace, Obj

QUALITY = {m.Quality.GOOD: "Good", m.Quality.UNCERTAIN: "Uncertain", m.Quality.BAD: "Bad"}


def to_value(payload: BaseModel) -> Any:
    """The i3X value of a UNS payload's `v` (camelCase keys, as the type schemas declare)."""
    v = payload.v
    match v:
        case None:
            return None
        case StrEnum():
            return v.value
        case str() | int() | float():
            return v
        case m.Prediction():
            return {
                "titer": v.titer.model_dump(),
                "batchDay": v.batch_day,
                "modelVersion": v.model_version,
            }
        case m.Recommendation():
            return {
                "id": v.id,
                "levers": {k: a.model_dump() for k, a in v.levers.items()},
                "predictedCurrent": v.predicted_current.model_dump(),
                "predictedRecommended": v.predicted_recommended.model_dump(),
                "gain": v.gain.model_dump(),
                "modelVersion": v.model_version,
            }
    raise TypeError(f"no i3X value for {type(v).__name__}")


def vqt(value: Any, quality: str, ts: datetime) -> dict[str, Any]:
    if isinstance(value, float) and not math.isfinite(value):
        value, quality = None, "Bad"
    if quality == "Bad" or value is None:
        value, quality = None, ("Bad" if quality == "Bad" else "GoodNoData")
    return {"value": value, "quality": quality, "timestamp": format_ts(ts)}


def payload_vqt(payload: BaseModel) -> dict[str, Any]:
    return vqt(to_value(payload), QUALITY[payload.q], payload.ts)


def is_numeric(topic: str) -> bool:
    """Scalar topics live in `tag_values`; structured ones in `uns_events`."""
    return m.model_for(topic) is m.ScalarPayload


class LiveCache:
    """The latest message per UNS topic, fed by the broker. Thread-safe.

    `now()` is the plant-time high-water mark: the stamp for values that have no
    timestamp of their own (static objects, data points that have not reported yet).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, BaseModel] = {}
        self._high_water: datetime | None = None

    def update(self, topic: str, payload: BaseModel | None) -> None:
        with self._lock:
            if payload is None:  # a retained clear, e.g. a withdrawn recommendation
                self._latest.pop(topic, None)
                return
            self._latest[topic] = payload
            ts = payload.ts
            if self._high_water is None or ts > self._high_water:
                self._high_water = ts

    def get(self, topic: str) -> BaseModel | None:
        with self._lock:
            return self._latest.get(topic)

    def now(self) -> datetime:
        with self._lock:
            return self._high_water or now_utc()


def own_vqt(obj: Obj, cache: LiveCache) -> dict[str, Any]:
    """An object's own value, without components."""
    if obj.topic is not None:
        payload = cache.get(obj.topic)
        if payload is None:
            return vqt(None, "GoodNoData", cache.now())
        return payload_vqt(payload)
    value = dict(obj.value) if isinstance(obj.value, dict) else obj.value
    for field_name, topic in obj.live.items():
        payload = cache.get(topic)
        value[field_name] = to_value(payload) if payload is not None else None
    return vqt(value, "Good", obj.ts or cache.now())


def current(space: AddressSpace, cache: LiveCache, element_id: str, depth: int) -> dict[str, Any]:
    """The `POST /objects/value` result for one element. `depth` is i3X maxDepth."""
    obj = space.objects[element_id]
    out = {"isComposition": obj.is_composition, **own_vqt(obj, cache)}
    if depth != 1 and obj.is_composition:
        out["components"] = {c: _component(space, cache, c, depth - 1) for c in obj.components}
    return out


def _component(space: AddressSpace, cache: LiveCache, element_id: str, depth: int) -> dict:
    obj = space.objects[element_id]
    out = own_vqt(obj, cache)
    if depth != 1 and obj.is_composition:
        out["components"] = {c: _component(space, cache, c, depth - 1) for c in obj.components}
    return out


# --- history ---------------------------------------------------------------------------------


class HistoryStore(Protocol):
    """Reads TimescaleDB. Returns at most `limit` rows in time order."""

    def numeric(
        self, topic: str, start: datetime, end: datetime, limit: int
    ) -> list[tuple[datetime, float, str]]: ...

    def structured(
        self, topic: str, start: datetime, end: datetime, limit: int
    ) -> list[tuple[datetime, dict[str, Any]]]: ...


def history(
    space: AddressSpace,
    store: HistoryStore,
    element_id: str,
    start: datetime,
    end: datetime,
    depth: int,
    limit: int,
    budget: Budget | None = None,
) -> tuple[dict[str, Any], bool]:
    """The `POST /objects/history` result for one element, and whether it was truncated.

    `limit` caps each element; `budget` caps the whole request, which matters for a
    composition read with `maxDepth: 0`. Only data points have history. Objects from
    the graph return an empty list: their value is a record, not a series.
    """
    budget = budget if budget is not None else Budget(limit)
    obj = space.objects[element_id]
    values: list[dict[str, Any]] = []
    truncated = False
    if obj.topic is not None:
        cap = min(limit, budget.points)
        if is_numeric(obj.topic):
            rows = store.numeric(obj.topic, start, end, cap + 1)
            values = [vqt(v, QUALITY[m.Quality(q)], ts) for ts, v, q in rows[:cap]]
        else:
            model = m.model_for(obj.topic)
            rows = store.structured(obj.topic, start, end, cap + 1)
            values = [payload_vqt(model.model_validate(p)) for _, p in rows[:cap]]
        truncated = len(rows) > cap
        budget.points -= len(values)
    out: dict[str, Any] = {"isComposition": obj.is_composition, "values": values}
    if depth != 1 and obj.is_composition:
        out["components"] = {}
        for c in obj.components:
            out["components"][c], cut = history(
                space, store, c, start, end, depth - 1, limit, budget
            )
            truncated = truncated or cut
    return out, truncated


@dataclass
class Budget:
    """History points a request may still return."""

    points: int
