"""Read the i3X catalog from Neo4j, and keep a briefly cached address space.

Every query matches explicit labels on the plant model and the batch context. None
reads ground truth (ADR-0012): whatever i3X serves, an LLM may see (ADR-0017).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from neo4j import Driver

from common.models import Levers
from i3x.space import AddressSpace, Catalog, build

log = logging.getLogger(__name__)

PLANT = (
    "MATCH (s:Site)-[:HAS_AREA]->(a:Area)-[:HAS_LINE]->(l:Line) "
    "RETURN s{.id, .name} AS site, a{.id, .name} AS area, l{.id, .name} AS line"
)
UNITS = (
    "MATCH (:Line)-[:HAS_UNIT]->(u:Equipment) "
    "RETURN u.id AS id, u.type AS type, u.working_volume_l AS working_volume_l ORDER BY id"
)
MODULES = (
    "MATCH (u:Equipment)-[:HAS_EM]->(em:EquipmentModule) "
    "RETURN em.id AS id, u.id AS unit, em.module AS module, em.name AS name, null AS parent "
    "UNION ALL "
    "MATCH (u:Equipment)-[:HAS_EM]->(em:EquipmentModule)-[:HAS_CM]->(cm:ControlModule) "
    "RETURN cm.id AS id, u.id AS unit, cm.module AS module, null AS name, em.id AS parent"
)
TAGS = (
    "MATCH (cm:ControlModule)-[:HAS_TAG]->(t:Tag) "
    "RETURN t.topic AS topic, cm.id AS cm, t.cell AS cell, t.unit AS unit, t.kind AS kind, "
    "t.raw_tag AS raw_tag ORDER BY topic"
)
SENSORS = (
    "MATCH (u:Equipment)-[:HAS_SENSOR]->(s:Sensor) OPTIONAL MATCH (s)-[:MEASURES]->(t:Tag) "
    "RETURN s.id AS id, u.id AS unit, s.model AS model, "
    "s.calibration_interval_days AS calibration_interval_days, collect(t.topic) AS measures "
    "ORDER BY id"
)
BINDINGS = (
    "MATCH (pc:PhaseClass)-[r:BOUND_TO]->(mod) "
    "WHERE mod:EquipmentModule OR mod:ControlModule "
    "RETURN pc.name AS phase, mod.id AS module, r.role AS role ORDER BY phase, module"
)
RECIPES = (
    "MATCH (r:Recipe) OPTIONAL MATCH (r)-[:HAS_LIMIT]->(l:SpecLimit) "
    "RETURN r AS recipe, collect(l{.parameter, .type, .low, .high, .sp_relative}) AS limits "
    "ORDER BY r.id"
)
BATCHES = (
    "MATCH (b:Batch) "
    "OPTIONAL MATCH (b)-[:RESULTED_IN]->(o:Outcome) "
    "OPTIONAL MATCH (b)-[:HAS_OPERATION]->(op:Operation) "
    "OPTIONAL MATCH (op)-[:HAS_PHASE]->(pi:PhaseInstance) "
    "OPTIONAL MATCH (pi)-[:HAD_HOLD]->(h:Hold) "
    "WITH b, o, op, pi, count(h) AS holds "
    "WITH b, o, op, collect(pi{.phase, .state, .start, .end, holds: holds}) AS phases "
    "WITH b, o, collect(op{.name, .start, .end, phases: phases}) AS operations "
    "RETURN b AS batch, o AS outcome, operations"
)
ALERTS = (
    "MATCH (b:Batch)-[:HAS_EVENT]->(e:Event {type: 'alert'}) "
    "OPTIONAL MATCH (e)-[:ON_TAG]->(t:Tag) "
    "RETURN e AS event, b.id AS batch, collect(t.topic) AS tags"
)
ACTIONS = (
    "MATCH (b:Batch)-[:HAS_EVENT]->(e:Event {type: 'operator'}) "
    "OPTIONAL MATCH (e)-[:ON_TAG]->(t:Tag) "
    "OPTIONAL MATCH (e)-[:ACTED_ON]->(r:Recommendation) "
    "RETURN e AS event, b.id AS batch, collect(DISTINCT t.topic) AS tags, "
    "head(collect(r.id)) AS recommendation"
)
RECOMMENDATIONS = (
    "MATCH (b:Batch)-[:HAS_RECOMMENDATION]->(r:Recommendation) RETURN r AS rec, b.id AS batch"
)


def native(value: Any) -> Any:
    """Neo4j temporal types -> Python, recursively; nodes -> property dicts."""
    if hasattr(value, "to_native"):
        return value.to_native()
    if hasattr(value, "items") and not isinstance(value, dict):  # a Node
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [native(v) for v in value]
    return value


def read(driver: Driver) -> Catalog:
    # Relationship types such as ACTED_ON exist only once an operator has acted; Neo4j
    # warns about unknown types on every refresh until then.
    with driver.session(notifications_disabled_classifications=["UNRECOGNIZED"]) as s:

        def rows(query: str) -> list[dict[str, Any]]:
            return [native(dict(r)) for r in s.run(query)]

        plant = rows(PLANT)[0]
        levers = list(Levers.model_fields)
        batches = [
            {
                **r["batch"],
                "end_reason": r["batch"].get("end_reason"),
                "end": r["batch"].get("end"),
                "planned": {k: r["batch"][f"planned_{k}"] for k in levers},
                "actual": {k: r["batch"][k] for k in levers},
                "outcome": r["outcome"],
                "operations": r["operations"],
            }
            for r in rows(BATCHES)
        ]
        return Catalog(
            site=plant["site"],
            area=plant["area"],
            line=plant["line"],
            units=rows(UNITS),
            modules=rows(MODULES),
            tags=rows(TAGS),
            sensors=rows(SENSORS),
            bindings=rows(BINDINGS),
            recipes=[
                {
                    **r["recipe"],
                    "effective_from": r["recipe"]["effective_from"].isoformat(),
                    "nominal": {k: r["recipe"][k] for k in levers},
                    "limits": [
                        {
                            "parameter": lim["parameter"],
                            "type": lim["type"],
                            "low": lim["low"],
                            "high": lim["high"],
                            "spRelative": lim.get("sp_relative"),
                        }
                        for lim in r["limits"]
                    ],
                }
                for r in rows(RECIPES)
            ],
            batches=batches,
            alerts=[
                {
                    **r["event"],
                    "batch": r["batch"],
                    "tags": r["tags"],
                    "cleared_at": r["event"].get("cleared_at"),
                    "fault_class": r["event"].get("fault_class"),
                }
                for r in rows(ALERTS)
            ],
            actions=[
                {
                    **r["event"],
                    "batch": r["batch"],
                    "tags": r["tags"],
                    "reason": r["event"].get("reason"),
                    "old": r["event"]["old_value"],
                    "new": r["event"]["new_value"],
                    "recommendation": r["recommendation"],
                }
                for r in rows(ACTIONS)
            ],
            recommendations=[{**r["rec"], "batch": r["batch"]} for r in rows(RECOMMENDATIONS)],
        )


class SpaceCache:
    """The address space, rebuilt from the graph at most every `ttl_s` seconds.

    A failed refresh keeps serving the last good space: the graph is context, and a
    few seconds of staleness is better than an outage.
    """

    def __init__(
        self,
        load: Callable[[], Catalog],
        ttl_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._load = load
        self._ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._space: AddressSpace | None = None
        self._at = float("-inf")

    def __call__(self) -> AddressSpace:
        with self._lock:
            if self._space is None or self._clock() - self._at >= self._ttl_s:
                try:
                    self._space = build(self._load())
                except Exception:
                    if self._space is None:
                        raise
                    log.exception("address space refresh failed; serving the previous one")
                self._at = self._clock()
            return self._space

    def peek(self) -> AddressSpace | None:
        """The current space without refreshing (for the broker thread)."""
        return self._space
