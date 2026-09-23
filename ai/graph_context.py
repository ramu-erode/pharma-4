"""Static batch context from the graph, read once per batch (ADR-0011): the recipe's action
limits and which PV signals each phase class *controls* on the unit. Falls back to the
same facts from configuration if the graph cannot answer yet."""

from __future__ import annotations

import logging

from neo4j import Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from ai.anomaly.detector import Limit
from ai.context import controlled_by_config
from ai.offline import limits_for

log = logging.getLogger(__name__)

LIMITS = """
MATCH (:Batch {id: $batch})-[:FOLLOWS]->(:Recipe)-[:HAS_LIMIT]->(l:SpecLimit {type: 'action'})
RETURN l.parameter AS name, l.low AS low, l.high AS high, l.sp_relative AS rel
"""

CONTROLLED = """
MATCH (pc:PhaseClass)-[:BOUND_TO {unit: $cell, role: 'control'}]->(m)
MATCH (m)-[:HAS_CM*0..1]->(:ControlModule)-[:HAS_TAG]->(t:Tag {class: 'pv'})
RETURN pc.name AS phase, collect(DISTINCT t.name) AS signals
"""


def batch_context(
    driver: Driver | None, batch_id: str, cell: str, recipe_id: str
) -> tuple[dict[str, Limit], dict[str, set[str]]]:
    if driver is not None:
        try:
            limits_rec, _, _ = driver.execute_query(LIMITS, batch=batch_id)
            ctrl_rec, _, _ = driver.execute_query(CONTROLLED, cell=cell)
            if limits_rec and ctrl_rec:
                limits = {
                    r["name"]: Limit(bool(r["rel"]), float(r["low"]), float(r["high"]))
                    for r in limits_rec
                }
                controlled = {r["phase"]: set(r["signals"]) for r in ctrl_rec}
                return limits, controlled
        except (Neo4jError, ServiceUnavailable):
            log.warning("graph unavailable; using configuration for %s", batch_id)
    return limits_for(recipe_id), controlled_by_config()
