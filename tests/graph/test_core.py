"""graph-sync core: which messages become which statements (no Neo4j)."""

from common import models as m
from common import uns
from common.levers import actual_levers
from edge.core import TagMap
from graph import core, load
from tests.samples import BATCH, LEVERS, TS, UNIT, samples


def test_only_context_topics_reach_the_graph():
    for topic, payload in samples():
        stmts = core.handle(topic, payload)
        p = uns.parse(topic)
        if p.kind is uns.TopicKind.UNS and p.cls in (uns.TopicClass.PV, uns.TopicClass.SP):
            assert stmts == [], "raw values never go into the graph (ADR-0004)"
        if p.kind in (uns.TopicKind.META_STATUS, uns.TopicKind.EDGE_RAW, uns.TopicKind.SIM_CMD):
            assert stmts == []


def test_every_statement_merges_or_matches():
    for topic, payload in samples():
        for query, params in core.handle(topic, payload):
            assert query.lstrip().startswith(("MERGE", "MATCH"))
            assert isinstance(params, dict)


def test_fault_labels_never_become_events():
    topic, payload = next((t, p) for t, p in samples() if t.startswith("_sim/faults"))
    [(query, _)] = core.handle(topic, payload)
    assert ":FaultInjection" in query and ":Event" not in query


def test_operator_event_updates_the_actual_lever_and_links_the_recommendation():
    ev = m.OperatorEvent(
        operator="demo", parameter="prod_temp", old=33.0, new=33.5, recommendation_id="R-7"
    )
    payload = m.OperatorEventPayload(v=ev, ts=TS, unit=None, batch=BATCH, src="operator")
    queries = " ".join(q for q, _ in core.handle(uns.events(UNIT, "operator"), payload))
    assert "SET b.prod_temp = $new" in queries
    assert "ACTED_ON" in queries and "ON_TAG" in queries


def test_only_batch_structure_triggers_projection():
    for topic, payload in samples():
        batch = core.projection_trigger(topic, payload)
        if batch:
            assert isinstance(payload, m.BatchEventPayload)
            assert not isinstance(payload.v, m.BatchStarted)


def test_actual_levers_apply_operator_changes_in_order():
    events = [
        m.OperatorEvent(operator="a", parameter="prod_temp", old=33.0, new=33.5),
        m.OperatorEvent(operator="a", parameter="prod_temp", old=33.5, new=34.0),
        m.OperatorEvent(operator="a", parameter="not_a_lever", old=0, new=1),
    ]
    assert actual_levers(LEVERS, events) == LEVERS.model_copy(update={"prod_temp": 34.0})


def test_config_covers_both_units_and_every_phase_class():
    stmts = load.config_statements("chennai", "upstream", "suite-1")
    bound = {(p["cell"], p["phase"]) for q, p in stmts if "BOUND_TO" in q}
    assert bound == {(c, ph.value) for c in ("BR-101", "BR-102") for ph in m.PhaseClass}
    tags = [p["topic"] for q, p in stmts if q.startswith("MERGE (t:Tag")]
    assert len(tags) == len(TagMap.load("chennai", "upstream", "suite-1").entries)
