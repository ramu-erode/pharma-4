"""graph-sync core: which messages become which statements (no Neo4j)."""

from common import models as m
from common import uns
from common.levers import actual_levers
from common.plant import get_plant
from edge.core import TagMap
from graph import core, load
from simulator.batches import BIO_PHASES
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


def test_config_covers_every_unit_and_every_phase_class():
    stmts = load.config_statements()
    bound = {(p["cell"], p["phase"]) for q, p in stmts if "BOUND_TO" in q}
    bio = {(c, ph.value) for c in ("BR-101", "BR-102") for ph in BIO_PHASES}
    assert {b for b in bound if b[0].startswith("BR")} == bio
    plant = get_plant()
    for cell, unit in plant.units.items():
        phases = set(plant.classes[unit.cls]["phase_classes"])
        assert {ph for c, ph in bound if c == cell} == phases, cell
    tags = [p["topic"] for q, p in stmts if q.startswith("MERGE (t:Tag")]
    assert len(tags) == len(TagMap.load().entries)
    units = {p["cell"]: p for q, p in stmts if "MERGE (u:Equipment" in q}
    assert set(units) == set(plant.units)
    assert units["FD-202"]["path"] == "pharmanextgen/tuas/api/train-1/FD-202"
    sites = {p["id"] for q, p in stmts if "MERGE (s:Site" in q}
    assert sites == {"grange-castle", "tuas", "freiburg"}


def test_recipe_limits_link_only_their_own_process_tags():
    stmts = load.config_statements()
    for q, p in stmts:
        if "FOR_TAG" not in q:
            continue
        for topic in p["topics"]:
            unit = uns.parse(topic).unit
            recipe_process = "bioreactor" if p["recipe"] in ("v1", "v2", "v3") else None
            if recipe_process:
                assert unit.site == "grange-castle", (p["id"], topic)
            else:
                assert unit.site != "grange-castle", (p["id"], topic)


# --- trains and genealogy (ADR-0018, ADR-0020) --------------------------------------------------

FD = uns.UnitPath("tuas", "api", "train-1", "FD-202")


def _stmts(topic: str, payload: m.Payload) -> list[tuple[str, dict]]:
    return core.handle(topic, payload)


def test_an_operation_links_the_batch_to_the_unit_it_ran_on():
    ev = m.OperationChanged(batch_id=BATCH, previous="Idle", current="Filtration")
    stmts = _stmts(
        uns.events(FD, "batch"),
        m.BatchEventPayload(v=ev, ts=TS, unit=None, batch=BATCH, src=m.Src.SIM),
    )
    q, p = stmts[0]
    assert "MERGE (b)-[:RAN_ON]->(u)" in q and p["cell"] == "FD-202"
    assert p["id"] == f"{BATCH}/Filtration"


def test_a_coa_rejection_sets_the_disposition():
    ev = m.BatchEnded(batch_id=BATCH, status="COMPLETE", disposition="REJECTED")
    ((_, p),) = _stmts(
        uns.events(FD, "batch"),
        m.BatchEventPayload(v=ev, ts=TS, unit=None, batch=BATCH, src=m.Src.SIM),
    )
    assert p["disposition"] == "REJECTED" and p["status"] == "COMPLETE"


def test_coa_results_become_outcome_properties():
    ((q, _),) = _stmts(
        uns.lab(FD, "yield"), m.ScalarPayload(v=87.2, ts=TS, unit="%", batch=BATCH, src="sim")
    )
    assert "o.yield_pct" in q  # YIELD is a Cypher keyword
    assert (
        _stmts(
            uns.lab(FD, "glucose"),
            m.ScalarPayload(v=1.0, ts=TS, unit="g/L", batch=BATCH, src="sim"),
        )
        == []
    )


def test_material_events_build_the_genealogy():
    made = m.MaterialProduced(batch_id=BATCH, lot=BATCH, material="API", quantity_kg=570.0)
    used = m.MaterialConsumed(batch_id="B2026-0150", lot=BATCH, material="API", quantity_kg=200.0)
    ((q1, p1),) = _stmts(
        uns.events(FD, "material"),
        m.MaterialEventPayload(v=made, ts=TS, unit="kg", batch=BATCH, src=m.Src.SIM),
    )
    ((q2, p2),) = _stmts(
        uns.events(UNIT, "material"),
        m.MaterialEventPayload(v=used, ts=TS, unit="kg", batch="B2026-0150", src=m.Src.SIM),
    )
    assert "[:PRODUCED]->(l)" in q1 and p1["lot"] == BATCH
    assert "[c:CONSUMED]->(l)" in q2 and p2["batch"] == "B2026-0150" and p2["kg"] == 200.0
