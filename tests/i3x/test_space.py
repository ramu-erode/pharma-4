"""The i3X address space: shape, identity and relationship rules (ADR-0016)."""

import jsonschema
import pytest

from common import uns
from i3x import space as sp
from i3x.values import LiveCache, current
from simulator.api.engine import LAB_UNITS as API_LAB
from simulator.engine import LAB_UNITS
from simulator.osd.engine import LAB_UNITS as OSD_LAB
from tests.i3x import fixtures as fx


@pytest.fixture(scope="module")
def space() -> sp.AddressSpace:
    return fx.space()


def test_lab_points_match_what_the_simulator_publishes():
    labs = sp.LAB_UNITS
    assert labs["bioreactor"] == LAB_UNITS
    assert labs["reactor"] | labs["filter_dryer"] == API_LAB
    assert labs["blender"] | labs["roller_compactor"] | labs["tablet_press"] == OSD_LAB


def test_every_object_type_has_a_valid_json_schema_in_a_declared_namespace():
    uris = {n["uri"] for n in sp.NAMESPACES}
    for t in sp.TYPES.values():
        jsonschema.Draft202012Validator.check_schema(t.schema)
        assert t.namespace in uris


def test_every_relationship_type_has_a_symmetric_registered_reverse():
    by_id = {r["elementId"]: r for r in sp.RELATIONSHIP_TYPES}
    for r in sp.RELATIONSHIP_TYPES:
        assert by_id[r["reverseOf"]]["reverseOf"] == r["elementId"]


def test_element_ids_are_unique_across_objects_types_and_relationships(space):
    ids = [*space.objects, *sp.TYPES, *(r["elementId"] for r in sp.RELATIONSHIP_TYPES)]
    assert len(ids) == len(set(ids))


def test_roots_are_the_plant_and_the_four_folders(space):
    assert {o.element_id for o in space.roots()} == {
        "pharmanextgen",
        sp.ROOT_BATCHES,
        sp.ROOT_RECIPES,
        sp.ROOT_PHASES,
        sp.ROOT_MATERIALS,
    }


def test_plant_hierarchy_uses_uns_path_prefixes(space):
    assert space.get(fx.UNIT.prefix).parent_id == "pharmanextgen/grange-castle/upstream/suite-1"
    assert space.get("pharmanextgen/grange-castle").parent_id == "pharmanextgen"


def test_a_data_points_element_id_is_its_uns_topic(space):
    for o in space.objects.values():
        if o.topic is not None:
            assert o.element_id == o.topic
            uns.parse(o.topic)  # a real UNS topic


def test_tags_are_components_of_their_control_module(space):
    cm = f"{fx.UNIT.prefix}/EM-PH/AIC-102"
    assert space.get(fx.PV_PH).parent_id == cm
    assert set(space.get(cm).components) == {fx.PV_PH, fx.SP_PH}
    assert space.get(f"{fx.UNIT.prefix}/EM-PH").parent_id == fx.UNIT.prefix


def test_every_edge_is_stored_both_ways(space):
    for o in space.objects.values():
        for rel, targets in o.edges.items():
            for t in targets:
                assert o.element_id in space.get(t).edges[sp.REVERSE[rel]], (o.element_id, rel, t)


def test_parent_id_matches_a_hierarchy_or_composition_edge(space):
    for o in space.objects.values():
        if o.parent_id is not None:
            ups = o.edges.get(sp.HAS_PARENT, []) + o.edges.get(sp.COMPONENT_OF, [])
            assert ups == [o.parent_id]


def test_phase_bindings_become_controls_and_monitors(space):
    related = space.related("phase-class/PH_CTRL")
    assert (sp.CONTROLS, space.get(f"{fx.UNIT.prefix}/EM-PH")) in related
    assert (sp.MONITORS, space.get(f"{fx.UNIT.prefix}/EM-THERMAL/TIC-101")) in related


def test_batch_context_links(space):
    batch = space.get("B2026-0142")
    assert batch.edges[sp.RAN_ON] == [fx.UNIT.prefix]
    assert batch.edges[sp.FOLLOWS_RECIPE] == ["recipe/v3"]
    assert set(batch.edges[sp.HAS_CHILDREN]) == {
        "B2026-0142-A001",
        "B2026-0142/operator/x/do_sp",
        "R-B2026-0142-d4.00",
    }
    assert space.get("B2026-0142-A001").edges[sp.CONCERNS_TAG] == [fx.PV_PH]
    assert space.get("B2026-0142/operator/x/do_sp").edges[sp.ACTED_ON] == ["R-B2026-0142-d4.00"]


def test_edges_to_objects_outside_the_space_are_dropped(space):
    # The operator action names sp/do, which this small plant does not have.
    assert sp.CONCERNS_TAG not in space.get("B2026-0142/operator/x/do_sp").edges


def test_batch_operations_are_in_time_order(space):
    ops = space.get("B2026-0142").value["operations"]
    assert [o["name"] for o in ops] == ["Setup", "Growth"]
    assert ops[1]["phases"][0]["holds"] == 1


def test_static_values_conform_to_their_type_schema(space):
    cache = LiveCache()
    for eid, o in space.objects.items():
        if o.topic is None:
            v = current(space, cache, eid, 1)["value"]
            jsonschema.validate(v, sp.TYPES[o.type_id].schema)


def test_descendants_follow_max_depth(space):
    unit = fx.UNIT.prefix
    assert space.descendants(unit, 1) == [unit]
    two = space.descendants(unit, 2)
    assert f"{unit}/EM-PH" in two and f"{unit}/EM-PH/AIC-102" not in two
    assert fx.PV_PH in space.descendants(unit, 0)


def test_only_known_equipment_types_are_modelled():
    cat = fx.catalog()
    cat.units[0]["type"] = "Centrifuge"
    with pytest.raises(ValueError, match="unknown equipment type"):
        sp.build(cat)


def test_sites_and_trains_hang_off_the_enterprise(space):
    tuas = space.get("pharmanextgen/tuas")
    assert tuas.parent_id == "pharmanextgen" and tuas.value["location"] == "Singapore"
    rx = space.get(fx.RX.prefix)
    assert rx.type_id == "ReactorCrystallizerType"
    assert rx.parent_id == "pharmanextgen/tuas/api/train-1"
    assert rx.value["equipmentClass"] == "reactor" and rx.value["process"] == "api"
    # a reactor's own phases and lab results, and it is the API train's home unit
    assert space.get(uns.state_phase(fx.RX, "DOSE_ADD"))
    assert not space.get(uns.state_phase(fx.RX, "PH_CTRL"))
    assert space.get(uns.lab(fx.RX, "conversion_ipc"))
    assert space.get(uns.ai_prediction(fx.RX))


def test_lots_carry_the_genealogy(space):
    lot = space.get("lot/B2026-0140")
    assert lot.value["producedBy"] == "B2026-0140"
    assert lot.value["consumedBy"] == [{"batch": "B2026-0143", "quantityKg": 200.0}]
    made = {t.element_id for _, t in space.related("B2026-0140", sp.PRODUCED)}
    used = {t.element_id for _, t in space.related("B2026-0143", sp.CONSUMED)}
    assert made == used == {"lot/B2026-0140"}
    assert space.get("B2026-0143").value["lotsConsumed"] == [
        {"lot": "B2026-0140", "quantityKg": 200.0}
    ]
    assert space.get("B2026-0140").value["process"] == "api"
