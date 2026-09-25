"""Values: payload -> VQT, null and quality rules, composition and history (ADR-0016)."""

from datetime import timedelta

import jsonschema
import pytest

from common import models as m
from common import uns
from i3x import space as sp
from i3x import values
from tests.i3x import fixtures as fx
from tests.samples import samples


@pytest.fixture
def space():
    return fx.space()


@pytest.mark.parametrize(("topic", "payload"), samples(), ids=lambda x: str(x)[:40])
def test_every_structured_payload_has_an_i3x_value(topic, payload):
    p = uns.parse(topic)
    if p.kind is not uns.TopicKind.UNS or p.cls is uns.TopicClass.EVENTS:
        return  # events become graph objects; alerts are read from the graph too
    if p.name.startswith("anomaly/alert/"):
        return
    out = values.payload_vqt(payload)
    assert out["quality"] in {"Good", "GoodNoData"}


def test_values_conform_to_the_data_point_type_schemas(space):
    cache = values.LiveCache()
    for topic, payload in samples():
        obj = space.get(topic)  # tests.samples uses the same unit as the fixtures
        if obj is None or obj.topic is None:
            continue
        cache.update(topic, payload)
        vqt = values.own_vqt(obj, cache)
        if vqt["value"] is not None:
            jsonschema.validate(vqt["value"], sp.TYPES[obj.type_id].schema)


def test_quality_mapping_and_null_rules():
    ts = fx.T0
    good = values.payload_vqt(fx.scalar(fx.PV_PH, 7.0, ts))
    assert good == {"value": 7.0, "quality": "Good", "timestamp": "2026-03-01T00:00:00.000Z"}
    unc = values.payload_vqt(fx.scalar(fx.PV_PH, 7.0, ts, m.Quality.UNCERTAIN))
    assert unc["quality"] == "Uncertain" and unc["value"] == 7.0
    bad = values.payload_vqt(fx.scalar(fx.PV_PH, 7.0, ts, m.Quality.BAD))
    assert bad["quality"] == "Bad" and bad["value"] is None
    assert values.vqt(float("nan"), "Good", ts)["quality"] == "Bad"
    assert values.vqt(None, "Good", ts)["quality"] == "GoodNoData"


def test_unreported_point_is_good_no_data_at_plant_time(space):
    cache = values.LiveCache()
    cache.update(fx.PV_TEMP, fx.scalar(fx.PV_TEMP, 36.5, fx.T0 + timedelta(hours=5)))
    out = values.own_vqt(space.get(fx.PV_PH), cache)
    assert out == {"value": None, "quality": "GoodNoData", "timestamp": "2026-03-01T05:00:00.000Z"}


def test_retained_clear_removes_the_value(space):
    cache = values.LiveCache()
    cache.update(fx.PV_PH, fx.scalar(fx.PV_PH, 7.0, fx.T0))
    cache.update(fx.PV_PH, None)
    assert values.own_vqt(space.get(fx.PV_PH), cache)["quality"] == "GoodNoData"


def test_unit_value_merges_its_live_state(space):
    cache = values.LiveCache()
    batch = m.BatchStatePayload(
        v="B2026-0143", ts=fx.T0, unit=None, batch="B2026-0143", src=m.Src.SIM
    )
    cache.update(uns.state_batch(fx.UNIT), batch)
    v = values.own_vqt(space.get(fx.UNIT.prefix), cache)["value"]
    assert v["batch"] == "B2026-0143" and v["operation"] is None
    assert space.get(fx.UNIT.prefix).value.get("batch") is None  # static part untouched


def test_max_depth_one_returns_no_components_zero_returns_the_tree(space):
    cache = values.LiveCache()
    cache.update(fx.PV_PH, fx.scalar(fx.PV_PH, 7.02, fx.T0))
    unit = fx.UNIT.prefix
    assert "components" not in values.current(space, cache, unit, 1)
    tree = values.current(space, cache, unit, 0)
    ph = tree["components"][f"{unit}/EM-PH"]["components"][f"{unit}/EM-PH/AIC-102"]
    assert ph["components"][fx.PV_PH]["value"] == 7.02
    two = values.current(space, cache, unit, 2)
    assert "components" not in two["components"][f"{unit}/EM-PH"]


def test_history_numeric_structured_and_graph_objects(space):
    store = fx.FakeHistory()
    store.numeric_rows[fx.PV_PH] = [
        (fx.T0 + timedelta(minutes=i), 7.0 + i / 100, "GOOD") for i in range(5)
    ]
    op = uns.state_operation(fx.UNIT)
    payload = m.OperationPayload(
        v=m.Operation.GROWTH, ts=fx.T0, unit=None, batch="B2026-0142", src=m.Src.SIM
    )
    store.structured_rows[op] = [(fx.T0, payload.model_dump(mode="json"))]
    end = fx.T0 + timedelta(days=1)
    h, cut = values.history(space, store, fx.PV_PH, fx.T0, end, 1, 100)
    assert [v["value"] for v in h["values"]] == [7.0, 7.01, 7.02, 7.03, 7.04] and not cut
    h, _ = values.history(space, store, op, fx.T0, end, 1, 100)
    assert h["values"][0]["value"] == "Growth"
    h, _ = values.history(space, store, "B2026-0142", fx.T0, end, 1, 100)
    assert h == {"isComposition": False, "values": []}


def test_history_truncates_at_the_limit_and_says_so(space):
    store = fx.FakeHistory()
    store.numeric_rows[fx.PV_PH] = [(fx.T0 + timedelta(minutes=i), 7.0, "GOOD") for i in range(10)]
    h, cut = values.history(space, store, fx.PV_PH, fx.T0, fx.T0 + timedelta(days=1), 1, 4)
    assert cut and len(h["values"]) == 4


def test_a_request_wide_budget_caps_composition_history(space):
    store = fx.FakeHistory()
    for topic in (fx.PV_PH, fx.SP_PH):
        store.numeric_rows[topic] = [(fx.T0 + timedelta(minutes=i), 7.0, "GOOD") for i in range(5)]
    cm = f"{fx.UNIT.prefix}/EM-PH/AIC-102"
    budget = values.Budget(7)
    h, cut = values.history(space, store, cm, fx.T0, fx.T0 + timedelta(days=1), 0, 100, budget)
    counts = [len(c["values"]) for c in h["components"].values()]
    assert cut and sorted(counts) == [2, 5] and budget.points == 0


def test_composition_history_mirrors_components(space):
    store = fx.FakeHistory()
    store.numeric_rows[fx.PV_PH] = [(fx.T0, 7.0, "GOOD")]
    cm = f"{fx.UNIT.prefix}/EM-PH/AIC-102"
    h, _ = values.history(space, store, cm, fx.T0, fx.T0 + timedelta(hours=1), 2, 10)
    assert h["isComposition"] and h["components"][fx.PV_PH]["values"][0]["value"] == 7.0
