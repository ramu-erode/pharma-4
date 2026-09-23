import pytest
from hypothesis import given
from hypothesis import strategies as st
from paho.mqtt.client import topic_matches_sub

from common import uns
from common.uns import Delivery, SimCommand, TopicClass, TopicError, TopicKind
from tests.samples import UNIT


def test_unit_prefix_and_device():
    assert UNIT.prefix == "pharmaco/chennai/upstream/suite-1/BR-101"
    assert UNIT.device == "BR101"


@pytest.mark.parametrize(
    ("topic", "expected"),
    [
        (uns.pv(UNIT, "ph"), "pharmaco/chennai/upstream/suite-1/BR-101/pv/ph"),
        (uns.state_phase(UNIT, "TEMP_CTRL"), f"{UNIT.prefix}/state/phase/temp_ctrl"),
        (uns.ai_alert(UNIT, "stats-co2_flow"), f"{UNIT.prefix}/ai/anomaly/alert/stats-co2_flow"),
        (uns.ai_recommendation(UNIT), f"{UNIT.prefix}/ai/yield/recommendation"),
        (uns.meta_tag("BR-101", TopicClass.SP, "do"), "pharmaco/_meta/tags/BR-101/sp/do"),
        (uns.meta_status("edge-adapter"), "pharmaco/_meta/edge-adapter/status"),
        (uns.edge_raw("BR101", "BR101.AIC-102.PV"), "edge/raw/BR101/BR101.AIC-102.PV"),
        (uns.sim_cmd(SimCommand.CLOCK), "_sim/cmd/clock"),
        (uns.sim_faults("BR-102"), "_sim/faults/BR-102"),
    ],
)
def test_builders(topic, expected):
    assert topic == expected


ALL_BUILT = [
    uns.pv(UNIT, "ph"),
    uns.sp(UNIT, "temperature"),
    uns.lab(UNIT, "ph_offline"),
    uns.state_batch(UNIT),
    uns.state_operation(UNIT),
    uns.state_phase(UNIT, "feed_add"),
    uns.events(UNIT, "batch"),
    uns.events(UNIT, "operator"),
    uns.ai_score(UNIT),
    uns.ai_alert(UNIT, "rules-stuck_do"),
    uns.ai_prediction(UNIT),
    uns.ai_recommendation(UNIT),
    uns.meta_tag("BR-101", TopicClass.PV, "co2_flow"),
    uns.meta_status("simulator"),
    uns.edge_raw("BR101", "BR101.TI-199.PV"),
    uns.edge_unmapped(),
    *[uns.sim_cmd(c) for c in SimCommand],
    uns.sim_clock(),
    uns.sim_faults("BR-101"),
]


@pytest.mark.parametrize("topic", ALL_BUILT)
def test_every_built_topic_parses(topic):
    p = uns.parse(topic)
    if p.kind is TopicKind.UNS:
        assert p.unit == UNIT
        assert f"{UNIT.prefix}/{p.cls}/{p.name}" == topic


def test_parse_details():
    p = uns.parse(uns.state_phase(UNIT, "PH_CTRL"))
    assert (p.cls, p.name) == (TopicClass.STATE, "phase/ph_ctrl")
    p = uns.parse(uns.edge_raw("BR101", "BR101.AIC-102.PV"))
    assert (p.device, p.raw_tag) == ("BR101", "BR101.AIC-102.PV")
    assert uns.parse("_sim/cmd/fault").command is SimCommand.FAULT
    assert uns.parse("pharmaco/_meta/tags/BR-101/pv/ph").cell == "BR-101"


@pytest.mark.parametrize(
    "bad",
    [
        "pharmaco/chennai/upstream/suite-1/BR-101/pv/PH",  # uppercase name
        "pharmaco/chennai/upstream/suite-1/br-101/pv/ph",  # cell not BR-101 style
        "pharmaco/chennai/upstream/suite-1/BR-101/xx/ph",  # unknown class
        "pharmaco/chennai/upstream/BR-101/pv/ph",  # missing a level
        "edge/raw/BR101/BR102.AIC-102.PV",  # tag of another device
        "_sim/cmd/reboot",
        "pharmaco/_meta/tags/BR-101/pv",
        "somewhere/else",
        "",
    ],
)
def test_parse_rejects(bad):
    with pytest.raises(TopicError):
        uns.parse(bad)


@pytest.mark.parametrize(
    "call",
    [
        lambda: uns.pv(UNIT, "pH"),
        lambda: uns.pv(UNIT, "ph/raw"),
        lambda: uns.ai_alert(UNIT, "nolayer"),
        lambda: uns.meta_status("tags"),
        lambda: uns.UnitPath("chennai", "upstream", "suite-1", "BR101"),
    ],
)
def test_builders_reject(call):
    with pytest.raises(TopicError):
        call()


@pytest.mark.parametrize(
    ("topic", "expected"),
    [
        (uns.pv(UNIT, "ph"), Delivery(0, True)),
        (uns.sp(UNIT, "ph"), Delivery(0, True)),
        (uns.lab(UNIT, "vcd"), Delivery(1, True)),
        (uns.state_operation(UNIT), Delivery(1, True)),
        (uns.events(UNIT, "batch"), Delivery(1, False)),
        (uns.ai_alert(UNIT, "mspc-do_fouling"), Delivery(1, True)),
        (uns.ai_recommendation(UNIT), Delivery(1, True)),
        (uns.ai_score(UNIT), Delivery(0, True)),
        (uns.ai_prediction(UNIT), Delivery(0, True)),
        (uns.meta_status("anomaly"), Delivery(1, True)),
        (uns.meta_tag("BR-101", TopicClass.PV, "ph"), Delivery(1, True)),
        (uns.edge_raw("BR101", "BR101.AIC-102.PV"), Delivery(0, False)),
        (uns.edge_unmapped(), Delivery(1, True)),
        (uns.sim_cmd(SimCommand.BATCH), Delivery(1, False)),
        (uns.sim_faults("BR-101"), Delivery(1, False)),
        (uns.sim_clock(), Delivery(1, True)),
    ],
)
def test_delivery_policy(topic, expected):
    assert uns.delivery(topic) == expected


def test_subscription_patterns():
    assert topic_matches_sub(uns.sub_class(TopicClass.PV), uns.pv(UNIT, "do"))
    assert not topic_matches_sub(uns.sub_class(TopicClass.PV), uns.sp(UNIT, "do"))
    # _meta/tags/<cell>/pv/... must not look like a live PV to a pv subscriber
    assert not topic_matches_sub(
        uns.sub_class(TopicClass.PV), uns.meta_tag("BR-101", TopicClass.PV, "do")
    )
    assert topic_matches_sub(uns.sub_state_batch(), uns.state_batch(UNIT))
    assert topic_matches_sub(uns.SUB_EDGE_RAW, uns.edge_raw("BR101", "BR101.AIC-102.PV"))


@given(st.from_regex(r"[a-z0-9_]{1,20}", fullmatch=True), st.sampled_from(list(TopicClass)))
def test_roundtrip_property(name, cls):
    topic = f"{UNIT.prefix}/{cls}/{name}"
    p = uns.parse(topic)
    assert (p.cls, p.name) == (cls, name)
