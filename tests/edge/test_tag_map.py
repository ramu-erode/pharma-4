"""Every tag-map change needs a mapping test (CLAUDE.md). The simulator's DCS config and the
edge tag map are separate artifacts; this is what keeps them in step (Q22)."""

from pathlib import Path

import pytest
import yaml

from common import uns
from common.plant import get_plant
from common.uns import TopicClass
from edge.core import TagMap

ROOT = Path(__file__).resolve().parents[2]
DCS = yaml.safe_load((ROOT / "simulator/dcs_tags.yaml").read_text())["classes"]
TAG_MAP = TagMap.load()
PLANT = get_plant()


def dcs_raw_tags(key: str = "tags") -> set[str]:
    """Every raw tag the simulated DCS exposes, for every unit of every site."""
    return {
        f"{u.path.device}.{suffix}"
        for u in PLANT.units.values()
        for suffix in DCS[u.cls].get(key, [])
    }


def test_every_dcs_tag_is_mapped_except_expected_unmapped():
    assert dcs_raw_tags() - set(TAG_MAP.entries) == dcs_raw_tags("expected_unmapped")


def test_map_has_no_tags_the_dcs_lacks():
    assert set(TAG_MAP.entries) <= dcs_raw_tags()


def test_at_least_one_tag_is_deliberately_unmapped():
    assert dcs_raw_tags("expected_unmapped"), "edge/unmapped needs something to show"


def test_every_unit_is_commissioned_with_its_equipment_class():
    assert TAG_MAP.cells == set(PLANT.units)
    for e in TAG_MAP.entries.values():
        assert e.unit_path == PLANT.path(e.unit_path.cell)


@pytest.mark.parametrize(
    ("raw_tag", "topic", "unit"),
    [
        ("BR101.AIC-102.PV", "pharmanextgen/grange-castle/upstream/suite-1/BR-101/pv/ph", "pH"),
        (
            "BR102.TIC-101.SP",
            "pharmanextgen/grange-castle/upstream/suite-1/BR-102/sp/temperature",
            "°C",
        ),
        (
            "BR101.FIC-108.PV",
            "pharmanextgen/grange-castle/upstream/suite-1/BR-101/pv/co2_flow",
            "L/min",
        ),
        (
            "BR101.FQI-109.PV",
            "pharmanextgen/grange-castle/upstream/suite-1/BR-101/pv/base_total",
            "mL",
        ),
        ("RX201.AI-209.PV", "pharmanextgen/tuas/api/train-1/RX-201/pv/conversion", "%"),
        ("FD202.PIC-216.SP", "pharmanextgen/tuas/api/train-1/FD-202/sp/vacuum", "mbar"),
        ("RC302.AI-316.PV", "pharmanextgen/freiburg/osd/line-1/RC-302/pv/ribbon_density", "g/cm³"),
        ("TP303.PI-325.PV", "pharmanextgen/freiburg/osd/line-1/TP-303/pv/ejection_force", "N"),
        ("BL301.MI-304.PV", "pharmanextgen/freiburg/osd/line-1/BL-301/pv/room_rh", "% RH"),
    ],
)
def test_mappings(raw_tag, topic, unit):
    e = TAG_MAP.entries[raw_tag]
    assert (e.topic, e.unit) == (topic, unit)


def test_topics_follow_architecture_names():
    names = {(e.cls, e.name) for e in TAG_MAP.entries.values()}
    assert {
        (TopicClass.PV, n)
        for n in (
            "temperature",
            "ph",
            "do",
            "agitation",
            "air_flow",
            "o2_flow",
            "co2_flow",
            "feed_total",
            "base_total",
            "pressure",
            "weight",
        )
    } <= names
    assert {(TopicClass.SP, n) for n in ("temperature", "ph", "do")} <= names


def test_every_topic_parses_and_is_unique():
    topics = [e.topic for e in TAG_MAP.entries.values()]
    assert len(topics) == len(set(topics))
    for t in topics:
        assert uns.parse(t).cls in (TopicClass.PV, TopicClass.SP)
