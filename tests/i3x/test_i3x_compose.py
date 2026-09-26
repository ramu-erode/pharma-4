"""The i3x service against the running stack (`pytest -m compose`, ADR-0016).

For the full check, run CESMII's conformance suite as well (README, "i3X").
"""

from datetime import datetime

import httpx
import pytest

from assistant.i3x_client import I3xClient
from common import uns
from common.settings import get_settings

pytestmark = pytest.mark.compose

UNIT = uns.UnitPath("grange-castle", "upstream", "suite-1", "BR-101")


@pytest.fixture(scope="module")
def i3x() -> I3xClient:
    s = get_settings()
    return I3xClient(s.i3x_url, s.i3x_api_key)


def test_info_is_open_and_data_is_not():
    base = get_settings().i3x_url
    assert httpx.get(f"{base}/info").json()["result"]["specVersion"] == "1.0"
    assert httpx.get(f"{base}/namespaces").status_code == 401


def test_backfilled_batches_are_objects(i3x):
    batches = i3x.objects("BatchType")
    assert len(batches) >= 200
    assert {o["elementId"] for o in i3x.objects(root=True)} >= {"pharmanextgen", "batches"}


def test_a_unit_reads_as_one_live_tree(i3x):
    [r] = i3x.value([UNIT.prefix], max_depth=0)
    components = r["result"]["components"]
    assert uns.state_batch(UNIT) in components
    assert any(c.endswith("/EM-PH") for c in components)


def test_history_of_a_completed_batch(i3x):
    ids = [o["elementId"] for o in i3x.objects("BatchType")]
    done = next(
        v
        for r in i3x.value(ids)
        if (v := r["result"]["value"])["status"] == "COMPLETE" and v["unit"] == UNIT.cell
    )
    start = datetime.fromisoformat(done["start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(done["end"].replace("Z", "+00:00"))
    results, note = i3x.history([uns.pv(UNIT, "ph"), uns.lab(UNIT, "titer")], start, end)
    counts = [len(r["result"]["values"]) for r in results]
    assert note is None and all(c > 0 for c in counts)


def test_every_site_and_the_genealogy_are_served(i3x):
    roots = {o["elementId"] for o in i3x.objects(root=True)}
    assert "materials" in roots
    sites = {o["elementId"] for o in i3x.objects("SiteType")}
    assert sites == {f"pharmanextgen/{s}" for s in ("grange-castle", "tuas", "freiburg")}
    assert i3x.objects("RotaryTabletPressType") and i3x.objects("FilterDryerType")
    ids = [o["elementId"] for o in i3x.objects("BatchType")]
    tablets = [v for r in i3x.value(ids) if (v := r["result"]["value"])["process"] == "osd"]
    assert len(tablets) >= 140
    used = next(t for t in tablets if t["lotsConsumed"])
    lot = used["lotsConsumed"][0]["lot"]
    [r] = i3x.value([f"lot/{lot}"])
    made = r["result"]["value"]
    assert made["producedBy"] == lot
    assert used["batchId"] in {c["batch"] for c in made["consumedBy"]}
