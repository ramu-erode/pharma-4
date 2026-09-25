"""The assistant's tools, run against the i3X API in-process (ADR-0017)."""

import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from assistant import tools
from assistant.i3x_client import I3xClient
from tests.i3x import fixtures as fx


@pytest.fixture
def backend():
    b = fx.backend()
    b.cache.update(fx.PV_PH, fx.scalar(fx.PV_PH, 7.01, fx.T0))
    b.history.numeric_rows[fx.PV_PH] = [
        (fx.T0 + timedelta(minutes=i), 7.0 + (i % 10) / 100, "GOOD") for i in range(1000)
    ]
    return b


@pytest.fixture
def client(backend):
    return I3xClient("http://testserver/v1", fx.KEY, http=TestClient(fx.app(backend)))


def call(client, name, **args):
    return json.loads(tools.run(client, name, args))


def test_tool_schemas_are_well_formed():
    names = [t["name"] for t in tools.TOOLS]
    assert len(names) == len(set(names)) and set(names) == set(tools._HANDLERS)
    for t in tools.TOOLS:
        assert t["input_schema"]["type"] == "object" and t["description"]


def test_describe_model(client):
    out = call(client, "describe_model")
    assert any(t["elementId"] == "BatchType" for t in out["objectTypes"])
    assert {"elementId": "RanOn", "reverseOf": "HostedBatch"} in out["relationshipTypes"]


def test_find_objects_by_type_and_text(client):
    out = call(client, "find_objects", type_element_id="BatchType", text="0142")
    assert out["total"] == 1 and out["objects"][0]["elementId"] == "B2026-0142"
    assert call(client, "find_objects", root_only=True)["total"] == 4


def test_describe_related_and_values(client):
    [d] = call(client, "describe_objects", element_ids=["B2026-0142"])
    assert d["metadata"]["relationships"]["FollowsRecipe"] == ["recipe/v3"]
    [rel] = call(client, "get_related", element_ids=[fx.PV_PH], relationship_type="ConcernedBy")
    assert rel["related"][0]["elementId"] == "B2026-0142-A001"
    [v] = call(client, "read_values", element_ids=[fx.PV_PH])
    assert v["value"] == 7.01
    [missing] = call(client, "read_values", element_ids=["nope"])
    assert "not found" in missing["error"]


def test_read_history_buckets_long_numeric_series(client):
    out = call(
        client,
        "read_history",
        element_ids=[fx.PV_PH],
        start_time="2026-03-01T00:00:00Z",
        end_time="2026-03-02T00:00:00Z",
        max_points=20,
    )
    series = out["series"][0]
    assert series["points"] == 1000 and len(series["buckets"]) == 20
    assert sum(b["n"] for b in series["buckets"]) == 1000
    assert all(b["min"] <= b["mean"] <= b["max"] for b in series["buckets"])


def test_summarise_keeps_changes_of_structured_series():
    vals = [
        {"value": s, "quality": "Good", "timestamp": f"2026-03-01T0{i}:00:00Z"}
        for i, s in enumerate(["Setup", "Setup", "Growth", "Growth", "Production"])
    ]
    out = tools.summarise(vals, 10)
    assert [c["value"] for c in out["changes"]] == ["Setup", "Growth", "Production"]


def test_summarise_short_numeric_series_verbatim_and_counts_nulls():
    vals = [
        {"value": 1.0, "quality": "Good", "timestamp": "t1"},
        {"value": None, "quality": "Bad", "timestamp": "t2"},
    ]
    assert tools.summarise(vals, 10) == {"points": 2, "values": [["t1", 1.0]], "nullPoints": 1}


@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("nope", {}, "unknown tool"),
        ("read_values", {"element_ids": []}, "non-empty"),
        ("read_values", {"element_ids": ["a"] * 26}, "at most 25"),
        ("read_values", {"element_ids": ["a"], "max_depth": 9}, "max_depth"),
        ("read_history", {"element_ids": ["a"], "start_time": "x", "end_time": "y"}, "RFC 3339"),
        (
            "read_history",
            {
                "element_ids": ["a"],
                "start_time": "2026-03-02T00:00:00Z",
                "end_time": "2026-03-01T00:00:00Z",
            },
            "before",
        ),
    ],
)
def test_bad_input_is_a_tool_error_the_model_can_read(client, name, args, message):
    with pytest.raises(tools.ToolError, match=message):
        tools.run(client, name, args)


def test_oversized_results_ask_the_model_to_narrow(client, monkeypatch):
    monkeypatch.setattr(tools, "MAX_RESULT_CHARS", 100)
    with pytest.raises(tools.ToolError, match="Narrow the request"):
        tools.run(client, "describe_model", {})


def test_an_unreachable_server_is_a_tool_error():
    dead = I3xClient("http://127.0.0.1:9/v1", "k", timeout_s=0.5)
    with pytest.raises(tools.ToolError, match="unreachable"):
        tools.run(dead, "describe_model", {})
