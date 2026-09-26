"""The i3X HTTP API: envelopes, auth, bulk rules, errors and the sync flow (ADR-0016).

CESMII's conformance suite is the full check against a running server (README); these
tests pin the behaviours it exercises so a regression shows up in `pytest`.
"""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from i3x import space as sp
from tests.i3x import fixtures as fx

AUTH = {"X-API-Key": fx.KEY}


@pytest.fixture
def backend():
    return fx.backend()


@pytest.fixture
def client(backend):
    return TestClient(fx.app(backend), headers=AUTH)


def test_info_needs_no_auth_and_declares_read_only():
    info = TestClient(fx.app()).get("/v1/info").json()
    assert info["success"] and info["result"]["specVersion"] == "1.0"
    caps = info["result"]["capabilities"]
    assert caps == {
        "query": {"history": True},
        "update": {"current": False, "history": False},
        "subscribe": {"stream": False},
    }


def test_everything_else_needs_the_key():
    anon = TestClient(fx.app())
    r = anon.get("/v1/namespaces")
    assert r.status_code == 401 and r.json()["success"] is False
    assert r.json()["responseDetail"]["status"] == 401
    assert anon.get("/v1/namespaces", headers={"Authorization": f"Bearer {fx.KEY}"}).is_success
    assert anon.get("/v1/namespaces", headers={"X-API-Key": "wrong"}).status_code == 401


def test_gzip_whenever_asked_even_for_small_bodies(client):
    r = client.get("/v1/namespaces", headers={"Accept-Encoding": "gzip"})
    assert r.headers["content-encoding"] == "gzip"
    assert r.json()["result"] == sp.NAMESPACES  # httpx decoded it


def test_type_filters_by_namespace(client):
    types = client.get("/v1/objecttypes", params={"namespaceUri": sp.NS_PHARMA}).json()["result"]
    assert len(types) == len(sp.TYPES)
    assert client.get("/v1/objecttypes", params={"namespaceUri": sp.NS_I3X}).json()["result"] == []
    rels = client.get("/v1/relationshiptypes", params={"namespaceUri": sp.NS_I3X}).json()
    assert {r["elementId"] for r in rels["result"]} == {
        "HasParent",
        "HasChildren",
        "HasComponent",
        "ComponentOf",
    }


def test_bulk_keeps_request_order_and_fails_unknown_ids_per_item(client):
    ids = ["B2026-0142", "nope", fx.PV_PH]
    body = client.post("/v1/objects/list", json={"elementIds": ids}).json()
    assert [r["elementId"] for r in body["results"]] == ids
    assert body["success"] is False
    assert body["results"][1]["responseDetail"]["status"] == 404
    assert body["results"][2]["result"]["typeElementId"] == "ProcessValueType"


def test_objects_filters_and_metadata(client):
    roots = client.get("/v1/objects", params={"root": "true"}).json()["result"]
    assert {o["elementId"] for o in roots} == {
        "pharmanextgen",
        "batches",
        "recipes",
        "phase-classes",
        "materials",
    }
    assert all(o["parentId"] is None for o in roots)
    batches = client.get("/v1/objects", params={"typeElementId": "BatchType"}).json()["result"]
    assert [b["elementId"] for b in batches] == ["B2026-0140", "B2026-0142", "B2026-0143"]
    meta = client.get("/v1/objects", params={"includeMetadata": "true"}).json()["result"]
    assert all(o["metadata"]["typeNamespaceUri"] == sp.NS_PHARMA for o in meta)


def test_related_includes_hierarchy_and_honours_the_filter(client):
    body = client.post("/v1/objects/related", json={"elementIds": [fx.PV_PH]}).json()
    rels = {
        (e["sourceRelationship"], e["object"]["elementId"]) for e in body["results"][0]["result"]
    }
    assert ("ComponentOf", f"{fx.UNIT.prefix}/EM-PH/AIC-102") in rels
    assert ("ConcernedBy", "B2026-0142-A001") in rels
    assert ("MeasuredBy", f"{fx.UNIT.prefix}/PH-PROBE") in rels
    only = client.post(
        "/v1/objects/related", json={"elementIds": [fx.PV_PH], "relationshipType": "MeasuredBy"}
    ).json()["results"][0]["result"]
    assert [e["sourceRelationship"] for e in only] == ["MeasuredBy"]


def test_value_reads_the_cache(client, backend):
    backend.cache.update(fx.PV_PH, fx.scalar(fx.PV_PH, 7.01, fx.T0))
    [r] = client.post("/v1/objects/value", json={"elementIds": [fx.PV_PH]}).json()["results"]
    assert r["result"] == {
        "isComposition": False,
        "value": 7.01,
        "quality": "Good",
        "timestamp": "2026-03-01T00:00:00.000Z",
    }


@pytest.mark.parametrize(
    "body",
    [
        {"elementIds": ["x"], "endTime": "2026-03-01T00:00:00Z"},
        {"elementIds": ["x"], "startTime": "2026-03-01T00:00:00Z"},
        {"elementIds": ["x"], "startTime": "not-a-date", "endTime": "2026-03-01T00:00:00Z"},
        {
            "elementIds": ["x"],
            "startTime": "2026-03-02T00:00:00Z",
            "endTime": "2026-03-01T00:00:00Z",
        },
    ],
)
def test_bad_history_requests_are_400(client, body):
    r = client.post("/v1/objects/history", json=body)
    assert r.status_code == 400 and r.json()["responseDetail"]["status"] == 400


def test_truncated_history_is_206_with_a_reason():
    b = fx.backend(history_limit=3)
    b.history.numeric_rows[fx.PV_PH] = [
        (fx.T0 + timedelta(minutes=i), 7.0, "GOOD") for i in range(5)
    ]
    r = TestClient(fx.app(b), headers=AUTH).post(
        "/v1/objects/history",
        json={
            "elementIds": [fx.PV_PH],
            "startTime": "2026-03-01T00:00:00Z",
            "endTime": "2026-03-02T00:00:00Z",
        },
    )
    assert r.status_code == 206 and "truncated" in r.json()["responseDetail"]["detail"]
    assert len(r.json()["results"][0]["result"]["values"]) == 3


@pytest.mark.parametrize(
    ("method", "path"),
    [("PUT", "/v1/objects/value"), ("PUT", "/v1/objects/history")],
)
def test_writes_are_not_implemented(client, method, path):
    r = client.request(method, path, json={"updates": []})
    assert r.status_code == 501 and r.json()["success"] is False


def test_subscription_flow(client, backend):
    backend.cache.update(fx.PV_PH, fx.scalar(fx.PV_PH, 7.0, fx.T0))
    sid = client.post("/v1/subscriptions", json={"clientId": "c1"}).json()["result"][
        "subscriptionId"
    ]
    reg = client.post(
        "/v1/subscriptions/register",
        json={"clientId": "c1", "subscriptionId": sid, "elementIds": [fx.PV_PH, "nope"]},
    ).json()
    assert [r["success"] for r in reg["results"]] == [True, False]
    first = client.post("/v1/subscriptions/sync", json={"clientId": "c1", "subscriptionId": sid})
    assert first.json()["result"][0]["updates"][0]["value"] == 7.0
    ack = client.post(
        "/v1/subscriptions/sync",
        json={"clientId": "c1", "subscriptionId": sid, "lastSequenceNumber": 1},
    )
    assert ack.json()["result"] == []
    listed = client.post(
        "/v1/subscriptions/list", json={"clientId": "c1", "subscriptionIds": [sid]}
    ).json()["results"][0]["result"]
    assert listed["monitoredObjects"] == [{"elementId": fx.PV_PH, "maxDepth": 1}]


def test_subscriptions_need_a_client_id_and_are_private(client):
    assert client.post("/v1/subscriptions", json={}).status_code == 400
    sid = client.post("/v1/subscriptions", json={"clientId": "c1"}).json()["result"][
        "subscriptionId"
    ]
    assert client.post("/v1/subscriptions/sync", json={"subscriptionId": sid}).status_code == 400
    other = client.post("/v1/subscriptions/sync", json={"clientId": "c2", "subscriptionId": sid})
    assert other.status_code == 404


def test_stream_is_not_offered(client):
    sid = client.post("/v1/subscriptions", json={"clientId": "c1"}).json()["result"][
        "subscriptionId"
    ]
    r = client.post("/v1/subscriptions/stream", json={"clientId": "c1", "subscriptionId": sid})
    assert r.status_code == 501


def test_deleted_subscription_is_gone(client):
    sid = client.post("/v1/subscriptions", json={"clientId": "c1"}).json()["result"][
        "subscriptionId"
    ]
    d = client.post("/v1/subscriptions/delete", json={"clientId": "c1", "subscriptionIds": [sid]})
    assert d.json()["results"][0]["success"]
    r = client.post("/v1/subscriptions/sync", json={"clientId": "c1", "subscriptionId": sid})
    assert r.status_code == 404 and r.json()["responseDetail"]["status"] == 404
