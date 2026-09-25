"""The i3X 1.0 HTTP API (ADR-0016): routes, envelopes, auth. No store access of its own.

Everything under `/v1`. Read-only: `PUT /objects/value|history` and
`/subscriptions/stream` answer 501, and `/info` declares them unsupported.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, BaseModel, Field
from starlette.exceptions import HTTPException

from i3x import values
from i3x.space import NAMESPACES, RELATIONSHIP_TYPES, TYPES, AddressSpace
from i3x.subscriptions import Hub, Subscription

SPEC_VERSION = "1.0"
SERVER_VERSION = "0.6.0"
HISTORY_LIMIT = 20_000  # points per element per request; more is truncated with a 206
HISTORY_BUDGET = 100_000  # points per request, across elements and components

TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    500: "Internal Server Error",
    501: "Not Implemented",
}


@dataclass
class Backend:
    """What the API reads. `space` is a callable so the address space can refresh."""

    space: Callable[[], AddressSpace]
    cache: values.LiveCache
    history: values.HistoryStore
    hub: Hub
    api_key: str | None = None
    history_limit: int = HISTORY_LIMIT
    history_budget: int = HISTORY_BUDGET


# --- envelopes ---------------------------------------------------------------------------------


def problem(status: int, detail: str) -> dict[str, Any]:
    return {"title": TITLES.get(status, "Error"), "status": status, "detail": detail}


def ok(result: Any) -> dict[str, Any]:
    return {"success": True, "result": result}


def item(key: str, id_: str, result: Any = None, error: tuple[int, str] | None = None) -> dict:
    if error is not None:
        return {"success": False, key: id_, "responseDetail": problem(*error)}
    return {"success": True, key: id_, "result": result}


def bulk(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"success": all(r["success"] for r in results), "results": results}


def not_found(what: str, id_: str) -> tuple[int, str]:
    return 404, f"{what} not found: {id_}"


# --- requests --------------------------------------------------------------------------------


class _Body(BaseModel):
    pass


class ElementIds(_Body):
    elementIds: list[str] = Field(min_length=1)


class ObjectList(ElementIds):
    includeMetadata: bool = False


class Related(ObjectList):
    relationshipType: str | None = None


class ValueQuery(ElementIds):
    maxDepth: int = Field(default=1, ge=0)


class HistoryQuery(ValueQuery):
    startTime: AwareDatetime
    endTime: AwareDatetime


class CreateSubscription(_Body):
    clientId: str = Field(min_length=1)
    displayName: str | None = None


class SubscriptionIds(_Body):
    clientId: str = Field(min_length=1)
    subscriptionIds: list[str] = Field(min_length=1)


class Registration(_Body):
    clientId: str = Field(min_length=1)
    subscriptionId: str
    elementIds: list[str] = Field(min_length=1)
    maxDepth: int = Field(default=1, ge=0)


class SyncRequest(_Body):
    clientId: str = Field(min_length=1)
    subscriptionId: str
    lastSequenceNumber: int | None = Field(default=None, ge=-1)


class StreamRequest(_Body):
    clientId: str = Field(min_length=1)
    subscriptionId: str


# --- app -------------------------------------------------------------------------------------


def create_app(backend: Backend) -> FastAPI:
    app = FastAPI(
        title="pharma-4 i3X",
        version=SERVER_VERSION,
        description="i3X 1.0 read API over the pharma-4 UNS, historian and knowledge graph.",
    )
    app.add_middleware(GZipMiddleware, minimum_size=0)  # the spec: gzip whenever asked

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            {"success": False, "responseDetail": problem(exc.status_code, str(exc.detail))},
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def _bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        detail = f"{where}: {first.get('msg', 'invalid request')}" if where else "invalid request"
        return JSONResponse(
            {"success": False, "responseDetail": problem(400, detail)}, status_code=400
        )

    def authenticate(request: Request) -> None:
        key = backend.api_key
        if not key:
            return
        given = request.headers.get("x-api-key") or ""
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            given = given or auth[7:].strip()
        if not hmac.compare_digest(given.encode(), key.encode()):
            raise HTTPException(401, "missing or invalid API key (X-API-Key or Bearer token)")

    public = APIRouter(prefix="/v1")
    api = APIRouter(prefix="/v1", dependencies=[Depends(authenticate)])

    @public.get("/info")
    def info() -> dict:
        return ok(
            {
                "specVersion": SPEC_VERSION,
                "serverVersion": SERVER_VERSION,
                "serverName": "pharma-4 i3X",
                "capabilities": {
                    "query": {"history": True},
                    "update": {"current": False, "history": False},
                    "subscribe": {"stream": False},
                },
            }
        )

    # -- exploratory --------------------------------------------------------------------------

    @api.get("/namespaces")
    def namespaces() -> dict:
        return ok(NAMESPACES)

    @api.get("/objecttypes")
    def object_types(namespaceUri: str | None = None) -> dict:
        return ok([t.to_json() for t in TYPES.values() if namespaceUri in (None, t.namespace)])

    @api.post("/objecttypes/query")
    def object_types_query(body: ElementIds) -> dict:
        return bulk(
            [
                item("elementId", i, TYPES[i].to_json())
                if i in TYPES
                else item("elementId", i, error=not_found("Object type", i))
                for i in body.elementIds
            ]
        )

    @api.get("/relationshiptypes")
    def relationship_types(namespaceUri: str | None = None) -> dict:
        return ok([r for r in RELATIONSHIP_TYPES if namespaceUri in (None, r["namespaceUri"])])

    @api.post("/relationshiptypes/query")
    def relationship_types_query(body: ElementIds) -> dict:
        by_id = {r["elementId"]: r for r in RELATIONSHIP_TYPES}
        return bulk(
            [
                item("elementId", i, by_id[i])
                if i in by_id
                else item("elementId", i, error=not_found("Relationship type", i))
                for i in body.elementIds
            ]
        )

    @api.get("/objects")
    def objects(
        typeElementId: str | None = None, includeMetadata: bool = False, root: bool = False
    ) -> dict:
        space = backend.space()
        found = space.roots() if root else space.objects.values()
        return ok(
            [
                o.to_json(includeMetadata)
                for o in found
                if typeElementId is None or o.type_id == typeElementId
            ]
        )

    @api.post("/objects/list")
    def objects_list(body: ObjectList) -> dict:
        space = backend.space()
        return bulk(
            [
                item("elementId", i, o.to_json(body.includeMetadata))
                if (o := space.get(i))
                else item("elementId", i, error=not_found("Element", i))
                for i in body.elementIds
            ]
        )

    @api.post("/objects/related")
    def objects_related(body: Related) -> dict:
        space = backend.space()
        return bulk(
            [
                item(
                    "elementId",
                    i,
                    [
                        {"sourceRelationship": rel, "object": o.to_json(body.includeMetadata)}
                        for rel, o in space.related(i, body.relationshipType)
                    ],
                )
                if i in space.objects
                else item("elementId", i, error=not_found("Element", i))
                for i in body.elementIds
            ]
        )

    # -- query --------------------------------------------------------------------------------

    @api.post("/objects/value")
    def objects_value(body: ValueQuery) -> dict:
        space = backend.space()
        return bulk(
            [
                item("elementId", i, values.current(space, backend.cache, i, body.maxDepth))
                if i in space.objects
                else item("elementId", i, error=not_found("Element", i))
                for i in body.elementIds
            ]
        )

    @api.post("/objects/history")
    def objects_history(body: HistoryQuery) -> JSONResponse:
        start, end = body.startTime.astimezone(UTC), body.endTime.astimezone(UTC)
        if start > end:
            raise HTTPException(400, "startTime is after endTime")
        space = backend.space()
        budget = values.Budget(backend.history_budget)
        results, truncated = [], False
        for i in body.elementIds:
            if i not in space.objects:
                results.append(item("elementId", i, error=not_found("Element", i)))
                continue
            result, cut = values.history(
                space, backend.history, i, start, end, body.maxDepth, backend.history_limit, budget
            )
            truncated = truncated or cut
            results.append(item("elementId", i, result))
        out = bulk(results)
        if truncated:
            out["responseDetail"] = problem(
                206,
                f"History was truncated: at most {backend.history_limit} points per element "
                f"and {backend.history_budget} per request, earliest first. Request fewer "
                "elements or a narrower time range for the rest.",
            ) | {"title": "Partial results returned"}
        return JSONResponse(out, status_code=206 if truncated else 200)

    # -- update (not offered) -------------------------------------------------------------------

    @api.put("/objects/value")
    def put_value() -> None:
        raise HTTPException(501, "pharma-4 i3X is read-only (ADR-0016)")

    @api.put("/objects/history")
    def put_history() -> None:
        raise HTTPException(501, "pharma-4 i3X is read-only (ADR-0016)")

    # -- subscribe ------------------------------------------------------------------------------

    def owned(client_id: str, subscription_id: str) -> Subscription:
        sub = backend.hub.get(client_id, subscription_id)
        if sub is None:
            raise HTTPException(404, f"Subscription not found: {subscription_id}")
        return sub

    @api.post("/subscriptions")
    def create_subscription(body: CreateSubscription) -> dict:
        sub = backend.hub.create(body.clientId, body.displayName)
        return ok(
            {
                "clientId": sub.client_id,
                "subscriptionId": sub.subscription_id,
                "displayName": sub.display_name,
            }
        )

    @api.post("/subscriptions/list")
    def list_subscriptions(body: SubscriptionIds) -> dict:
        results = []
        for sid in body.subscriptionIds:
            sub = backend.hub.get(body.clientId, sid)
            results.append(
                item("subscriptionId", sid, sub.describe())
                if sub
                else item("subscriptionId", sid, error=not_found("Subscription", sid))
            )
        return bulk(results)

    @api.post("/subscriptions/delete")
    def delete_subscriptions(body: SubscriptionIds) -> dict:
        return bulk(
            [
                item("subscriptionId", sid, None)
                if backend.hub.delete(body.clientId, sid)
                else item("subscriptionId", sid, error=not_found("Subscription", sid))
                for sid in body.subscriptionIds
            ]
        )

    @api.post("/subscriptions/register")
    def register(body: Registration) -> dict:
        sub = owned(body.clientId, body.subscriptionId)
        space = backend.space()
        results = []
        for i in body.elementIds:
            if i not in space.objects:
                results.append(item("elementId", i, error=not_found("Element", i)))
                continue
            ids = space.descendants(i, body.maxDepth)
            initial = [
                {"elementId": e, **values.own_vqt(space.objects[e], backend.cache)} for e in ids
            ]
            backend.hub.register(sub, i, body.maxDepth, ids, initial)
            results.append(item("elementId", i, None))
        return bulk(results)

    @api.post("/subscriptions/unregister")
    def unregister(body: Registration) -> dict:
        sub = owned(body.clientId, body.subscriptionId)
        space = backend.space()
        results = []
        for i in body.elementIds:
            if i not in space.objects:
                results.append(item("elementId", i, error=not_found("Element", i)))
                continue
            backend.hub.unregister(sub, i)
            results.append(item("elementId", i, None))
        return bulk(results)

    @api.post("/subscriptions/sync")
    def sync(body: SyncRequest) -> JSONResponse:
        sub = owned(body.clientId, body.subscriptionId)
        batches, dropped = backend.hub.sync(sub, body.lastSequenceNumber)
        out: dict[str, Any] = ok(batches)
        if dropped:
            out["responseDetail"] = {
                "title": "Updates dropped due to queue overflow",
                "status": 206,
                "detail": f"{dropped} update(s) were dropped; the queue holds "
                f"{backend.hub.max_queue}. Sync more often, or read the gap from history.",
            }
        return JSONResponse(out, status_code=206 if dropped else 200)

    @api.post("/subscriptions/stream")
    def stream(body: StreamRequest) -> None:
        owned(body.clientId, body.subscriptionId)
        raise HTTPException(501, "streaming is not offered; use /subscriptions/sync (ADR-0016)")

    app.include_router(public)
    app.include_router(api)
    return app
