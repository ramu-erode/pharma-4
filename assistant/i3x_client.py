"""A small i3X 1.0 client: the assistant's only way to see the plant (ADR-0017)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from common.models import format_ts


class I3xError(RuntimeError):
    """The server answered with an i3X error envelope, or not at all."""


class I3xClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        http: httpx.Client | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._base = base_url.rstrip("/")
        self._http = http or httpx.Client(timeout=timeout_s)
        self._headers = headers

    def _call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        # A full URL, so a TestClient (which has its own base_url) and a plain client
        # both resolve it the same way.
        try:
            r = self._http.request(method, self._base + path, headers=self._headers, **kwargs)
        except httpx.HTTPError as exc:
            raise I3xError(f"i3X server unreachable at {self._base}: {exc}") from exc
        try:
            body = r.json()
        except ValueError as exc:
            raise I3xError(f"{method} {path}: HTTP {r.status_code}, not JSON") from exc
        if r.status_code >= 400:
            detail = (body.get("responseDetail") or {}).get("detail", r.text)
            raise I3xError(f"{method} {path}: HTTP {r.status_code}: {detail}")
        return body

    def _results(self, path: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        return self._call("POST", path, json=body)["results"]

    # -- exploratory ---------------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return self._call("GET", "/info")["result"]

    def object_types(self) -> list[dict[str, Any]]:
        return self._call("GET", "/objecttypes")["result"]

    def relationship_types(self) -> list[dict[str, Any]]:
        return self._call("GET", "/relationshiptypes")["result"]

    def objects(self, type_element_id: str | None = None, root: bool = False) -> list[dict]:
        params: dict[str, Any] = {}
        if type_element_id:
            params["typeElementId"] = type_element_id
        if root:
            params["root"] = "true"
        return self._call("GET", "/objects", params=params)["result"]

    def list(self, element_ids: list[str], metadata: bool = True) -> list[dict[str, Any]]:
        return self._results(
            "/objects/list", {"elementIds": element_ids, "includeMetadata": metadata}
        )

    def related(self, element_ids: list[str], relationship: str | None = None) -> list[dict]:
        body: dict[str, Any] = {"elementIds": element_ids}
        if relationship:
            body["relationshipType"] = relationship
        return self._results("/objects/related", body)

    # -- query ------------------------------------------------------------------------------------

    def value(self, element_ids: list[str], max_depth: int = 1) -> list[dict[str, Any]]:
        return self._results("/objects/value", {"elementIds": element_ids, "maxDepth": max_depth})

    def history(
        self, element_ids: list[str], start: datetime, end: datetime
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Results, and the server's note if it truncated them (HTTP 206)."""
        body = self._call(
            "POST",
            "/objects/history",
            json={
                "elementIds": element_ids,
                "startTime": format_ts(start),
                "endTime": format_ts(end),
            },
        )
        note = (body.get("responseDetail") or {}).get("detail")
        return body["results"], note
