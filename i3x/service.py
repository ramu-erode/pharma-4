"""i3x service: the i3X 1.0 read API over the UNS, historian and graph (ADR-0016).

    python -m i3x.service        # http://localhost:8600/v1/info

The broker feeds the last-value cache and the subscription queues. The graph gives the
address space, and TimescaleDB gives history.
"""

from __future__ import annotations

import logging

import uvicorn
from pydantic import BaseModel

from common import models as m
from common import uns
from common.mqtt import UnsClient
from common.settings import get_settings
from graph import db as graph_db
from i3x import catalog, values
from i3x.api import Backend, create_app
from i3x.store import PgHistory
from i3x.subscriptions import Hub

SERVICE = "i3x"
log = logging.getLogger(SERVICE)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    driver = graph_db.connect(settings)
    spaces = catalog.SpaceCache(lambda: catalog.read(driver))
    backend = Backend(
        space=spaces,
        cache=values.LiveCache(),
        history=PgHistory(settings.postgres_dsn),
        hub=Hub(),
        api_key=settings.i3x_api_key,
    )
    space = spaces()
    log.info("address space: %d objects", len(space.objects))

    client = UnsClient(service=SERVICE, src=m.Src.I3X, settings=settings)

    def on_message(topic: str, payload: BaseModel | None) -> None:
        if uns.parse(topic).kind is not uns.TopicKind.UNS:
            return
        backend.cache.update(topic, payload)
        if payload is not None:
            client.mark_processed(payload.ts)
        current = spaces.peek()
        if current is None:
            return
        backend.hub.publish(
            {"elementId": e, **values.own_vqt(current.objects[e], backend.cache)}
            for e in current.affected_by(topic)
        )

    client.subscribe(uns.SUB_UNS_ALL, on_message)
    client.start()
    log.info("serving i3X on port %d", settings.i3x_port)
    try:
        uvicorn.run(
            create_app(backend), host="0.0.0.0", port=settings.i3x_port, log_level="warning"
        )
    finally:
        client.close()
        driver.close()


if __name__ == "__main__":
    main()
