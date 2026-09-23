"""Edge adapter service: edge/raw/# in, contextualised pv/* and sp/* out (ADR-0005).

python -m edge.adapter
"""

from __future__ import annotations

import logging
import signal
import threading

from common import models, uns
from common.models import RawSample, Src
from common.mqtt import UnsClient, connect
from common.settings import get_settings
from edge.core import Deadband, Mapped, TagMap, Unmapped, map_and_enrich

SERVICE = "edge-adapter"
log = logging.getLogger(SERVICE)


class EdgeAdapter:
    def __init__(self, client: UnsClient, tag_map: TagMap, deadband: Deadband) -> None:
        self.client = client
        self.tag_map = tag_map
        self.deadband = deadband
        self.batch_of_cell: dict[str, str | None] = {cell: None for cell in tag_map.cells}

    def start(self) -> None:
        self.publish_meta()
        # Batch cache first, so retained state/batch arrives before raw values.
        self.client.subscribe(uns.sub_state_batch(), self.on_batch)
        self.client.subscribe(uns.SUB_EDGE_RAW, self.on_raw, qos=0)

    def publish_meta(self) -> None:
        for e in self.tag_map.entries.values():
            meta = models.TagMetaPayload(
                v=models.TagMeta(
                    raw_tag=e.raw_tag, topic=e.topic, unit=e.unit, kind=e.kind, deadband=e.deadband
                ),
                ts=models.now_utc(),
                unit=None,
                batch=None,
                src=Src.EDGE,
            )
            self.client.publish(uns.meta_tag(e.unit_path.cell, e.cls, e.name), meta)

    def on_batch(self, topic: str, payload: object) -> None:
        if isinstance(payload, models.BatchStatePayload):
            cell = uns.parse(topic).unit.cell
            self.batch_of_cell[cell] = payload.v

    def on_raw(self, topic: str, payload: object) -> None:
        if not isinstance(payload, RawSample):
            return
        out = map_and_enrich(
            payload.tag, payload.value, payload.t, payload.q, self.tag_map, self.batch_of_cell
        )
        if isinstance(out, Unmapped):
            self.client.publish(uns.edge_unmapped(), payload)
            return
        if self.deadband.offer(out):
            self.client.publish(out.topic, to_payload(out))


def to_payload(m: Mapped) -> models.ScalarPayload:
    return models.ScalarPayload(v=m.value, ts=m.ts, unit=m.unit, q=m.q, batch=m.batch, src=Src.EDGE)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    s = get_settings()
    tag_map = TagMap.load(s.site, s.area, s.line)
    deadband = Deadband(s.deadband_floor_s, s.publish_period_s)
    client = connect(SERVICE, Src.EDGE, s)
    EdgeAdapter(client, tag_map, deadband).start()
    log.info("mapping %d raw tags for %s", len(tag_map.entries), sorted(tag_map.cells))

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    client.close()


if __name__ == "__main__":
    main()
