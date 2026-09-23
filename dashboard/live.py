"""The dashboard's view of the broker: one background MQTT client (the `dashboard` user,
the only one allowed to write `_sim/cmd`, ADR-0012) holding the latest message per topic.
Renders never call MQTT; they read a snapshot."""

from __future__ import annotations

import threading
from collections import Counter

from pydantic import BaseModel

from common import models as m
from common import uns
from common.mqtt import UnsClient
from common.settings import Settings
from common.uns import SimCommand

SUBSCRIPTIONS = (uns.SUB_UNS_ALL, "edge/#", uns.sim_clock(), uns.SUB_SIM_FAULTS)


class LiveState:
    def __init__(self, settings: Settings) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, BaseModel] = {}
        self.counts: Counter[str] = Counter()
        self.client = UnsClient(service="dashboard", src=m.Src.DASHBOARD, settings=settings)
        for pattern in SUBSCRIPTIONS:
            self.client.subscribe(pattern, self._on)
        self.client.start()

    def _on(self, topic: str, payload: BaseModel | None) -> None:
        with self._lock:
            self.counts[topic.split("/")[0] if topic.startswith(("edge", "_sim")) else "uns"] += 1
            if payload is None:
                self._latest.pop(topic, None)  # a retained clear: the item is gone
            else:
                self._latest[topic] = payload

    def snapshot(self, prefix: str = "") -> dict[str, BaseModel]:
        with self._lock:
            return {t: p for t, p in self._latest.items() if t.startswith(prefix)}

    def get(self, topic: str) -> BaseModel | None:
        with self._lock:
            return self._latest.get(topic)

    def clock(self) -> m.ClockStatus | None:
        p = self.get(uns.sim_clock())
        return p.v if p is not None else None

    def send(self, kind: SimCommand, command: BaseModel) -> None:
        """A person's command to the simulator (src = operator)."""
        clock = self.clock()
        payload = m.SIM_COMMAND_MODELS[kind](
            v=command,
            ts=clock.sim_time if clock else m.now_utc(),
            unit=None,
            batch=None,
            src=m.Src.OPERATOR,
        )
        self.client.publish(uns.sim_cmd(kind), payload)
