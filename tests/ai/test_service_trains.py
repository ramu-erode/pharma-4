"""The live anomaly service scores a train batch exactly as training does (parity,
ADR-0021), even when a unit's first values reach it before the event that says the batch
arrived there: those come from the edge adapter and the simulator, on different
connections, so the broker does not order them."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta

import pytest

from ai.anomaly import service as svc
from ai.anomaly import train
from ai.offline import from_engine_units, to_input
from common import models as m
from common import uns
from common.settings import Settings
from edge.adapter import to_payload
from edge.core import Deadband, Mapped, TagMap, map_and_enrich
from simulator import recipes
from simulator.messages import RawOut
from simulator.osd.engine import OsdRun
from simulator.train import TrainSpec
from simulator.wire import to_wire
from tests.ai.harness_trains import harness

pytestmark = pytest.mark.slow
S = Settings()


class Client:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    def publish(self, topic: str, payload: object) -> None:
        self.published.append((topic, payload))

    def clear_retained(self, topic: str) -> None:
        pass


def uns_messages(spec: TrainSpec) -> list[tuple[str, object]]:
    """What the broker would deliver for a batch, in publish order."""
    run = OsdRun(spec)
    tag_map, deadband = TagMap.load(), Deadband(S.deadband_floor_s, S.publish_period_s)
    cache: dict[str, str | None] = dict.fromkeys(run.cells)
    unit = uns.parse(uns.state_batch(tag_map.entries["BL301.SIC-301.PV"].unit_path)).unit
    out: list[tuple[str, object]] = []
    for msg in run.run_to_end():
        if isinstance(msg, RawOut):
            mp = map_and_enrich(msg.tag, msg.value, msg.t, msg.q, tag_map, cache)
            if isinstance(mp, Mapped) and deadband.offer(mp):
                out.append((mp.topic, to_payload(mp)))
            continue
        wire = to_wire(unit, spec.batch_id, msg)
        if wire is not None:
            if wire[0].endswith("/state/batch"):
                cache[uns.parse(wire[0]).unit.cell] = wire[1].v
            out.append(wire)
    return out


def events_late(messages: list[tuple[str, object]]) -> list[tuple[str, object]]:
    """Deliver each `events/batch` message after the next few process values."""
    out, held = [], []
    for topic, payload in messages:
        if topic.endswith("/events/batch"):
            held.append((topic, payload))
            continue
        out.append((topic, payload))
        if held and isinstance(payload, m.ScalarPayload) and payload.batch:
            out += held
            held = []
    return out + held


def test_live_scoring_matches_offline_when_events_arrive_late(monkeypatch):
    h = harness()
    model = h.models["tablet_press"]
    monkeypatch.setattr(svc.graph_db, "connect", lambda *a, **k: (_ for _ in ()).throw(OSError))
    spec = TrainSpec(
        "B2026-0950", "BL-301", datetime(2026, 11, 1, tzinfo=UTC), "tab-v2",
        m.Campaign.MFG, recipes.get("tab-v2").nominal, seed=77,
    )  # fmt: skip
    client = Client()
    service = svc.AnomalyService(client, S, {"tablet_press": model})
    for topic, payload in events_late(uns_messages(spec)):
        service.handle(topic, payload)

    live = {p.v.key for t, p in client.published if "/alert/" in t and p.v.state == "OPEN"}
    c = from_engine_units(spec, S, TagMap.load())["TP-303"]
    offline = {ch.key for ch in train.run(model, to_input(spec.batch_id, c)).changes
               if ch.state == "OPEN"}  # fmt: skip
    assert live == offline
    scores = [t for t, _ in client.published if t.endswith("/ai/anomaly/score")]
    assert scores and all("/TP-303/" in t for t in scores)


def test_recovery_from_a_historian_ahead_keeps_the_series_in_order(monkeypatch):
    """The service missed a unit's arrival (a restart), and the historian is ahead of it:
    it recovers the batch so far, then skips the backlog it already has."""
    h = harness()
    model = h.models["tablet_press"]
    monkeypatch.setattr(svc.graph_db, "connect", lambda *a, **k: (_ for _ in ()).throw(OSError))
    spec = TrainSpec(
        "B2026-0951", "BL-301", datetime(2026, 11, 2, tzinfo=UTC), "tab-v2",
        m.Campaign.MFG, recipes.get("tab-v2").nominal, seed=78,
    )  # fmt: skip
    msgs = uns_messages(spec)
    arrival = next(p.ts for t, p in msgs if t.endswith("TP-303/events/batch"))
    cut = arrival + timedelta(minutes=100)

    historian = svc.AnomalyService(Client(), S, {"tablet_press": model})
    for topic, payload in msgs:
        if payload.ts <= cut:
            historian.handle(topic, payload)
    so_far = historian.units["TP-303"].collector.collected()
    monkeypatch.setattr(svc, "from_history", lambda conn, batch, cell: so_far)
    monkeypatch.setattr(svc, "pg_connect", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(svc, "MAX_PENDING", 0)

    client = Client()
    live = svc.AnomalyService(client, S, {"tablet_press": model})
    for topic, payload in msgs:
        if topic.endswith("TP-303/events/batch") and payload.ts == arrival:
            continue  # missed while it was down
        live.handle(topic, payload)

    c = from_engine_units(spec, S, TagMap.load())["TP-303"]
    bi = to_input(spec.batch_id, c)
    offline = {w.end: i for w, i in zip(bi.windows, train.run(model, bi).index, strict=True)}
    scored = {
        round((p.ts - c.series.origin).total_seconds() / 60): p.v
        for t, p in client.published
        if t.endswith("TP-303/ai/anomaly/score")
    }
    assert scored and min(scored) >= 95  # from the last full step the historian had
    # Within the EWMA's memory, which restarts at recovery; before the fix the index
    # climbed past 100 because the backlog landed out of time order.
    for minute, value in scored.items():
        assert value == pytest.approx(offline[minute], rel=0.05), minute
