"""Against the running broker (`docker compose up -d mosquitto`; `pytest -m compose`)."""

import contextlib
import queue
import time

import psycopg
import pytest

from common import models as m
from common import uns
from common.mqtt import UnsClient
from common.settings import get_settings
from tests.samples import BATCH, TS, UNIT

pytestmark = pytest.mark.compose


def client(service: str, src: m.Src) -> UnsClient:
    return UnsClient(service=service, src=src, settings=get_settings()).start(wait_s=10)


def collect(c: UnsClient, pattern: str) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    c.subscribe(pattern, lambda topic, payload: q.put((topic, payload)))
    time.sleep(0.3)
    return q


def drain(q: queue.Queue, wait_s: float = 1.0) -> list:
    out, deadline = [], time.monotonic() + wait_s
    while time.monotonic() < deadline:
        with contextlib.suppress(queue.Empty):
            out.append(q.get(timeout=0.05))
    return out


def test_heartbeat_is_retained_and_last_will_fires():
    sim = client("simulator", m.Src.SIM)
    watcher = client("dashboard", m.Src.DASHBOARD)
    try:
        q = collect(watcher, uns.meta_status("simulator"))
        states = [p.v.state for _, p in drain(q) if p is not None]
        assert "online" in states
        sim._client._sock.close()  # simulate a crash: no clean disconnect
        states = [p.v.state for _, p in drain(q, 5.0) if p is not None]
        assert "offline" in states
    finally:
        sim._stop.set()
        sim._client.loop_stop()
        watcher.close()


def test_owner_publish_is_delivered():
    sim = client("simulator", m.Src.SIM)
    anomaly = client("anomaly", m.Src.ANOMALY)
    try:
        q = collect(anomaly, uns.lab(UNIT, "titer"))
        payload = m.ScalarPayload(v=3.3, ts=TS, unit="g/L", batch=BATCH, src=m.Src.SIM)
        sim.publish(uns.lab(UNIT, "titer"), payload).wait_for_publish(2)
        assert (uns.lab(UNIT, "titer"), payload) in drain(q)
    finally:
        sim.close()
        anomaly.close()


def test_non_owner_publish_is_dropped():
    sim = client("simulator", m.Src.SIM)
    explorer = client("explorer", m.Src.OPERATOR)
    try:
        q = collect(explorer, uns.pv(UNIT, "temperature"))
        drain(q, 0.5)  # discard any retained value from earlier runs
        fake = m.ScalarPayload(v=99.0, ts=TS, unit="°C", batch=BATCH, src=m.Src.SIM)
        sim.publish(uns.pv(UNIT, "temperature"), fake)  # pv/* belongs to the edge adapter
        assert all(p != fake for _, p in drain(q))
    finally:
        sim.close()
        explorer.close()


def test_ai_services_never_receive_ground_truth():
    sim = client("simulator", m.Src.SIM)
    anomaly = client("anomaly", m.Src.ANOMALY)
    try:
        q = collect(anomaly, "_sim/#")
        label = m.FaultLabelPayload(
            v=m.FaultLabel(id="F-t", fault="stuck_sensor", cell="BR-101", batch_id=BATCH, onset=TS),
            ts=TS,
            unit=None,
            batch=BATCH,
            src=m.Src.SIM,
        )
        sim.publish(uns.sim_faults("BR-101"), label).wait_for_publish(2)
        assert drain(q) == []
    finally:
        sim.close()
        anomaly.close()
        forget_test_label("F-t")


def forget_test_label(label_id: str) -> None:
    """The historian stores what the test published; leave the stack's history clean."""
    time.sleep(1.5)  # let the historian flush first
    with psycopg.connect(get_settings().postgres_dsn, autocommit=True) as conn:
        conn.execute("DELETE FROM fault_labels WHERE id = %s", (label_id,))
