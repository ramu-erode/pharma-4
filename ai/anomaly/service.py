"""Anomaly service: scores every closed 30-minute window live and publishes into the UNS
(ADR-0014): `ai/anomaly/score` per window and `ai/anomaly/alert/<key>` per alert
lifecycle (ADR-0013).

    python -m ai.anomaly.service

Per unit it keeps the batch's published series and context (the same `Collector` the
training uses), scores a window once a value newer than its end has arrived, and runs
the model's layers through the same functions as training (parity). Neo4j is read once
per batch for limits and bindings, never per window. After a restart mid-batch, the
batch so far is rebuilt from the historian.
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
from dataclasses import dataclass, field
from datetime import timedelta

from pydantic import BaseModel

from ai import store
from ai.anomaly.detector import AlertChange, Detector, score_channels
from ai.anomaly.model import AnomalyModel
from ai.anomaly.windows import window_at
from ai.features import STEP_MIN, WINDOW_MIN, signal_of_topic
from ai.graph_context import batch_context
from ai.offline import Collector, from_history
from common import models as m
from common import uns
from common.mqtt import UnsClient, connect
from common.settings import Settings, get_settings
from common.uns import TopicClass, UnitPath
from graph import db as graph_db
from historian.migrate import connect as pg_connect

SERVICE = "anomaly"
log = logging.getLogger(SERVICE)


@dataclass
class UnitState:
    unit: UnitPath
    batch_id: str
    collector: Collector
    next_end: int = WINDOW_MIN
    detector: Detector | None = None
    controlled: dict[str, set[str]] = field(default_factory=dict)


class AnomalyService:
    def __init__(self, client: UnsClient, settings: Settings, model: AnomalyModel) -> None:
        self.client = client
        self.settings = settings
        self.model = model
        self.units: dict[str, UnitState] = {}
        self.inbox: queue.Queue[tuple[str, BaseModel | None]] = queue.Queue()
        try:
            self.driver = graph_db.connect(settings, wait_s=5)
        except Exception:
            log.warning("graph unavailable at start; limits and bindings from configuration")
            self.driver = None

    # -- inputs -------------------------------------------------------------------------------

    def subscribe(self) -> None:
        for pattern in (
            uns.sub_class(TopicClass.PV),
            uns.sub_class(TopicClass.SP),
            uns.sub_class(TopicClass.EVENTS),
        ):
            self.client.subscribe(pattern, lambda t, p: self.inbox.put((t, p)))

    def work(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                topic, payload = self.inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.handle(topic, payload)
            except Exception:
                log.exception("failed on %s", topic)

    def handle(self, topic: str, payload: BaseModel | None) -> None:
        if payload is None:
            return
        unit = uns.parse(topic).unit
        cell = unit.cell
        if isinstance(payload, m.BatchEventPayload):
            ev = payload.v
            if isinstance(ev, m.BatchStarted):
                self.units[cell] = UnitState(unit, ev.batch_id, Collector(payload.ts))
            state = self.units.get(cell)
            if state is None or state.batch_id != ev.batch_id:
                return
            state.collector.message(topic, payload)
            if isinstance(ev, m.BatchEnded):
                self._close(state, payload.ts)
                del self.units[cell]
            return
        if not isinstance(payload, m.ScalarPayload) or signal_of_topic(topic) is None:
            return
        if payload.batch is None:
            return  # an idle unit: nothing to score
        state = self.units.get(cell)
        if state is None or state.batch_id != payload.batch:
            state = self._recover(unit, payload.batch)
            if state is None:
                return
        minute = state.collector.minute(payload.ts)
        while minute > state.next_end:  # every value up to next_end has arrived
            self._score(state, state.next_end)
            state.next_end += STEP_MIN
        state.collector.value(topic, payload.ts, payload.v, payload.q.value)

    def _recover(self, unit: UnitPath, batch_id: str) -> UnitState | None:
        """Joined mid-batch (a restart): rebuild the batch so far from the historian."""
        try:
            with pg_connect(self.settings.postgres_dsn, wait_s=5) as conn:
                c = from_history(conn, batch_id)
        except Exception:
            log.warning("cannot recover %s from history yet", batch_id)
            return None
        col = Collector(c.series.origin)
        col.series, col.ctx, col.recipe_id = c.series, c.ctx, c.recipe_id
        last = int(c.series.last_minute)
        state = UnitState(unit, batch_id, col, next_end=max(WINDOW_MIN, last - last % STEP_MIN))
        self.units[unit.cell] = state
        log.info("%s: recovered %s from history up to minute %d", unit.cell, batch_id, last)
        return state

    # -- scoring ------------------------------------------------------------------------------

    def _detector(self, state: UnitState) -> Detector:
        if state.detector is None:
            limits, controlled = batch_context(
                self.driver, state.batch_id, state.unit.cell, state.collector.recipe_id
            )
            state.controlled = controlled
            state.detector = Detector(self.model, state.batch_id, limits)
        return state.detector

    def _score(self, state: UnitState, end: int) -> None:
        det = self._detector(state)
        w = window_at(state.collector.series, state.collector.ctx, state.controlled, end)
        if w is None:
            return
        index, changes, _ = det.step(w, score_channels(self.model, [w]), 0)
        ts = state.collector.series.origin + timedelta(minutes=end)
        self.client.publish(
            uns.ai_score(state.unit),
            m.ScalarPayload(
                v=float(index), ts=ts, unit=None, batch=state.batch_id, src=m.Src.ANOMALY
            ),
        )
        for change in changes:
            self._publish(state, change)

    def _publish(self, state: UnitState, change: AlertChange) -> None:
        origin = state.collector.series.origin
        ev = change.evidence
        topic = uns.ai_alert(state.unit, change.key)
        alert = m.Alert(
            id=change.alert_id,
            key=change.key,
            state=change.state,
            layer=ev.layer,
            score=float(ev.score),
            threshold=float(ev.threshold),
            fault_class=ev.fault_class,
            top_tags=ev.top_tags[:3],
            opened_at=origin + timedelta(minutes=change.opened_end),
            cleared_at=origin + timedelta(minutes=change.end)
            if change.state == "CLEARED"
            else None,
        )
        ts = origin + timedelta(minutes=change.end)
        self.client.publish(
            topic,
            m.AlertPayload(v=alert, ts=ts, unit=None, batch=state.batch_id, src=m.Src.ANOMALY),
        )
        if change.state == "CLEARED":
            self.client.clear_retained(topic)  # the open set on the broker (ADR-0013)
        log.info("%s %s %s (%s)", state.batch_id, change.state, change.key, ev.fault_class)

    def _close(self, state: UnitState, ts) -> None:
        """Batch over: clear whatever is still open."""
        if state.detector is None:
            return
        end = int(state.collector.minute(ts))
        for key in state.detector.alerts.open_keys:
            st = state.detector.alerts.keys[key]
            self._publish(
                state,
                AlertChange(key, "CLEARED", st.open_id, st.evidence, end, st.opened_end),
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    model = store.load(settings.models_dir, "anomaly")
    if model is None:
        raise SystemExit(f"no anomaly model in {settings.models_dir}; run python -m ai.train_all")
    client = connect(SERVICE, m.Src.ANOMALY, settings)
    service = AnomalyService(client, settings, model)
    stop = threading.Event()
    worker = threading.Thread(target=service.work, args=(stop,), name="score")
    worker.start()
    service.subscribe()
    log.info("scoring with %s", model.version)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    client.close()
    worker.join(timeout=5)


if __name__ == "__main__":
    main()
