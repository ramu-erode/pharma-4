"""Anomaly service: scores every closed 30-minute window live and publishes into the UNS
(ADR-0015): `ai/anomaly/score` per window and `ai/anomaly/alert/<key>` per alert
lifecycle (ADR-0013).

    python -m ai.anomaly.service

One service for the whole enterprise, with one model per equipment class (ADR-0021).
Per unit it keeps the batch's published series and context (the same `Collector` the
training uses) from when the batch arrived on that unit, scores a window once a value
newer than its end has arrived, and runs the model's layers through the same functions
as training (parity). Neo4j is read once per batch and unit for limits and bindings,
never per window. After a restart mid-batch, the batch so far is rebuilt from the
historian.
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
from ai.features import STEP_MIN, WINDOW_MIN
from ai.graph_context import batch_context
from ai.offline import Collector, from_history
from ai.profiles import PROFILES
from common import models as m
from common import uns
from common.mqtt import UnsClient, connect
from common.plant import get_plant
from common.settings import Settings, get_settings
from common.uns import TopicClass, UnitPath
from graph import db as graph_db
from historian.migrate import connect as pg_connect
from simulator import recipes

SERVICE = "anomaly"
log = logging.getLogger(SERVICE)
# Values of a batch a unit has not seen arrive yet: held until its arrival event, which
# comes from the simulator on another connection and can land after the edge adapter's
# first values. Beyond this many, the service assumes a restart mid-batch.
MAX_PENDING = 400


@dataclass
class UnitState:
    unit: UnitPath
    batch_id: str
    collector: Collector
    model: AnomalyModel
    next_end: int = WINDOW_MIN
    detector: Detector | None = None
    controlled: dict[str, set[str]] = field(default_factory=dict)
    # After a recovery from the historian: messages up to here are already in the series.
    recovered_to: float = float("-inf")


class AnomalyService:
    def __init__(
        self, client: UnsClient, settings: Settings, models: dict[str, AnomalyModel]
    ) -> None:
        self.client = client
        self.settings = settings
        self.models = models  # by equipment class
        self.plant = get_plant()
        self.units: dict[str, UnitState] = {}
        self.recipe_of: dict[str, str] = {}  # batch -> recipe, from its BATCH_START
        self.pending: dict[tuple[str, str], list[tuple[str, m.ScalarPayload]]] = {}
        self.left: set[tuple[str, str]] = set()  # (unit, batch) pairs that have finished
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

    def _model(self, cell: str) -> AnomalyModel | None:
        try:
            return self.models.get(self.plant.unit(cell).cls)
        except KeyError:
            return None

    def _recipe(self, batch_id: str, cell: str) -> str:
        known = self.recipe_of.get(batch_id)
        if known:
            return known
        process = self.plant.unit(cell).process
        return recipes.current(m.now_utc().date(), process).id

    def handle(self, topic: str, payload: BaseModel | None) -> None:
        if payload is None:
            return
        unit = uns.parse(topic).unit
        cell = unit.cell
        if isinstance(payload, m.BatchEventPayload):
            ev = payload.v
            if isinstance(ev, m.BatchStarted):
                self.recipe_of[ev.batch_id] = ev.recipe
            model = self._model(cell)
            if model is None:
                return
            arrived = isinstance(ev, m.BatchStarted) or (
                isinstance(ev, m.OperationChanged)
                and ev.previous is m.Operation.IDLE
                and self.units.get(cell, None) is None
            )
            if arrived:  # the batch starts, or arrives on this unit of its train
                col = Collector(payload.ts, model.profile, self._recipe(ev.batch_id, cell))
                col.cell = cell
                self.units[cell] = UnitState(unit, ev.batch_id, col, model)
            state = self.units.get(cell)
            if state is None or state.batch_id != ev.batch_id:
                return
            if state.collector.minute(payload.ts) > state.recovered_to:
                state.collector.message(topic, payload)
            for t, p in self.pending.pop((cell, ev.batch_id), []):
                self._value(state, t, p)
            left = isinstance(ev, m.BatchEnded) or (
                isinstance(ev, m.OperationChanged) and ev.current is m.Operation.IDLE
            )
            if left:
                self._close(state, payload.ts)
                del self.units[cell]
                self.left.add((cell, ev.batch_id))
                self.pending.pop((cell, ev.batch_id), None)
            return
        if not isinstance(payload, m.ScalarPayload) or payload.batch is None:
            return  # an idle unit: nothing to score
        model = self._model(cell)
        if model is None or model.profile.signal_of(topic) is None:
            return
        state = self.units.get(cell)
        if state is None or state.batch_id != payload.batch:
            if (cell, payload.batch) in self.left:
                return  # a straggler from a batch that has left this unit
            held = self.pending.setdefault((cell, payload.batch), [])
            held.append((topic, payload))
            if len(held) <= MAX_PENDING:
                return  # its arrival event is probably still on its way
            log.info(
                "%s: %d values of %s but no arrival seen (holding %s); recovering",
                cell, len(held), payload.batch, state.batch_id if state else "nothing",
            )  # fmt: skip
            state = self._recover(unit, payload.batch, model)
            del self.pending[(cell, payload.batch)]
            if state is None:
                return
        self._value(state, topic, payload)

    def _value(self, state: UnitState, topic: str, payload: m.ScalarPayload) -> None:
        minute = state.collector.minute(payload.ts)
        if minute <= state.recovered_to:
            return  # the historian already had it; the series must stay in time order
        while minute > state.next_end:  # every value up to next_end has arrived
            self._score(state, state.next_end)
            state.next_end += STEP_MIN
        state.collector.value(topic, payload.ts, payload.v, payload.q.value)

    def _recover(self, unit: UnitPath, batch_id: str, model: AnomalyModel) -> UnitState | None:
        """Joined mid-batch (a restart): rebuild the batch so far from the historian."""
        try:
            with pg_connect(self.settings.postgres_dsn, wait_s=5) as conn:
                c = from_history(conn, batch_id, unit.cell)
        except Exception:
            log.warning("cannot recover %s on %s from history yet", batch_id, unit.cell)
            return None
        col = Collector(c.series.origin, model.profile, c.recipe_id)
        col.series, col.ctx, col.cell = c.series, c.ctx, unit.cell
        last = int(c.series.last_minute)
        state = UnitState(
            unit, batch_id, col, model, next_end=max(WINDOW_MIN, last - last % STEP_MIN),
            recovered_to=c.series.last_minute,
        )  # fmt: skip
        self.units[unit.cell] = state
        log.info("%s: recovered %s from history up to minute %d", unit.cell, batch_id, last)
        return state

    # -- scoring ------------------------------------------------------------------------------

    def _detector(self, state: UnitState) -> Detector:
        if state.detector is None:
            limits, controlled = batch_context(
                self.driver,
                state.batch_id,
                state.unit.cell,
                state.collector.recipe_id,
                state.model.cls,
            )
            state.controlled = controlled
            state.detector = Detector(state.model, state.batch_id, limits)
        return state.detector

    def _score(self, state: UnitState, end: int) -> None:
        det = self._detector(state)
        w = window_at(
            state.collector.series,
            state.collector.ctx,
            state.controlled,
            end,
            state.model.profile,
        )
        if w is None:
            return
        index, changes, _ = det.step(w, score_channels(state.model, [w]), 0)
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


def load_models(models_dir: str) -> dict[str, AnomalyModel]:
    out = {}
    for cls in PROFILES:
        model = store.load(models_dir, store.anomaly_name(cls))
        if model is not None:
            out[cls] = model
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    models = load_models(settings.models_dir)
    if not models:
        raise SystemExit(f"no anomaly model in {settings.models_dir}; run python -m ai.train_all")
    client = connect(SERVICE, m.Src.ANOMALY, settings)
    service = AnomalyService(client, settings, models)
    stop = threading.Event()
    worker = threading.Thread(target=service.work, args=(stop,), name="score")
    worker.start()
    service.subscribe()
    log.info("scoring with %s", ", ".join(f"{c}: {mo.version}" for c, mo in models.items()))
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    client.close()
    worker.join(timeout=5)


if __name__ == "__main__":
    main()
