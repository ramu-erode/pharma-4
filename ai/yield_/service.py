"""Yield service: mid-batch titer band every 6 simulated hours and gated setpoint advice
(ADR-0015). Advisory only: it publishes `ai/yield/prediction` and a retained
`ai/yield/recommendation`, and never writes to control.

    python -m ai.yield_.service

Keeps each running batch's series, lab results, levers (planned + operator changes) and
alerts so far, from the UNS. PARs come from the graph once per batch.
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from pydantic import BaseModel

from ai import store
from ai.offline import Collector, from_history
from ai.yield_ import optimize
from ai.yield_.features import FIRST_DAY, LAST_DAY, YieldInput, features_at
from ai.yield_.model import YieldModel
from common import models as m
from common import uns
from common.mqtt import UnsClient, connect
from common.settings import Settings, get_settings
from common.uns import TopicClass, UnitPath
from graph import db as graph_db
from historian.migrate import connect as pg_connect
from simulator import recipes

SERVICE = "yield"
EVERY_DAYS = 0.25  # six simulated hours
log = logging.getLogger(SERVICE)

PARS = """
MATCH (:Batch {id: $batch})-[:FOLLOWS]->(:Recipe)-[:HAS_LIMIT]->(l:SpecLimit {type: 'PAR'})
RETURN l.parameter AS name, l.low AS low, l.high AS high
"""


@dataclass
class UnitState:
    unit: UnitPath
    batch_id: str
    collector: Collector
    alerts: list[float] = field(default_factory=list)
    next_day: float = FIRST_DAY
    par: dict[str, tuple[float, float]] | None = None
    has_recommendation: bool = False


def quantiles(values: np.ndarray) -> m.Quantiles:
    p10, p50, p90 = np.quantile(values, [0.1, 0.5, 0.9])
    return m.Quantiles(p10=float(p10), p50=float(p50), p90=float(p90))


class YieldService:
    def __init__(self, client: UnsClient, settings: Settings, model: YieldModel) -> None:
        self.client, self.settings, self.model = client, settings, model
        self.units: dict[str, UnitState] = {}
        self.inbox: queue.Queue[tuple[str, BaseModel | None]] = queue.Queue()
        try:
            self.driver = graph_db.connect(settings, wait_s=5)
        except Exception:
            self.driver = None

    def subscribe(self) -> None:
        for pattern in (
            uns.sub_class(TopicClass.PV),
            uns.sub_class(TopicClass.SP),
            uns.sub_class(TopicClass.LAB),
            uns.sub_class(TopicClass.EVENTS),
            f"{uns.ENTERPRISE}/+/+/+/+/ai/anomaly/alert/+",
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
        if isinstance(payload, m.BatchEventPayload) and isinstance(payload.v, m.BatchStarted):
            self.units[cell] = UnitState(unit, payload.v.batch_id, Collector(payload.ts))
        batch = getattr(payload, "batch", None)
        if batch is None:
            return
        state = self.units.get(cell)
        if state is None or state.batch_id != batch:
            state = self._recover(unit, batch)
            if state is None:
                return
        if isinstance(payload, m.AlertPayload):
            if payload.v.state == "OPEN" and not payload.v.key.startswith("rules-quality"):
                state.alerts.append(state.collector.minute(payload.v.opened_at))
            return
        if isinstance(payload, m.BatchEventPayload | m.OperatorEventPayload):
            state.collector.message(topic, payload)
            if isinstance(payload.v, m.BatchEnded):
                if state.has_recommendation:
                    self.client.clear_retained(uns.ai_recommendation(unit))
                del self.units[cell]
            return
        if isinstance(payload, m.ScalarPayload):
            state.collector.value(topic, payload.ts, payload.v, payload.q.value)
            self._maybe_predict(state)

    def _recover(self, unit: UnitPath, batch_id: str) -> UnitState | None:
        try:
            with pg_connect(self.settings.postgres_dsn, wait_s=5) as conn:
                c = from_history(conn, batch_id)
        except Exception:
            return None
        col = Collector(c.series.origin)
        col.series, col.ctx, col.recipe_id, col.lab = c.series, c.ctx, c.recipe_id, c.lab
        col.planned = c.levers
        state = UnitState(unit, batch_id, col)
        inoc = col.ctx.inoculation_min
        if inoc is not None:  # skip predictions for the part already past
            day = (col.series.last_minute - inoc) / 1440.0
            state.next_day = max(FIRST_DAY, EVERY_DAYS * np.ceil(day / EVERY_DAYS))
        self.units[unit.cell] = state
        log.info("%s: recovered %s from history", unit.cell, batch_id)
        return state

    # -- prediction and advice ----------------------------------------------------------------

    def _maybe_predict(self, state: UnitState) -> None:
        col = state.collector
        inoc = col.ctx.inoculation_min
        if inoc is None or col.levers is None or state.next_day > LAST_DAY:
            return
        if col.series.last_minute < inoc + state.next_day * 1440.0:
            return
        day = state.next_day
        state.next_day += EVERY_DAYS
        x = features_at(YieldInput(col.series, col.ctx, col.lab, col.levers, state.alerts), day)
        ts = col.series.origin + timedelta(minutes=inoc + day * 1440.0)
        band = self.model.band(x)
        self.client.publish(
            uns.ai_prediction(state.unit),
            m.PredictionPayload(
                v=m.Prediction(
                    titer=m.Quantiles(
                        p10=float(band[0.1][0]), p50=float(band[0.5][0]), p90=float(band[0.9][0])
                    ),
                    batch_day=day,
                    model_version=self.model.version,
                ),
                ts=ts,
                unit="g/L",
                batch=state.batch_id,
                src=m.Src.YIELD,
            ),
        )
        operation = col.ctx.operation_at(col.series.last_minute)
        if operation not in ("Growth", "TempShift", "Production"):
            return
        advice = optimize.recommend(
            self.model, x, col.levers, day, operation, self._par(state), seed=int(day * 100)
        )
        topic = uns.ai_recommendation(state.unit)
        if not advice.passes_gate:
            if state.has_recommendation:
                self.client.clear_retained(topic)
                state.has_recommendation = False
            return
        rec = m.Recommendation(
            id=f"R-{state.batch_id}-d{day:.2f}",
            levers={
                k: m.LeverAdvice(
                    current=advice.current[k],
                    recommended=advice.recommended[k],
                    frozen=k in advice.frozen,
                )
                for k in advice.current
            },
            predicted_current=quantiles(advice.predicted_current),
            predicted_recommended=quantiles(advice.predicted_recommended),
            gain=quantiles(advice.gain),
            model_version=self.model.version,
        )
        self.client.publish(
            topic,
            m.RecommendationPayload(
                v=rec, ts=ts, unit="g/L", batch=state.batch_id, src=m.Src.YIELD
            ),
        )
        state.has_recommendation = True
        log.info("%s day %.2f: recommend %s (gain P50 %+.2f)", state.batch_id, day,
                 rec.id, rec.gain.p50)  # fmt: skip

    def _par(self, state: UnitState) -> dict[str, tuple[float, float]]:
        """PARs from the graph once per batch (ADR-0015), configuration as fallback."""
        if state.par is None:
            par = {}
            if self.driver is not None:
                try:
                    recs, _, _ = self.driver.execute_query(PARS, batch=state.batch_id)
                    par = {r["name"]: (float(r["low"]), float(r["high"])) for r in recs}
                except Exception:
                    par = {}
            state.par = par or recipes.get(state.collector.recipe_id).par
        return state.par


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    model = store.load(settings.models_dir, "yield")
    if model is None:
        raise SystemExit(f"no yield model in {settings.models_dir}; run python -m ai.train_all")
    client = connect(SERVICE, m.Src.YIELD, settings)
    service = YieldService(client, settings, model)
    stop = threading.Event()
    worker = threading.Thread(target=service.work, args=(stop,), name="predict")
    worker.start()
    service.subscribe()
    log.info("predicting with %s", model.version)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    client.close()
    worker.join(timeout=5)


if __name__ == "__main__":
    main()
