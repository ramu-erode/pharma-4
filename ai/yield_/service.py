"""Yield service: mid-batch prediction bands and gated setpoint advice for every process
(ADR-0015, ADR-0021). Advisory only: it publishes `ai/yield/prediction` and a retained
`ai/yield/recommendation`, and never writes to control.

    python -m ai.yield_.service

A bioreactor batch is followed on its unit: series, lab results, levers (planned +
operator changes) and alerts so far, predicted every 6 simulated hours from day 3. A
train batch (ADR-0018) is followed across its units and predicted at its process's
cadence; its prediction and advice are published on the unit it started on. PARs come
from the graph once per batch.
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel

from ai import store
from ai.offline import Collector, from_history
from ai.yield_ import optimize
from ai.yield_ import profiles as yp
from ai.yield_.features import FIRST_DAY, LAST_DAY, YieldInput, features_at
from ai.yield_.model import YieldModel
from common import models as m
from common import uns
from common.mqtt import UnsClient, connect
from common.plant import get_plant
from common.settings import Settings, get_settings
from common.uns import TopicClass, UnitPath
from graph import db as graph_db
from historian.migrate import connect as pg_connect
from simulator import recipes

SERVICE = "yield"
EVERY_DAYS = 0.25  # six simulated hours (the bioreactor)
log = logging.getLogger(SERVICE)

PARS = """
MATCH (:Batch {id: $batch})-[:FOLLOWS]->(:Recipe)-[:HAS_LIMIT]->(l:SpecLimit {type: 'PAR'})
RETURN l.parameter AS name, l.low AS low, l.high AS high
"""


@dataclass
class UnitState:
    """A bioreactor batch."""

    unit: UnitPath
    batch_id: str
    collector: Collector
    alerts: list[float] = field(default_factory=list)
    next_day: float = FIRST_DAY
    par: dict[str, tuple[float, float]] | None = None
    has_recommendation: bool = False


@dataclass
class TrainState:
    """A train batch, across its units; published on `unit`, where it started."""

    unit: UnitPath
    batch_id: str
    process: str
    collector: yp.TrainCollector
    recipe_id: str
    next_day: float | None = None
    par: dict[str, tuple[float, float]] | None = None
    has_recommendation: bool = False


def quantiles(values: np.ndarray) -> m.Quantiles:
    p10, p50, p90 = np.quantile(values, [0.1, 0.5, 0.9])
    return m.Quantiles(p10=float(p10), p50=float(p50), p90=float(p90))


class YieldService:
    def __init__(
        self, client: UnsClient, settings: Settings, models: dict[str, YieldModel]
    ) -> None:
        self.client, self.settings, self.models = client, settings, models
        self.model = models.get("bioreactor")
        self.units: dict[str, UnitState] = {}
        self.trains: dict[str, TrainState] = {}  # by batch
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
        if isinstance(payload, m.BatchEventPayload) and isinstance(payload.v, m.BatchStarted):
            ev = payload.v
            if ev.process != "bioreactor":
                if ev.process in self.models and ev.batch_id not in self.trains:
                    self.trains[ev.batch_id] = TrainState(
                        uns.parse(topic).unit, ev.batch_id, ev.process,
                        yp.TrainCollector(payload.ts), ev.recipe,
                    )  # fmt: skip
            elif self.model is not None:
                unit = uns.parse(topic).unit
                self.units[unit.cell] = UnitState(unit, ev.batch_id, Collector(payload.ts))
        batch = getattr(payload, "batch", None)
        if batch is None:
            return
        train = self.trains.get(batch)
        if train is not None or self._train_batch(batch, topic):
            self._handle_train(self.trains.get(batch) or self._recover_train(batch), topic, payload)
            return
        if self.model is None:
            return
        unit = uns.parse(topic).unit
        cell = unit.cell
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

    # -- bioreactor ------------------------------------------------------------------------------

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
        self._predict(state.unit, state.batch_id, self.model, yp.BIOREACTOR, x, day, ts)
        operation = col.ctx.operation_at(col.series.last_minute)
        if operation not in yp.BIOREACTOR.advise_ops:
            return
        advice = optimize.recommend(
            self.model, x, col.levers, day, operation, self._par(state, col.recipe_id),
            seed=int(day * 100),
        )  # fmt: skip
        state.has_recommendation = self._advise(
            state.unit, state.batch_id, self.model, yp.BIOREACTOR, advice, day, ts,
            state.has_recommendation,
        )  # fmt: skip

    # -- trains ----------------------------------------------------------------------------------

    def _train_batch(self, batch_id: str, topic: str) -> bool:
        """A batch this service has not seen start: is it a train batch (a restart)?"""
        try:
            p = uns.parse(topic)
        except uns.TopicError:
            return False
        if p.unit is None:
            return False
        process = get_plant().unit(p.unit.cell).process
        return process != "bioreactor" and process in self.models

    def _recover_train(self, batch_id: str) -> TrainState | None:
        try:
            with pg_connect(self.settings.postgres_dsn, wait_s=5) as conn:
                col = yp.train_from_history(conn, batch_id)
                row = conn.execute(
                    "SELECT topic, payload -> 'v' ->> 'recipe' FROM uns_events "
                    "WHERE batch_id = %s AND payload -> 'v' ->> 'kind' = 'BATCH_START'",
                    (batch_id,),
                ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        unit = uns.parse(row[0]).unit
        process = get_plant().unit(unit.cell).process
        state = TrainState(unit, batch_id, process, col, row[1])
        state.next_day = self._first_due(state, col.inp.last_minute)
        self.trains[batch_id] = state
        log.info("%s: recovered train batch %s from history", unit.cell, batch_id)
        return state

    def _first_due(self, state: TrainState, after_minute: float = 0.0) -> float | None:
        first, _ = yp.TRAIN_WINDOW[state.process]
        start, _ = state.collector.inp.bounds(first)
        if start is None:
            return None
        step = yp.PROFILES[state.process].every_days * 1440.0
        t = start + step * max(0, np.ceil((after_minute - start) / step))
        return t / 1440.0

    def _handle_train(self, state: TrainState | None, topic: str, payload: BaseModel) -> None:
        if state is None:
            return
        col = state.collector
        if isinstance(payload, m.AlertPayload):
            if payload.v.state == "OPEN" and not payload.v.key.startswith("rules-quality"):
                col.inp.alert_minutes.append(col.minute(payload.v.opened_at))
            return
        col.message(topic, payload)
        if isinstance(payload, m.BatchEventPayload) and isinstance(payload.v, m.BatchEnded):
            if state.has_recommendation:
                self.client.clear_retained(uns.ai_recommendation(state.unit))
            del self.trains[state.batch_id]
            return
        if state.next_day is None:
            state.next_day = self._first_due(state)
        self._maybe_predict_train(state)

    def _maybe_predict_train(self, state: TrainState) -> None:
        inp, process = state.collector.inp, state.process
        profile, model = yp.PROFILES[process], self.models[process]
        _, last = yp.TRAIN_WINDOW[process]
        while state.next_day is not None and inp.levers is not None:
            t = state.next_day * 1440.0
            _, end = inp.bounds(last)
            if inp.last_minute < t or (end is not None and t >= end):
                return
            day = state.next_day
            state.next_day += profile.every_days
            x = yp.TRAIN_FEATURES[process](inp, day)
            ts = state.collector.origin + timedelta(minutes=t)
            self._predict(state.unit, state.batch_id, model, profile, x, day, ts)
            at = yp.at_for(process, inp, day)
            if at.operation not in profile.advise_ops:
                continue
            advice = optimize.recommend(
                model, x, inp.levers, day, at.operation, self._par(state, state.recipe_id),
                seed=int(day * 1000), at=at,
            )  # fmt: skip
            state.has_recommendation = self._advise(
                state.unit, state.batch_id, model, profile, advice, day, ts,
                state.has_recommendation,
            )  # fmt: skip

    # -- publishing ------------------------------------------------------------------------------

    def _predict(
        self,
        unit: UnitPath,
        batch_id: str,
        model: YieldModel,
        profile: yp.YieldProfile,
        x: np.ndarray,
        day: float,
        ts: datetime,
    ) -> None:
        band = model.band(x)
        self.client.publish(
            uns.ai_prediction(unit),
            m.PredictionPayload(
                v=m.Prediction(
                    target=profile.target,
                    value=m.Quantiles(
                        p10=float(band[0.1][0]), p50=float(band[0.5][0]), p90=float(band[0.9][0])
                    ),
                    batch_day=day,
                    model_version=model.version,
                ),
                ts=ts,
                unit=profile.unit,
                batch=batch_id,
                src=m.Src.YIELD,
            ),
        )

    def _advise(
        self,
        unit: UnitPath,
        batch_id: str,
        model: YieldModel,
        profile: yp.YieldProfile,
        advice: optimize.Advice,
        day: float,
        ts: datetime,
        had: bool,
    ) -> bool:
        """Publish or clear the retained recommendation. Returns whether one is open."""
        topic = uns.ai_recommendation(unit)
        if not advice.passes_gate:
            if had:
                self.client.clear_retained(topic)
            return False
        rec = m.Recommendation(
            id=f"R-{batch_id}-d{day:.3f}",
            target=profile.target,
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
            model_version=model.version,
        )
        self.client.publish(
            topic,
            m.RecommendationPayload(
                v=rec, ts=ts, unit=profile.unit, batch=batch_id, src=m.Src.YIELD
            ),
        )
        log.info("%s day %.3f: recommend %s (gain P50 %+.2f %s)", batch_id, day, rec.id,
                 rec.gain.p50, profile.unit)  # fmt: skip
        return True

    def _par(self, state: UnitState | TrainState, recipe_id: str) -> dict[str, tuple[float, float]]:
        """PARs from the graph once per batch (ADR-0015), configuration as fallback."""
        if state.par is None:
            par = {}
            if self.driver is not None:
                try:
                    recs, _, _ = self.driver.execute_query(PARS, batch=state.batch_id)
                    par = {r["name"]: (float(r["low"]), float(r["high"])) for r in recs}
                except Exception:
                    par = {}
            state.par = par or recipes.get(recipe_id).par
        return state.par


def load_models(models_dir: str) -> dict[str, YieldModel]:
    out = {}
    for process in yp.PROFILES:
        model = store.load(models_dir, store.yield_name(process))
        if model is not None:
            out[process] = model
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    models = load_models(settings.models_dir)
    if not models:
        raise SystemExit(f"no yield model in {settings.models_dir}; run python -m ai.train_all")
    client = connect(SERVICE, m.Src.YIELD, settings)
    service = YieldService(client, settings, models)
    stop = threading.Event()
    worker = threading.Thread(target=service.work, args=(stop,), name="predict")
    worker.start()
    service.subscribe()
    log.info("predicting with %s", ", ".join(f"{p}: {mo.version}" for p, mo in models.items()))
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    client.close()
    worker.join(timeout=5)


if __name__ == "__main__":
    main()
