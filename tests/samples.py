"""One valid topic + payload for every topic class. Shared by model and ACL tests."""

from __future__ import annotations

from datetime import UTC, datetime

from common import models as m
from common import uns
from common.uns import SimCommand, TopicClass

UNIT = uns.UnitPath("chennai", "upstream", "suite-1", "BR-101")
TS = datetime(2026, 9, 22, 10, 15, 5, 123000, tzinfo=UTC)
BATCH = "B2026-0142"
LEVERS = m.Levers(shift_day=5.0, prod_temp=33.0, ph_sp=7.0, do_sp=40.0, feed_mult=1.0)
Q = m.Quantiles(p10=3.1, p50=3.4, p90=3.7)


def _p(model: type, v: object, unit: str | None = None, src: m.Src = m.Src.SIM, batch=BATCH):
    return model(v=v, ts=TS, unit=unit, batch=batch, src=src)


def samples() -> list[tuple[str, object]]:
    """(topic, payload) pairs covering every branch of models.model_for."""
    return [
        (uns.pv(UNIT, "ph"), _p(m.ScalarPayload, 7.03, "pH", m.Src.EDGE)),
        (uns.sp(UNIT, "temperature"), _p(m.ScalarPayload, 36.5, "°C", m.Src.EDGE)),
        (uns.lab(UNIT, "titer"), _p(m.ScalarPayload, 3.2, "g/L")),
        (uns.state_batch(UNIT), _p(m.BatchStatePayload, BATCH)),
        (uns.state_batch(UNIT), _p(m.BatchStatePayload, None, batch=None)),
        (uns.state_operation(UNIT), _p(m.OperationPayload, m.Operation.GROWTH)),
        (uns.state_phase(UNIT, "PH_CTRL"), _p(m.PhaseStatePayload, m.PhaseState.HELD)),
        (
            uns.events(UNIT, "batch"),
            _p(
                m.BatchEventPayload,
                m.BatchStarted(batch_id=BATCH, recipe="v3", campaign="MFG", planned_levers=LEVERS),
            ),
        ),
        (
            uns.events(UNIT, "batch"),
            _p(m.BatchEventPayload, m.BatchEnded(batch_id=BATCH, status="EARLY_HARVEST")),
        ),
        (
            uns.events(UNIT, "operator"),
            _p(
                m.OperatorEventPayload,
                m.OperatorEvent(operator="demo", parameter="temperature", old=36.5, new=33.0),
                src=m.Src.OPERATOR,
            ),
        ),
        (uns.ai_score(UNIT), _p(m.ScalarPayload, 1.7, None, m.Src.ANOMALY)),
        (
            uns.ai_alert(UNIT, "stats-co2_flow"),
            _p(
                m.AlertPayload,
                m.Alert(
                    id="A-1",
                    key="stats-co2_flow",
                    state="OPEN",
                    layer="stats",
                    score=4.2,
                    threshold=3.0,
                    fault_class="ph_probe_drift",
                    top_tags=["co2_flow", "base_total", "ph"],
                    opened_at=TS,
                ),
                src=m.Src.ANOMALY,
            ),
        ),
        (
            uns.ai_prediction(UNIT),
            _p(
                m.PredictionPayload,
                m.Prediction(titer=Q, batch_day=4.0, model_version="y-1"),
                "g/L",
                m.Src.YIELD,
            ),
        ),
        (
            uns.ai_recommendation(UNIT),
            _p(
                m.RecommendationPayload,
                m.Recommendation(
                    id="R-1",
                    levers={"shift_day": m.LeverAdvice(current=6.0, recommended=5.0, frozen=False)},
                    predicted_current=Q,
                    predicted_recommended=Q,
                    gain=m.Quantiles(p10=0.05, p50=0.2, p90=0.35),
                    model_version="y-1",
                ),
                "g/L",
                m.Src.YIELD,
            ),
        ),
        (
            uns.meta_tag("BR-101", TopicClass.PV, "ph"),
            _p(
                m.TagMetaPayload,
                m.TagMeta(
                    raw_tag="BR101.AIC-102.PV",
                    topic=uns.pv(UNIT, "ph"),
                    unit="pH",
                    kind="PV",
                    deadband=0.01,
                ),
                src=m.Src.EDGE,
                batch=None,
            ),
        ),
        (
            uns.meta_status("historian"),
            _p(
                m.HeartbeatPayload,
                m.Heartbeat(service="historian", state="online", wall=TS),
                src=m.Src.HISTORIAN,
                batch=None,
            ),
        ),
        (
            uns.edge_raw("BR101", "BR101.AIC-102.PV"),
            m.RawSample(tag="BR101.AIC-102.PV", value=7.03, t=TS),
        ),
        (uns.edge_unmapped(), m.RawSample(tag="BR101.TI-199.PV", value=21.4, t=TS)),
        (
            uns.sim_cmd(SimCommand.BATCH),
            _p(
                m.Payload[m.BatchCommand],
                m.BatchCommand(action="start", cell="BR-101"),
                src=m.Src.OPERATOR,
                batch=None,
            ),
        ),
        (
            uns.sim_cmd(SimCommand.FAULT),
            _p(
                m.Payload[m.FaultCommand],
                m.FaultCommand(action="inject", cell="BR-101", fault="ph_probe_drift"),
                src=m.Src.OPERATOR,
                batch=None,
            ),
        ),
        (
            uns.sim_cmd(SimCommand.CLOCK),
            _p(
                m.Payload[m.ClockCommand],
                m.ClockCommand(action="speed", speed=600),
                src=m.Src.OPERATOR,
                batch=None,
            ),
        ),
        (
            uns.sim_cmd(SimCommand.SETPOINT),
            _p(
                m.Payload[m.SetpointCommand],
                m.SetpointCommand(
                    cell="BR-101",
                    parameter="temperature",
                    value=33.0,
                    operator="demo",
                    recommendation_id="R-1",
                ),
                src=m.Src.OPERATOR,
                batch=None,
            ),
        ),
        (
            uns.sim_clock(),
            _p(
                m.ClockStatusPayload,
                m.ClockStatus(sim_time=TS, speed=3600, paused=False),
                batch=None,
            ),
        ),
        (
            uns.sim_faults("BR-101"),
            _p(
                m.FaultLabelPayload,
                m.FaultLabel(
                    id="F-1",
                    fault="stuck_sensor",
                    cell="BR-101",
                    batch_id=BATCH,
                    onset=TS,
                    params={"tag": "do"},
                ),
            ),
        ),
    ]
