"""The live runner's pieces, without a broker: clock, wire mapping, command handling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from common import models as m
from common import uns
from common.models import FaultType, Operation, Src
from common.settings import Settings
from simulator import batches as b
from simulator.clock import SimClock
from simulator.engine import BatchRun
from simulator.run import Retained, Simulator
from simulator.wire import to_wire
from tests.samples import UNIT
from tests.simulator.conftest import make_spec

T0 = datetime(2026, 9, 24, tzinfo=UTC)


# --- clock -------------------------------------------------------------------------------


def test_clock_start_is_monotonic():
    later = T0 + timedelta(days=30)
    assert SimClock.starting_at(T0, [later, T0 - timedelta(days=1)], 3600).now == later
    assert SimClock.starting_at(T0, [], 3600).now == T0


def test_clock_speed_and_pause():
    c = SimClock(now=T0, speed=3600)
    assert c.advance(1.0) == timedelta(hours=1)
    c.paused = True
    assert c.advance(1.0) == timedelta(0) and c.now == T0 + timedelta(hours=1)
    with pytest.raises(ValueError):
        c.set_speed(0)


def test_run_to_day_pauses_once():
    c = SimClock(now=T0)
    c.arm_run_to_day("BR-101", 4.0)
    assert not c.check_run_to_day("BR-101", 3.9)
    assert not c.check_run_to_day("BR-102", 4.5)
    assert c.check_run_to_day("BR-101", 4.0) and c.paused
    assert not c.check_run_to_day("BR-101", 5.0)


# --- wire --------------------------------------------------------------------------------


def test_every_engine_message_maps_to_its_topic_model():
    run = BatchRun(make_spec(faults=(b.FaultSpec(FaultType.STUCK_SENSOR, 0.1, {"tag": "do"}),)))
    msgs = run.advance(b.t_of_day(0.5))
    msgs += run.change_lever("ph_sp", 7.02, operator="demo")
    seen_topics = set()
    for msg in msgs:
        wire = to_wire(UNIT, run.spec.batch_id, msg)
        assert wire is not None
        topic, payload = wire
        assert isinstance(payload, m.model_for(topic))
        seen_topics.add(uns.parse(topic).kind)
    assert {uns.TopicKind.EDGE_RAW, uns.TopicKind.UNS, uns.TopicKind.SIM_FAULTS} <= seen_topics


# --- command handling --------------------------------------------------------------------


class FakeClient:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    def publish(self, topic, payload):
        assert isinstance(payload, m.model_for(topic))
        self.published.append((topic, payload))

    def subscribe(self, pattern, handler, qos=1):
        pass


def settings() -> Settings:
    return Settings(service="simulator", backfill_batches=200, sim_speed=3600)


def cmd(v):
    kind = {
        m.BatchCommand: uns.SimCommand.BATCH,
        m.FaultCommand: uns.SimCommand.FAULT,
        m.ClockCommand: uns.SimCommand.CLOCK,
        m.SetpointCommand: uns.SimCommand.SETPOINT,
    }[type(v)]
    return m.SIM_COMMAND_MODELS[kind](v=v, ts=T0, unit=None, batch=None, src=Src.OPERATOR)


def values(client: FakeClient, suffix: str) -> list:
    return [p.v for t, p in client.published if t.endswith(suffix)]


def test_fresh_start_publishes_idle_state_and_first_live_batch_follows_all_history():
    client = FakeClient()
    sim = Simulator(client, settings(), Retained(latest_ts=[T0]))
    for cell in ("BR-101", "RX-201", "FD-202", "BL-301", "RC-302", "TP-303"):
        assert values(client, f"{cell}/state/operation") == [Operation.IDLE]
    sim.commands.put(cmd(m.BatchCommand(action="start", cell="BR-101")))
    sim._drain_commands()
    # 200 bioreactor + 140 API + 140 tablet batches of history, numbered across sites
    assert sim.units["BR-101"].run.spec.batch_id == "B2026-0480"
    assert values(client, "BR-101/state/batch")[-1] == "B2026-0480"


def test_restart_continues_the_sequence_and_closes_orphans():
    client = FakeClient()
    retained = Retained(
        latest_ts=[T0],
        last_batch_seq=507,
        batch_of={"BR-101": "B2026-0507"},
        operation_of={"BR-101": Operation.PRODUCTION},
        phases_of={"BR-101": {"temp_ctrl": m.PhaseState.RUNNING}},
    )
    sim = Simulator(client, settings(), retained)
    assert sim.next_seq == 508
    ended = [v for v in values(client, "BR-101/events/batch") if isinstance(v, m.BatchEnded)]
    assert ended[0].batch_id == "B2026-0507" and ended[0].status is m.BatchStatus.ABORTED
    assert values(client, "BR-101/state/phase/temp_ctrl") == [m.PhaseState.COMPLETE]
    assert values(client, "BR-101/state/batch") == [None]


def test_run_to_day_stops_the_clock_on_the_day():
    client = FakeClient()
    sim = Simulator(client, settings(), Retained(latest_ts=[T0]))
    for c in (
        m.BatchCommand(action="start", cell="BR-101"),
        m.ClockCommand(action="run_to_day", cell="BR-101", day=1.0),
    ):
        sim.commands.put(cmd(c))
    sim._drain_commands()
    for _ in range(60):  # 60 ticks of 1 s at 3600x: 60 simulated hours offered
        sim._tick(1.0)
    run = sim.units["BR-101"].run
    assert sim.clock.paused
    assert run.batch_day == pytest.approx(1.0, abs=0.01)
    # the idle unit kept publishing raw values up to the same instant
    idle_raw = [p for t, p in client.published if t.startswith("edge/raw/BR102/")]
    assert idle_raw and max(p.t for p in idle_raw) <= sim.clock.now


def test_bad_commands_are_rejected_not_fatal(caplog):
    sim = Simulator(FakeClient(), settings(), Retained(latest_ts=[T0]))
    sim.commands.put(cmd(m.FaultCommand(action="inject", cell="BR-101", fault="stuck_sensor")))
    sim._drain_commands()
    assert "no running batch" in caplog.text


def test_setpoint_command_emits_operator_event():
    client = FakeClient()
    sim = Simulator(client, settings(), Retained(latest_ts=[T0]))
    sim.commands.put(cmd(m.BatchCommand(action="start", cell="BR-102")))
    sim.commands.put(cmd(m.SetpointCommand(
        cell="BR-102", parameter="prod_temp", value=33.5, operator="demo",
        recommendation_id="R-9")))  # fmt: skip
    sim._drain_commands()
    ev = values(client, "BR-102/events/operator")
    assert ev[0].recommendation_id == "R-9" and ev[0].new == 33.5


# --- trains and the Freiburg stock (ADR-0018, ADR-0020) -----------------------------------


def stock(kg: float = 700.0) -> m.Inventory:
    return m.Inventory(
        site="freiburg",
        lots=[
            m.LotStock(
                lot="B2026-0401", material="Acetylsalicylic acid API", quantity_kg=150.0,
                released=T0 - timedelta(days=30), properties={"d50_um": 250.0},
            ),
            m.LotStock(
                lot="B2026-0405", material="Acetylsalicylic acid API", quantity_kg=kg,
                released=T0 - timedelta(days=20), properties={"d50_um": 210.0},
            ),
        ],
    )  # fmt: skip


def test_a_train_batch_holds_every_unit_of_its_train():
    sim = Simulator(FakeClient(), settings(), Retained(latest_ts=[T0]))
    sim.commands.put(cmd(m.BatchCommand(action="start", cell="RX-201")))
    sim._drain_commands()
    run = sim.units["RX-201"].run
    assert run is not None and sim.units["FD-202"].run is run
    assert sim.units["BL-301"].run is None


def test_a_train_batch_starts_only_on_its_first_unit(caplog):
    sim = Simulator(FakeClient(), settings(), Retained(latest_ts=[T0]))
    sim.commands.put(cmd(m.BatchCommand(action="start", cell="FD-202")))
    sim._drain_commands()
    assert "starts on RX-201" in caplog.text and sim.units["FD-202"].run is None


def test_tablets_need_released_api(caplog):
    sim = Simulator(FakeClient(), settings(), Retained(latest_ts=[T0]))
    sim.commands.put(cmd(m.BatchCommand(action="start", cell="BL-301")))
    sim._drain_commands()
    assert "released API" in caplog.text and sim.units["BL-301"].run is None


def test_opening_stock_is_adopted_and_drawn_first_in_first_out():
    client = FakeClient()
    sim = Simulator(client, settings(), Retained(latest_ts=[T0], opening_stock=stock()))
    sim.commands.put(cmd(m.BatchCommand(action="start", cell="BL-301")))
    sim._drain_commands()
    lots = sim.units["BL-301"].run.spec.lots
    assert [(u.lot, u.quantity_kg) for u in lots] == [("B2026-0401", 150.0), ("B2026-0405", 50.0)]
    left = values(client, "_sim/inventory")[-1].lots
    assert [(lot.lot, lot.quantity_kg) for lot in left] == [("B2026-0405", 650.0)]
    consumed = values(client, "BL-301/events/material")
    assert {c.lot for c in consumed} == {"B2026-0401", "B2026-0405"}


def test_own_inventory_wins_over_the_opening_stock():
    mine = stock(kg=300.0)
    sim = Simulator(
        FakeClient(), settings(), Retained(latest_ts=[T0], inventory=mine, opening_stock=stock())
    )
    assert sim.inventory == mine


def test_a_released_api_lot_goes_into_freiburg_stock():
    client = FakeClient()
    sim = Simulator(client, settings(), Retained(latest_ts=[T0], opening_stock=stock()))
    sim.commands.put(cmd(m.BatchCommand(action="start", cell="RX-201")))
    sim._drain_commands()
    batch = sim.units["RX-201"].run.spec.batch_id
    for _ in range(80):  # 80 simulated hours offered: an API batch takes about 30
        sim._tick(1.0)
    assert sim.units["RX-201"].run is None and sim.units["FD-202"].run is None
    produced = values(client, "FD-202/events/material")
    assert produced and produced[0].lot == batch
    assert batch in {lot.lot for lot in sim.inventory.lots}
