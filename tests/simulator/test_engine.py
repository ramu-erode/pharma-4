from __future__ import annotations

from datetime import timedelta
from itertools import pairwise

import pytest

from common.models import (
    BatchEnded,
    BatchStatus,
    FaultType,
    Operation,
    OperatorEvent,
    PhaseChanged,
    PhaseState,
    Quality,
)
from simulator import batches as b
from simulator.engine import BatchRun, EventOut, LabelOut, LabOut, RawOut, StateOut, TruthOut
from tests.simulator.conftest import fault_run, make_spec


def raw(msgs, suffix):
    return [m for m in msgs if isinstance(m, RawOut) and m.tag.endswith(suffix)]


def truth(msgs):
    return [m for m in msgs if isinstance(m, TruthOut)]


def test_deterministic_for_a_seed(spec):
    a = BatchRun(spec).advance(48.0)
    c = BatchRun(spec).advance(48.0)
    assert a == c
    d = BatchRun(make_spec(seed=8)).advance(48.0)
    assert [m.value for m in raw(a, "AIC-102.PV")] != [m.value for m in raw(d, "AIC-102.PV")]


def test_raw_samples_every_publish_period(spec):
    msgs = BatchRun(spec).advance(2.0)
    ph = raw(msgs, "AIC-102.PV")
    assert len(ph) == 120
    assert all(n.t - p.t == timedelta(seconds=60) for p, n in pairwise(ph))
    tags = {m.tag for m in msgs if isinstance(m, RawOut)}
    assert len(tags) == 15 and all(t.startswith("BR101.") for t in tags)


def test_isa88_sequence_and_end_state(spec):
    run = BatchRun(spec)
    msgs = run.run_to_end()
    ops = [m.value for m in msgs if isinstance(m, StateOut) and m.name == "operation"]
    assert ops == [
        o.value
        for o in (
            Operation.SETUP,
            Operation.INOCULATION,
            Operation.GROWTH,
            Operation.TEMP_SHIFT,
            Operation.PRODUCTION,
            Operation.HARVEST,
            Operation.IDLE,
        )
    ]
    states = [m for m in msgs if isinstance(m, StateOut)]
    assert states[0].name == "batch" and states[0].value == "B2026-0200"
    assert states[-1].name == "batch" and states[-1].value is None
    end = [m.v for m in msgs if isinstance(m, EventOut) and isinstance(m.v, BatchEnded)]
    assert end[0].status is BatchStatus.COMPLETE
    # every phase instance that started also completed, per operation
    changes = [m.v for m in msgs if isinstance(m, EventOut) and isinstance(m.v, PhaseChanged)]
    running = {(c.operation, c.phase) for c in changes if c.state is PhaseState.RUNNING}
    complete = {(c.operation, c.phase) for c in changes if c.state is PhaseState.COMPLETE}
    assert running == complete
    assert {p for o, p in running if o is Operation.GROWTH} == set(b.PHASES[Operation.GROWTH])
    assert run.batch_day == pytest.approx(b.HARVEST_DAY + b.HARVEST_H / 24, abs=0.01)


def test_hold_is_published_and_freezes_the_loop():
    hold = b.Hold(b.PhaseClass.PH_CTRL, b.t_of_day(2.0), b.t_of_day(2.0) + 1.0)
    run = BatchRun(make_spec(holds=(hold,)))
    msgs = run.advance(hold.end_h + 0.5)
    ph_states = [m.value for m in msgs if isinstance(m, StateOut) and m.name == "phase/ph_ctrl"]
    assert "HELD" in ph_states and ph_states[-1] == "RUNNING"
    co2 = [
        m.value
        for m in raw(msgs, "FIC-108.PV")
        if hold.start_h + 0.1 <= (m.t - run.spec.start).total_seconds() / 3600 < hold.end_h
    ]
    assert max(co2) - min(co2) < 0.02  # output frozen; only meter noise


def test_ph_probe_drift_closed_loop_signature():
    """Architecture/Q21: the PV sits on SP while the true pH falls, CO2 rises, base is flat."""
    run = fault_run(FaultType.PH_PROBE_DRIFT, onset_day=2.0)
    onset = b.t_of_day(2.0)
    before = run.advance(onset)
    after = run.advance(onset + 5.0)
    pv = [m.value for m in raw(after, "AIC-102.PV")]
    assert max(abs(v - 7.0) for v in pv) < 0.03
    true_ph = [m.state.ph for m in truth(after)]
    assert true_ph[0] - true_ph[-1] > 0.08
    co2_before = sum(m.value for m in raw(before, "FIC-108.PV")[-60:]) / 60
    co2_after = sum(m.value for m in raw(after, "FIC-108.PV")[-60:]) / 60
    # The loop must remove 0.02 pH/h more: +0.067 L/min CO2 at 0.3 pH/h per L/min.
    assert co2_after > co2_before + 0.05 and co2_after > 3 * co2_before
    base = [m.value for m in raw(after, "FQI-109.PV")]
    assert base[-1] - base[0] < 1.0


def test_ph_probe_drift_ends_at_offline_check():
    run = fault_run(FaultType.PH_PROBE_DRIFT, onset_day=2.0)
    msgs = run.advance(b.t_of_day(3.5))
    labels = [m.label for m in msgs if isinstance(m, LabelOut)]
    assert labels[-1].end is not None
    assert labels[-1].end == run.ts(b.t_of_day(3.0) + b.LAB_OFFSET_H)


def test_do_sparger_fouling_signature():
    run = fault_run(FaultType.DO_SPARGER_FOULING, onset_day=4.0)
    msgs = run.advance(b.t_of_day(5.5))
    agit = [m.value for m in raw(msgs, "SIC-104.PV")]
    o2 = [m.value for m in raw(msgs, "FIC-107.PV")]
    assert max(agit) > 139.0 and max(o2) > 0.95  # both actuators at their limits...
    assert min(m.state.do for m in truth(msgs)[-120:]) < 30.0  # ...and DO still sags


def test_temperature_control_loss_oscillates():
    run = fault_run(FaultType.TEMP_CONTROL_LOSS, onset_day=2.0)
    run.advance(b.t_of_day(2.0) + 4.0)
    msgs = run.advance(b.t_of_day(2.0) + 10.0)
    temps = [m.value for m in raw(msgs, "TIC-101.PV")]
    amplitude = (max(temps) - min(temps)) / 2
    assert 0.5 < amplitude < 1.2


def test_feed_pump_failure_stops_feed():
    run = fault_run(FaultType.FEED_PUMP_FAILURE, onset_day=4.5)
    msgs = run.advance(b.t_of_day(7.0))
    onset_t = run.ts(b.t_of_day(4.5))
    feed = [m.value for m in raw(msgs, "FQI-105.PV") if m.t >= onset_t]
    assert feed[-1] == feed[0]


def test_stuck_sensor_repeats_bit_identical_value():
    run = fault_run(FaultType.STUCK_SENSOR, onset_day=2.0, tag="pressure")
    before = raw(run.advance(b.t_of_day(2.0)), "PIC-110.PV")
    after = raw(run.advance(b.t_of_day(2.0) + 3.0), "PIC-110.PV")
    assert len({m.value for m in before}) == len(before)  # healthy: noise, never repeats
    assert len({m.value for m in after[1:]}) == 1
    assert {m.q for m in after[1:]} <= {Quality.GOOD, Quality.UNCERTAIN}


def test_contamination_aborts_with_signature():
    run = fault_run(FaultType.CONTAMINATION, onset_day=3.0)
    msgs = run.run_to_end()
    assert run.status is BatchStatus.ABORTED
    lactate = [m.value for m in msgs if isinstance(m, LabOut) and m.name == "lactate"]
    assert max(lactate) > 3.0  # the confirming sample shows the spike
    final = truth(msgs)[-1].state
    assert final.do < 25.0 and final.ph < 6.8  # DO and pH drop together


def test_fault_labels_mark_onset_and_end():
    run = fault_run(FaultType.STUCK_SENSOR, onset_day=2.0, tag="weight")
    msgs = run.run_to_end()
    labels = [m.label for m in msgs if isinstance(m, LabelOut)]
    assert len(labels) == 2
    assert labels[0].end is None and labels[0].onset == run.ts(b.t_of_day(2.0))
    assert labels[1].end - labels[1].onset == timedelta(hours=24)


def test_operator_lever_change():
    run = BatchRun(make_spec())
    run.advance(b.t_of_day(3.0))
    msgs = run.change_lever("shift_day", 4.5, operator="demo", recommendation_id="R-1")
    ev = msgs[0].v
    assert isinstance(ev, OperatorEvent) and (ev.old, ev.new) == (5.0, 4.5)
    run.advance(b.t_of_day(4.6))
    assert run.operation in (Operation.TEMP_SHIFT, Operation.PRODUCTION)
    with pytest.raises(ValueError):
        run.change_lever("shift_day", 6.0, operator="demo")
    run.change_lever("prod_temp", 33.5, operator="demo")  # still open
