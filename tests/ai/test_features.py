"""Train/serve parity and feature behaviour (plan: Increment 4 tests)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from ai.anomaly.windows import batch_windows, window_at
from ai.context import Context, controlled_by_config
from ai.features import FEATURES, STEP_MIN, Series, grid, last_bolus, stuck_signals
from ai.offline import Collector, from_engine
from common.settings import Settings
from edge.core import TagMap
from simulator.engine import BatchRun
from simulator.wire import to_wire
from tests.samples import UNIT
from tests.simulator.conftest import make_spec

T0 = datetime(2026, 9, 24, tzinfo=UTC)
S = Settings()
TM = TagMap.load()


def test_grid_holds_last_value_forward():
    s = Series(T0)
    s.add("ph", T0 + timedelta(minutes=2.5), 7.0)
    s.add("ph", T0 + timedelta(minutes=5.0), 7.1)
    g = grid(s, 0, 7)[:, 1]  # "ph" is the second signal
    assert np.isnan(g[0]) and np.isnan(g[1])  # minutes 0-1 and 1-2: nothing yet
    assert g[2] == 7.0 and g[3] == 7.0  # held forward
    assert g[4] == 7.1 and g[6] == 7.1  # published at the end of minute 4


def test_last_bolus_and_stuck():
    s = Series(T0)
    for m, v in [(10, 100.0), (60, 100.0), (61, 101.5), (62, 103.0), (90, 103.0)]:
        s.add("feed_total", T0 + timedelta(minutes=m), v)
    assert last_bolus(s, 70) == 62.0
    assert last_bolus(s, 61) is None
    for m in (10, 20, 30):
        s.add("pressure", T0 + timedelta(minutes=m), 70.123)
    assert stuck_signals(s, 31) == ["pressure"]


def test_live_windows_equal_offline_windows():
    """The live service scores each window once all data up to its end has arrived; that
    must give exactly the features training computed for the same window."""
    spec = make_spec()
    collected = from_engine(spec, S, TM)
    offline = {
        w.end: w for w in batch_windows(collected.series, collected.ctx, controlled_by_config())
    }

    # Replay the same batch as the live service receives it: message by message.
    from edge.core import Deadband, Mapped, map_and_enrich
    from simulator.engine import RawOut

    live = Collector(spec.start)
    deadband = Deadband(S.deadband_floor_s, S.publish_period_s)
    cache: dict[str, str | None] = {spec.cell: None}
    next_end = 30
    checked = 0
    for msg in BatchRun(spec).advance(24.0 * 3):
        if isinstance(msg, RawOut):
            m = map_and_enrich(msg.tag, msg.value, msg.t, msg.q, TM, cache)
            if not (isinstance(m, Mapped) and deadband.offer(m)):
                continue
            minute = (m.ts - spec.start).total_seconds() / 60
            while minute > next_end:  # every value up to next_end has arrived
                w = window_at(live.series, live.ctx, controlled_by_config(), next_end)
                if w is not None and next_end in offline:
                    np.testing.assert_array_equal(w.x, offline[next_end].x)
                    assert (w.operation, w.age_h) == (
                        offline[next_end].operation,
                        offline[next_end].age_h,
                    )
                    checked += 1
                next_end += STEP_MIN
            live.value(m.topic, m.ts, m.value, m.q.value)
        else:
            wire = to_wire(UNIT, spec.batch_id, msg)
            if wire:
                if wire[0].endswith("/state/batch"):
                    cache[spec.cell] = wire[1].v
                live.message(*wire)
    assert checked > 500


def test_feature_vector_shape_and_hold_context():
    ctx = Context()
    ctx.open_operation("Setup", 0)
    ctx.open_operation("Inoculation", 360)
    ctx.open_operation("Growth", 420)
    ctx.hold("PH_CTRL", 1000)
    ctx.release("PH_CTRL", 1060)
    assert ctx.operation_at(500) == "Growth" and ctx.age_h(420) == 1.0
    assert ctx.held_phases(1030) == {"PH_CTRL"} and ctx.held_phases(1070) == set()
    controlled = controlled_by_config()
    assert {"ph", "co2_flow", "base_total"} <= controlled["PH_CTRL"]
    assert "ph" not in controlled["FEED_ADD"]  # FEED_ADD only monitors pH
    assert len(FEATURES) == 19
