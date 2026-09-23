from datetime import UTC, datetime, timedelta

import pytest

from common.models import Quality
from edge.core import Deadband, Mapped, TagMap, Unmapped, map_and_enrich
from simulator.engine import BatchRun, RawOut
from tests.simulator.conftest import make_spec

TAG_MAP = TagMap.load("chennai", "upstream", "suite-1")
T0 = datetime(2026, 9, 24, tzinfo=UTC)
BATCHES = {"BR-101": "B2026-0200", "BR-102": None}


def mapped(value: float, minutes: float = 0.0, q: Quality = Quality.GOOD, batch="B2026-0200"):
    out = map_and_enrich(
        "BR101.AIC-102.PV",
        value,
        T0 + timedelta(minutes=minutes),
        q,
        TAG_MAP,
        {"BR-101": batch},
    )
    assert isinstance(out, Mapped)
    return out


def test_enrichment_adds_unit_batch_and_keeps_value_and_time():
    v = 7.0123456789012345
    out = mapped(v)
    assert out.topic.endswith("BR-101/pv/ph")
    assert (out.unit, out.batch, out.q) == ("pH", "B2026-0200", Quality.GOOD)
    assert out.value == v  # never rounded
    assert out.ts == T0  # never restamped (ADR-0009)


def test_idle_unit_gets_no_batch():
    out = map_and_enrich("BR102.AIC-102.PV", 7.0, T0, Quality.GOOD, TAG_MAP, BATCHES)
    assert isinstance(out, Mapped) and out.batch is None


def test_unknown_tag_goes_to_unmapped():
    out = map_and_enrich("BR101.TI-199.PV", 21.9, T0, Quality.GOOD, TAG_MAP, BATCHES)
    assert isinstance(out, Unmapped) and out.raw_tag == "BR101.TI-199.PV"


def test_deadband_suppresses_small_changes():
    db = Deadband(floor_s=600)
    assert db.offer(mapped(7.000, 0))
    assert not db.offer(mapped(7.005, 1))  # within 0.01
    assert db.offer(mapped(7.015, 2))  # beyond, relative to last *published*
    assert not db.offer(mapped(7.010, 3))


def test_deadband_floor_republishes_in_source_time():
    db = Deadband(floor_s=600)
    assert db.offer(mapped(7.000, 0))
    assert not db.offer(mapped(7.001, 9))
    assert db.offer(mapped(7.001, 10))  # 10 simulated minutes


def test_quality_or_batch_change_always_publishes():
    db = Deadband(floor_s=600)
    assert db.offer(mapped(7.0, 0))
    assert db.offer(mapped(7.0, 1, q=Quality.UNCERTAIN))
    assert db.offer(mapped(7.0, 2, q=Quality.UNCERTAIN, batch=None))


def test_floor_must_exceed_publish_period():
    with pytest.raises(ValueError, match="deadband floor"):
        Deadband(floor_s=60, publish_period_s=60)


def test_stuck_sensor_survives_the_deadband_as_identical_heartbeats():
    """Q8: after the deadband a stuck tag shows up only on floor publishes, each with a
    bit-identical value; a healthy noisy tag never repeats exactly."""
    db = Deadband(floor_s=600)
    stuck = [mapped(40.123456, m) for m in range(0, 61)]
    published = [m.value for m in stuck if db.offer(m)]
    assert len(published) == 7 and len(set(published)) == 1


def test_deadband_reduces_real_simulator_traffic():
    run = BatchRun(make_spec())
    raw = [m for m in run.advance(24.0 * 3) if isinstance(m, RawOut)]
    db = Deadband(floor_s=600, publish_period_s=60)
    out = [map_and_enrich(r.tag, r.value, r.t, r.q, TAG_MAP, BATCHES) for r in raw]
    kept = [m for m in out if isinstance(m, Mapped) and db.offer(m)]
    unmapped = [m for m in out if isinstance(m, Unmapped)]
    assert len(unmapped) == len(raw) // 15
    ratio = len(kept) / (len(raw) - len(unmapped))
    assert 0.1 < ratio < 0.7, ratio
