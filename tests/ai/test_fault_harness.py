"""Every fault type in the simulator is caught by the anomaly layer (CLAUDE.md), and clean
batches stay quiet (architecture target: fewer than 1 false alert per clean batch)."""

from __future__ import annotations

import pytest

from ai.anomaly import train
from common.models import FaultType
from tests.ai.harness import CASES, breach_minute, harness, onset_minute, reference_minute

pytestmark = pytest.mark.slow


def opened(result: train.BatchResult, after: float = float("-inf")):
    return [c for c in result.changes if c.state == "OPEN" and c.end >= after]


@pytest.mark.parametrize("kind", list(CASES), ids=lambda k: k.value)
def test_fault_is_caught_in_time(kind: FaultType):
    h = harness()
    collected, batch = h.faults[kind]
    result = train.run(h.model, batch)
    onset = onset_minute(kind)
    alerts = opened(result, after=onset)
    assert alerts, f"no alert for {kind.value}"
    first = alerts[0].end
    criterion = CASES[kind][2]
    if criterion == "before_breach":
        breach = breach_minute(kind, collected)
        assert breach is not None, "the fault never breached; the case is not testing anything"
        assert first < breach, (
            f"alert at +{(first - onset) / 60:.1f} h, breach +{(breach - onset) / 60:.1f} h"
        )
    else:
        ref = reference_minute(kind)
        assert first - ref <= criterion * 60, f"alert {(first - ref) / 60:.1f} h after reference"


def test_stuck_sensor_is_named_by_the_rules_layer():
    h = harness()
    _, batch = h.faults[FaultType.STUCK_SENSOR]
    keys = {c.key for c in opened(train.run(h.model, batch), onset_minute(FaultType.STUCK_SENSOR))}
    assert "rules-stuck_pressure" in keys


def test_alerts_suggest_the_right_fault_class_for_developing_faults():
    h = harness()
    for kind in (
        FaultType.PH_PROBE_DRIFT,
        FaultType.TEMP_CONTROL_LOSS,
        FaultType.FEED_PUMP_FAILURE,
    ):
        _, batch = h.faults[kind]
        classes = {
            c.evidence.fault_class for c in opened(train.run(h.model, batch), onset_minute(kind))
        }
        assert kind in classes, f"{kind.value}: suggested {classes}"


def test_clean_batches_stay_quiet():
    h = harness()
    counts = [len(opened(train.run(h.model, b))) for b in h.holdout]
    assert sum(counts) / len(counts) < 1.0, counts


def test_no_alert_before_onset_on_faulty_batches():
    h = harness()
    early = sum(
        len([c for c in opened(train.run(h.model, batch)) if c.end < onset_minute(kind)])
        for kind, (_, batch) in h.faults.items()
    )
    assert early <= 1
