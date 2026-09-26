"""Every API and OSD fault type is caught by its equipment class's anomaly model
(CLAUDE.md, ADR-0021), and clean batches stay quiet (fewer than 1 false alert per clean
batch)."""

from __future__ import annotations

import pytest

from ai.anomaly import train
from common.models import FaultType
from tests.ai.harness_trains import CASES, STUCK, STUCK_KEY, breach_minute, harness, onset_minute

pytestmark = pytest.mark.slow


def opened(result: train.BatchResult, after: float = float("-inf")):
    return [c for c in result.changes if c.state == "OPEN" and c.end >= after]


@pytest.mark.parametrize("kind", list(CASES), ids=lambda k: k.value)
def test_fault_is_caught_in_time(kind: FaultType):
    h = harness()
    case, collected, batch = h.faults[kind.value]
    result = train.run(h.models[case.cls], batch)
    onset = onset_minute(collected)
    alerts = opened(result, after=onset)
    assert alerts, f"no alert for {kind.value}"
    first = alerts[0].end
    if case.criterion == "before_breach":
        breach = breach_minute(kind, collected)
        assert breach is not None, "the fault never breached; the case is not testing anything"
        assert first < breach, (
            f"alert at +{(first - onset) / 60:.2f} h, breach +{(breach - onset) / 60:.2f} h"
        )
    else:
        assert first - onset <= case.criterion * 60, f"alert +{(first - onset) / 60:.2f} h"


@pytest.mark.parametrize("process", list(STUCK))
def test_stuck_sensor_is_named_by_the_rules_layer(process: str):
    h = harness()
    case, collected, batch = h.faults[f"stuck_{process}"]
    onset = onset_minute(collected)
    alerts = opened(train.run(h.models[case.cls], batch), onset)
    assert STUCK_KEY[process] in {c.key for c in alerts}
    assert min(c.end for c in alerts) - onset <= case.criterion * 60


def test_alerts_suggest_the_right_fault_class():
    h = harness()
    for kind, case in CASES.items():
        _, collected, batch = h.faults[kind.value]
        alerts = opened(train.run(h.models[case.cls], batch), onset_minute(collected))
        classes = {c.evidence.fault_class for c in alerts}
        assert kind in classes, f"{kind.value}: suggested {classes}"


@pytest.mark.parametrize("cls", ["reactor", "filter_dryer", "roller_compactor", "tablet_press"])
def test_clean_batches_stay_quiet(cls: str):
    h = harness()
    counts = [len(opened(train.run(h.models[cls], b))) for b in h.holdout[cls]]
    assert sum(counts) / len(counts) < 1.0, counts


def test_no_alert_before_onset_on_faulty_batches():
    h = harness()
    early = 0
    for case, collected, batch in h.faults.values():
        onset = onset_minute(collected)
        early += len([c for c in opened(train.run(h.models[case.cls], batch)) if c.end < onset])
    assert early <= 2
