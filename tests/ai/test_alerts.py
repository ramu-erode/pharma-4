"""Alert lifecycle (ADR-0013): open after 2 windows over, clear after 4 under."""

from ai.anomaly.detector import AlertManager, Evidence, suggest
from ai.features import FEATURES
from common.models import FaultType

EV = Evidence("stats", 5.0, 3.0, ["co2_flow"], None, 5 / 3)


def feed(am: AlertManager, pattern: str):
    out = []
    for i, ch in enumerate(pattern):
        out += am.update(i * 5, {"stats-co2_flow": EV} if ch == "x" else {})
    return out


def test_one_window_is_not_enough():
    assert feed(AlertManager("B2026-0200"), "x.x.x.") == []


def test_open_then_clear_once():
    changes = feed(AlertManager("B2026-0200"), "xxxx...x....")
    assert [c.state for c in changes] == ["OPEN", "CLEARED"]
    assert changes[0].alert_id == changes[1].alert_id == "B2026-0200-A001"
    assert changes[0].end == 5 and changes[1].opened_end == 5


def test_a_short_dip_does_not_clear():
    changes = feed(AlertManager("B2026-0200"), "xx...xx")
    assert [c.state for c in changes] == ["OPEN"]


def test_new_lifecycle_gets_a_new_id():
    changes = feed(AlertManager("B2026-0200"), "xx....xx")
    assert [c.alert_id for c in changes] == ["B2026-0200-A001"] * 2 + ["B2026-0200-A002"]


def test_fault_class_suggestion():
    z = dict.fromkeys(FEATURES, 0.0)
    assert suggest({**z, "co2_mean": 4.0}, "co2_mean") is FaultType.PH_PROBE_DRIFT
    assert (
        suggest({**z, "agitation_mean": 4.0, "do_err_mean": -1.0}, "x")
        is FaultType.DO_SPARGER_FOULING
    )
    assert suggest({**z, "hours_since_bolus": 6.0}, "x") is FaultType.FEED_PUMP_FAILURE
    assert suggest(z, "x") is None
