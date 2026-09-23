"""ADR-0006 attribution, without Neo4j: segmenting and row building."""

from datetime import UTC, datetime, timedelta

from graph.project import OPEN_END, rows_from_records, segments

T0 = datetime(2026, 9, 24, tzinfo=UTC)


def h(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


def test_segments_without_holds():
    assert segments(h(0), h(10), []) == [("RUNNING", h(0), h(10))]


def test_segments_split_around_holds_and_clip():
    out = segments(h(0), h(10), [(h(6), h(7)), (h(2), h(3)), (h(9), h(12))])
    assert out == [
        ("RUNNING", h(0), h(2)),
        ("HELD", h(2), h(3)),
        ("RUNNING", h(3), h(6)),
        ("HELD", h(6), h(7)),
        ("RUNNING", h(7), h(9)),
        ("HELD", h(9), h(10)),
    ]


def test_open_hold_runs_to_the_end():
    assert segments(h(0), h(5), [(h(4), None)]) == [
        ("RUNNING", h(0), h(4)),
        ("HELD", h(4), h(5)),
    ]


END = h(24)


def rec(topic, op, phase=None, role=None, holds=(), op_end=END):
    return {
        "topic": topic,
        "operation": op,
        "op_start": h(0),
        "op_end": op_end,
        "phase": phase,
        "role": role,
        "pi_start": h(0) if phase else None,
        "pi_end": op_end if phase else None,
        "holds": [list(x) for x in holds],
    }


def test_overlapping_phases_keep_both_roles():
    """pv/ph in Growth: PH_CTRL controls it and FEED_ADD monitors it, at the same time."""
    rows = rows_from_records(
        "B2026-0200",
        [
            rec("x/pv/ph", "Growth", "PH_CTRL", "control"),
            rec("x/pv/ph", "Growth", "FEED_ADD", "monitor", holds=[(h(5), h(6))]),
        ],
    )
    got = {(r.phase, r.role, r.phase_state, r.t_start, r.t_end) for r in rows}
    assert got == {
        ("PH_CTRL", "control", "RUNNING", h(0), h(24)),
        ("FEED_ADD", "monitor", "RUNNING", h(0), h(5)),
        ("FEED_ADD", "monitor", "HELD", h(5), h(6)),
        ("FEED_ADD", "monitor", "RUNNING", h(6), h(24)),
    }


def test_no_bound_phase_falls_back_to_the_operation():
    rows = rows_from_records("B2026-0200", [rec("x/pv/weight", "Setup")])
    assert len(rows) == 1
    r = rows[0]
    assert (r.operation, r.phase, r.role, r.phase_state) == ("Setup", None, None, None)


def test_open_operation_gets_an_open_end():
    rows = rows_from_records(
        "B2026-0200", [rec("x/pv/ph", "Growth", "PH_CTRL", "control", op_end=None)]
    )
    assert rows[0].t_end == OPEN_END
