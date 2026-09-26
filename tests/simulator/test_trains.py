"""The API and OSD trains (ADR-0018, ADR-0019): sequencing across units, genealogy events,
plausible CoAs, interior lever optima, fault signatures, and FIFO lot allocation."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest

from common.models import (
    ApiLevers,
    BatchEnded,
    BatchStarted,
    Campaign,
    MaterialConsumed,
    MaterialProduced,
    Operation,
    OperationChanged,
    OsdLevers,
)
from common.models import FaultType as F
from simulator import recipes
from simulator import train_campaign as tc
from simulator.api.engine import ApiRun
from simulator.messages import EventOut, LabOut, RawOut, StateOut, TruthOut
from simulator.osd.engine import OsdRun
from simulator.train import LotUse, TrainFaultSpec, TrainSpec

START = datetime(2026, 9, 24, tzinfo=UTC)
API_NOMINAL = recipes.get("asa-v2").nominal
OSD_NOMINAL = recipes.get("tab-v2").nominal


def api(levers: ApiLevers = API_NOMINAL, faults=(), seed: int = 3, **kw) -> ApiRun:
    spec = TrainSpec("B2026-0600", "RX-201", START, "asa-v2", Campaign.MFG, levers, seed,
                     faults=tuple(faults))  # fmt: skip
    return ApiRun(spec, **kw)


def osd(levers: OsdLevers = OSD_NOMINAL, faults=(), seed: int = 3, lots=(), **kw) -> OsdRun:
    spec = TrainSpec("B2026-0601", "BL-301", START, "tab-v2", Campaign.MFG, levers, seed,
                     faults=tuple(faults), lots=tuple(lots))  # fmt: skip
    return OsdRun(spec, **kw)


def api_yield(**overrides: float) -> float:
    run = api(API_NOMINAL.model_copy(update=overrides))
    run.run_to_end()
    return run.target()


def osd_yield(**overrides: float) -> float:
    run = osd(OSD_NOMINAL.model_copy(update=overrides))
    run.run_to_end()
    return run.target()


def operations(msgs) -> list[tuple[str, str]]:
    return [
        (m.cell, m.v.current.value)
        for m in msgs
        if isinstance(m, EventOut) and isinstance(m.v, OperationChanged)
        and m.v.current is not Operation.IDLE
    ]  # fmt: skip


# --- sequencing and messages -------------------------------------------------------------------


def test_api_batch_moves_from_reactor_to_filter_dryer():
    run = api()
    msgs = run.run_to_end()
    assert operations(msgs) == [
        ("RX-201", "Charge"), ("RX-201", "Reaction"), ("RX-201", "Crystallization"),
        ("RX-201", "Transfer"), ("FD-202", "Filtration"), ("FD-202", "Washing"),
        ("FD-202", "Drying"), ("FD-202", "Discharge"),
    ]  # fmt: skip
    starts = [m for m in msgs if isinstance(m, EventOut) and isinstance(m.v, BatchStarted)]
    ends = [m for m in msgs if isinstance(m, EventOut) and isinstance(m.v, BatchEnded)]
    assert [(m.cell, m.v.process) for m in starts] == [("RX-201", "api")]
    assert [m.cell for m in ends] == ["FD-202"] and ends[0].v.disposition == "ACCEPTED"
    held = [(m.cell, m.value) for m in msgs if isinstance(m, StateOut) and m.name == "batch"]
    assert held == [("RX-201", run.spec.batch_id), ("RX-201", None),
                    ("FD-202", run.spec.batch_id), ("FD-202", None)]  # fmt: skip
    assert 20 < run.t_h < 40


def test_every_train_unit_publishes_raw_tags_on_every_tick():
    run = api()
    msgs = run.advance(3.0)
    cells = {m.cell for m in msgs if isinstance(m, RawOut)}
    assert cells == {"RX-201", "FD-202"}  # the dryer reads idle values meanwhile


def test_api_coa_is_plausible_and_the_lot_goes_into_stock():
    run = api()
    msgs = run.run_to_end()
    lab = {m.name: m.value for m in msgs if isinstance(m, LabOut)}
    assert 98.5 < lab["conversion_ipc"] <= 100
    assert 84 < lab["yield"] < 92 and lab["free_sa"] < 0.1 and lab["lod"] < 0.5
    assert 150 < lab["d50"] < 300 and lab["assay"] > 99.5
    made = [m.v for m in msgs if isinstance(m, EventOut) and isinstance(m.v, MaterialProduced)]
    assert made[0].lot == run.spec.batch_id and 540 < made[0].quantity_kg < 600


def test_tablet_batch_consumes_its_api_lots_and_runs_the_line():
    lots = (
        LotUse("B2026-0401", "Acetylsalicylic acid API", 120.0, {"d50_um": 220.0}),
        LotUse("B2026-0405", "Acetylsalicylic acid API", 80.0, {"d50_um": 260.0}),
    )
    run = osd(lots=lots)
    msgs = run.run_to_end()
    assert operations(msgs) == [
        ("BL-301", "Charge"), ("BL-301", "Blending"), ("BL-301", "Lubrication"),
        ("BL-301", "Discharge"), ("RC-302", "Compaction"), ("TP-303", "Compression"),
    ]  # fmt: skip
    used = [m for m in msgs if isinstance(m, EventOut) and isinstance(m.v, MaterialConsumed)]
    assert [(m.cell, m.v.lot, m.v.quantity_kg) for m in used] == [
        ("BL-301", "B2026-0401", 120.0), ("BL-301", "B2026-0405", 80.0)
    ]  # fmt: skip
    lab = {m.name: m.value for m in msgs if isinstance(m, LabOut)}
    assert lab["dissolution"] >= 80 and lab["av"] <= 15 and lab["friability"] <= 1.0
    assert 94 < lab["yield"] < 99
    assert run.api_d50 == pytest.approx(236.0)


def test_coarse_api_slows_dissolution():
    fine = osd(lots=(LotUse("B2026-0401", "API", 200.0, {"d50_um": 180.0}),))
    coarse = osd(lots=(LotUse("B2026-0402", "API", 200.0, {"d50_um": 380.0}),))
    fine.run_to_end()
    coarse.run_to_end()
    assert coarse.coa["dissolution"] < fine.coa["dissolution"] - 5


# --- levers ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lever", "low", "best", "high"),
    [
        ("rxn_temp", 80.0, 83.0, 90.0),
        ("ac2o_ratio", 1.10, 1.25, 1.40),
        ("cool_rate", 5.0, 10.0, 20.0),
        ("dry_temp", 40.0, 48.0, 60.0),
    ],
)
def test_api_levers_have_interior_optima(lever, low, best, high):
    y = api_yield(**{lever: best})
    assert y > api_yield(**{lever: low}) and y > api_yield(**{lever: high})


@pytest.mark.parametrize(
    ("lever", "low", "best", "high"),
    [
        ("lube_time", 2.0, 4.0, 6.0),
        ("roll_force", 4.0, 8.5, 10.0),
        ("comp_force", 10.0, 13.0, 20.0),
        ("turret_speed", 30.0, 40.0, 60.0),
        ("feed_frame", 20.0, 32.0, 40.0),
    ],
)
def test_osd_levers_have_interior_optima(lever, low, best, high):
    y = osd_yield(**{lever: best})
    assert y > osd_yield(**{lever: low}) and y >= osd_yield(**{lever: high}) - 0.02


def test_the_old_recipes_leave_room_for_the_optimizer():
    assert api_yield() > api_yield(**recipes.get("asa-v1").nominal.model_dump()) + 0.5
    assert osd_yield() > osd_yield(**recipes.get("tab-v1").nominal.model_dump()) + 0.5


def test_levers_freeze_once_their_window_has_passed():
    run = api()
    run.advance(3.0)  # in Reaction
    assert run.operation is Operation.REACTION
    with pytest.raises(ValueError, match="no longer"):
        run.change_lever("ac2o_ratio", 1.3, "demo")
    msgs = run.change_lever("dry_temp", 48.0, "demo")
    assert msgs[0].cell == "FD-202" and run.levers.dry_temp == 48.0


# --- faults -------------------------------------------------------------------------------------


def truth(run) -> list[TruthOut]:
    return [m for m in run.run_to_end() if isinstance(m, TruthOut)]


def test_jacket_fouling_leaves_the_reactor_behind_its_ramp():
    run = api(faults=(TrainFaultSpec(F.JACKET_FOULING, Operation.CRYSTALLIZATION, 1.0),),
              record_truth=True)  # fmt: skip
    worst = max(abs(t.state["rx"].temp - t.sp["temperature"]) for t in truth(run)
                if t.operation is Operation.CRYSTALLIZATION)  # fmt: skip
    assert worst > 5.0


def test_dosing_meter_drift_shows_a_perfect_total_and_leaves_sa_behind():
    run = api(faults=(TrainFaultSpec(F.DOSING_METER_DRIFT, Operation.REACTION, 0.1),))
    run.run_to_end()
    assert run.rx.ac2o_metered_kg >= run.dose_target_kg  # the PV reached target
    assert run.ratio_dosed < API_NOMINAL.ac2o_ratio - 0.1  # but less went in
    assert run.disposition == "REJECTED" and run.coa["free_sa"] > 0.1


def test_filter_blinding_slows_filtration():
    clean, blinded = api(), api(faults=(TrainFaultSpec(F.FILTER_BLINDING, Operation.FILTRATION,
                                                       0.3),))  # fmt: skip
    for r in (clean, blinded):
        r.run_to_end()
    hours = [r.op_ended_h[Operation.FILTRATION] - r.op_started[Operation.FILTRATION]
             for r in (clean, blinded)]  # fmt: skip
    assert hours[1] > hours[0] + 1.0


def test_vacuum_leak_stalls_drying_and_hydrolyses_product():
    run = api(faults=(TrainFaultSpec(F.VACUUM_LEAK, Operation.DRYING, 1.0),), record_truth=True)
    peak = max(t.state["fd"].vacuum for t in truth(run) if t.operation is Operation.DRYING
               and t.t_h > run.op_started[Operation.DRYING] + 0.5)  # fmt: skip
    assert peak > 150 and run.coa["free_sa"] > 0.1


def test_roll_force_drift_keeps_the_pv_on_setpoint_while_ribbons_soften():
    run = osd(faults=(TrainFaultSpec(F.ROLL_FORCE_DRIFT, Operation.COMPACTION, 0.5),),
              record_truth=True)  # fmt: skip
    msgs = run.run_to_end()
    pv = [m.value for m in msgs if isinstance(m, RawOut) and m.tag == "RC302.PIC-311.PV"
          and m.cell == "RC-302" and m.value > 1]  # fmt: skip
    assert abs(sum(pv) / len(pv) - OSD_NOMINAL.roll_force) < 0.1
    true_force = [t.state["rc"].force for t in msgs if isinstance(t, TruthOut)
                  and t.operation is Operation.COMPACTION]  # fmt: skip
    assert min(true_force) < OSD_NOMINAL.roll_force - 1.0


def test_punch_sticking_raises_ejection_force():
    run = osd(faults=(TrainFaultSpec(F.PUNCH_STICKING, Operation.COMPRESSION, 0.5),))
    run.run_to_end()
    assert run.tp.ejection > 500


def test_hvac_failure_raises_room_humidity_and_free_sa():
    run = osd(faults=(TrainFaultSpec(F.HVAC_HUMIDITY, Operation.COMPACTION, 0.5),))
    run.run_to_end()
    assert run.rh > 55 and run.disposition == "REJECTED"


def test_stuck_sensor_repeats_bit_identical_values():
    run = osd(faults=(TrainFaultSpec(F.STUCK_SENSOR, Operation.COMPRESSION, 1.0,
                                     {"tag": "ejection_force", "duration_h": 1.0}),))  # fmt: skip
    msgs = run.run_to_end()
    ej = [m.value for m in msgs if isinstance(m, RawOut) and m.tag == "TP303.PI-325.PV"]
    repeats = max(sum(1 for _ in g) for _, g in itertools.groupby(ej))
    assert repeats >= 50


# --- planning and genealogy -------------------------------------------------------------------


def test_planner_mixes_recipes_campaigns_and_faults():
    end = datetime(2026, 9, 26, tzinfo=UTC)
    for process in ("api", "osd"):
        specs = tc.plan(process, 140, end, 42)
        assert len(specs) == 140 and specs[-1].start < end - timedelta(days=1)
        assert {s.recipe_id for s in specs} == {r.id for r in recipes.of_process(process)}
        assert sum(s.campaign is Campaign.PC for s in specs) == 42
        assert 15 <= sum(1 for s in specs if s.faults) <= 25
        assert tc.plan(process, 140, end, 42) == specs


def test_allocation_is_fifo_and_keeps_freiburg_stocked():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    lots = [
        tc.Lot(f"B2026-{i:04d}", "API", 570.0, t0 + timedelta(days=2.5 * i), {"d50_um": 200.0})
        for i in range(12)
    ]
    tablets = [
        TrainSpec(f"B2026-{100 + i:04d}", "BL-301", t0 + timedelta(days=3 + 1.5 * i), "tab-v2",
                  Campaign.MFG, OSD_NOMINAL, i)
        for i in range(12)
    ]  # fmt: skip
    specs, left = tc.allocate(tablets, lots)
    drawn = [u.lot for s in specs for u in s.lots]
    assert all(sum(u.quantity_kg for u in s.lots) == pytest.approx(200.0) for s in specs)
    assert drawn == sorted(drawn)  # first in, first out
    assert sum(lot.quantity_kg for lot in left) < tc.FREIBURG_REORDER_KG + 570.0
