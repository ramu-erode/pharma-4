"""A small hand-built plant for i3X and assistant tests: one bioreactor with two batches,
and one API reactor whose batch made a lot the bioreactor's second batch drew on (the
genealogy shape, not a real route)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from common import models as m
from common import uns
from i3x import values
from i3x.api import Backend, create_app
from i3x.space import AddressSpace, Catalog, build
from i3x.subscriptions import Hub

UNIT = uns.UnitPath("grange-castle", "upstream", "suite-1", "BR-101")
RX = uns.UnitPath("tuas", "api", "train-1", "RX-201")
API_LEVERS = {"rxn_temp": 85.0, "ac2o_ratio": 1.25, "rxn_time": 3.0, "cool_rate": 10.0,
              "dry_temp": 50.0}  # fmt: skip
PV_PH = uns.pv(UNIT, "ph")
SP_PH = uns.sp(UNIT, "ph")
PV_TEMP = uns.pv(UNIT, "temperature")
T0 = datetime(2026, 3, 1, tzinfo=UTC)
KEY = "test-key"
LEVERS = {"shift_day": 5.0, "prod_temp": 33.0, "ph_sp": 7.0, "do_sp": 40.0, "feed_mult": 1.0}


def catalog() -> Catalog:
    op = [
        {
            "name": "Growth",
            "start": T0 + timedelta(hours=7),
            "end": T0 + timedelta(days=5),
            "phases": [
                {
                    "phase": "PH_CTRL",
                    "state": "COMPLETE",
                    "start": T0 + timedelta(hours=7),
                    "end": T0 + timedelta(days=5),
                    "holds": 1,
                }
            ],
        },
        {"name": "Setup", "start": T0, "end": T0 + timedelta(hours=6), "phases": []},
    ]
    return Catalog(
        enterprise={"id": "pharmanextgen", "name": "PharmaNextGen"},
        sites=[
            {
                "id": "grange-castle",
                "name": "Grange Castle",
                "location": "Dublin, Ireland",
                "role": "Biologics drug substance",
            },
            {"id": "tuas", "name": "Tuas", "location": "Singapore", "role": "Small-molecule API"},
        ],
        lines=[
            {
                "site": "grange-castle",
                "area": {"id": "upstream", "name": "Upstream processing"},
                "line": {"id": "suite-1", "name": "Suite 1", "process": "bioreactor"},
            },
            {
                "site": "tuas",
                "area": {"id": "api", "name": "API manufacturing"},
                "line": {"id": "train-1", "name": "Train 1", "process": "api"},
            },
        ],
        units=[
            {
                "id": "BR-101",
                "line": "suite-1",
                "type": "Bioreactor",
                "class": "bioreactor",
                "process": "bioreactor",
                "position": 0,
                "working_volume_l": 2000,
            },
            {
                "id": "RX-201",
                "line": "train-1",
                "type": "ReactorCrystallizer",
                "class": "reactor",
                "process": "api",
                "position": 0,
                "working_volume_l": 4000,
            },
        ],
        modules=[
            {
                "id": "BR-101/AIC-102",
                "unit": "BR-101",
                "module": "AIC-102",
                "name": None,
                "parent": "BR-101/EM-PH",
            },
            {
                "id": "BR-101/EM-PH",
                "unit": "BR-101",
                "module": "EM-PH",
                "name": "pH control",
                "parent": None,
            },
            {
                "id": "BR-101/EM-THERMAL",
                "unit": "BR-101",
                "module": "EM-THERMAL",
                "name": "Jacket temperature",
                "parent": None,
            },
            {
                "id": "BR-101/TIC-101",
                "unit": "BR-101",
                "module": "TIC-101",
                "name": None,
                "parent": "BR-101/EM-THERMAL",
            },
        ],
        tags=[
            {
                "topic": PV_PH,
                "cm": "BR-101/AIC-102",
                "cell": "BR-101",
                "unit": "pH",
                "kind": "PV",
                "raw_tag": "BR101.AIC-102.PV",
            },
            {
                "topic": SP_PH,
                "cm": "BR-101/AIC-102",
                "cell": "BR-101",
                "unit": "pH",
                "kind": "SP",
                "raw_tag": "BR101.AIC-102.SP",
            },
            {
                "topic": PV_TEMP,
                "cm": "BR-101/TIC-101",
                "cell": "BR-101",
                "unit": "°C",
                "kind": "PV",
                "raw_tag": "BR101.TIC-101.PV",
            },
        ],
        sensors=[
            {
                "id": "BR-101/PH-PROBE",
                "unit": "BR-101",
                "model": "InPro 3253i",
                "calibration_interval_days": 30,
                "measures": [PV_PH],
            },
        ],
        bindings=[
            {"phase": "PH_CTRL", "module": "BR-101/EM-PH", "role": "control"},
            {"phase": "PH_CTRL", "module": "BR-101/TIC-101", "role": "monitor"},
            {"phase": "TEMP_CTRL", "module": "BR-101/EM-THERMAL", "role": "control"},
            {"phase": "DOSE_ADD", "module": "RX-201/EM-DOSING", "role": "control"},
        ],
        recipes=[
            {
                "id": "v3",
                "name": "CHO-mAb fed-batch",
                "version": 3,
                "effective_from": "2025-01-01",
                "nominal": LEVERS,
                "limits": [
                    {
                        "parameter": "ph",
                        "type": "action",
                        "low": -0.1,
                        "high": 0.1,
                        "spRelative": True,
                    },
                    {
                        "parameter": "shift_day",
                        "type": "PAR",
                        "low": 4.0,
                        "high": 7.0,
                        "spRelative": None,
                    },
                ],
            },
        ],
        batches=[
            {
                "id": "B2026-0142",
                "cell": "BR-101",
                "recipe": "v3",
                "campaign": "MFG",
                "status": "COMPLETE",
                "start": T0,
                "end": T0 + timedelta(days=14),
                "end_reason": None,
                "planned": LEVERS,
                "actual": {**LEVERS, "do_sp": 35.0},
                "outcome": {
                    "titer": 3.1,
                    "peak_vcd": 20.5,
                    "viability": 71.0,
                    "harvest_day": 14.0,
                    "disposition": "ACCEPTED",
                },
                "operations": op,
            },
            {
                "id": "B2026-0143",
                "cell": "BR-101",
                "recipe": "v3",
                "campaign": "MFG",
                "status": "RUNNING",
                "start": T0 + timedelta(days=15),
                "end": None,
                "end_reason": None,
                "planned": LEVERS,
                "actual": LEVERS,
                "outcome": None,
                "operations": [],
                "consumed": [{"lot": "B2026-0140", "kg": 200.0}],
            },
            {
                "id": "B2026-0140",
                "process": "api",
                "cell": "RX-201",
                "cells": ["RX-201"],
                "recipe": "asa-v2",
                "campaign": "MFG",
                "status": "COMPLETE",
                "start": T0 - timedelta(days=3),
                "end": T0 - timedelta(days=2),
                "end_reason": None,
                "planned": API_LEVERS,
                "actual": API_LEVERS,
                "outcome": {"yield_pct": 87.2, "free_sa": 0.05, "disposition": "ACCEPTED"},
                "operations": [],
                "produced": "B2026-0140",
                "consumed": [],
            },
        ],
        lots=[
            {
                "id": "B2026-0140",
                "material": "Acetylsalicylic acid API",
                "quantity_kg": 571.0,
                "produced_by": "B2026-0140",
                "consumed_by": [{"batch": "B2026-0143", "kg": 200.0}],
            }
        ],
        alerts=[
            {
                "id": "B2026-0142-A001",
                "batch": "B2026-0142",
                "key": "stats-co2_flow",
                "layer": "stats",
                "state": "CLEARED",
                "score": 2.3,
                "threshold": 1.0,
                "fault_class": "ph_probe_drift",
                "top_tags": ["ph"],
                "opened_at": T0 + timedelta(days=3),
                "cleared_at": T0 + timedelta(days=4),
                "tags": [PV_PH],
            },
        ],
        actions=[
            {
                "id": "B2026-0142/operator/x/do_sp",
                "batch": "B2026-0142",
                "ts": T0 + timedelta(days=4),
                "operator": "demo",
                "parameter": "do_sp",
                "old": 40.0,
                "new": 35.0,
                "reason": "advice",
                "tags": [uns.sp(UNIT, "do")],
                "recommendation": "R-B2026-0142-d4.00",
            },
        ],
        recommendations=[
            {
                "id": "R-B2026-0142-d4.00",
                "batch": "B2026-0142",
                "ts": T0 + timedelta(days=4),
                "model_version": "yield-test",
                "gain_p10": 0.05,
                "gain_p50": 0.2,
                "gain_p90": 0.4,
                "levers": '{"do_sp": {"current": 40.0, "recommended": 35.0, "frozen": false}}',
            },
        ],
    )


def space() -> AddressSpace:
    return build(catalog())


def scalar(topic: str, v: float, ts: datetime, q: m.Quality = m.Quality.GOOD) -> m.ScalarPayload:
    return m.ScalarPayload(v=v, ts=ts, unit="pH", q=q, batch="B2026-0143", src=m.Src.EDGE)


class FakeHistory:
    def __init__(self) -> None:
        self.numeric_rows: dict[str, list[tuple[datetime, float, str]]] = {}
        self.structured_rows: dict[str, list[tuple[datetime, dict[str, Any]]]] = {}

    def numeric(self, topic, start, end, limit):
        return [r for r in self.numeric_rows.get(topic, []) if start <= r[0] <= end][:limit]

    def structured(self, topic, start, end, limit):
        return [r for r in self.structured_rows.get(topic, []) if start <= r[0] <= end][:limit]


def backend(**kwargs: Any) -> Backend:
    s = space()
    return Backend(
        space=lambda: s,
        cache=values.LiveCache(),
        history=FakeHistory(),
        hub=Hub(),
        api_key=KEY,
        **kwargs,
    )


def app(b: Backend | None = None):
    return create_app(b or backend())
