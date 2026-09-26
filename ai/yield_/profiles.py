"""Yield profiles: what each process predicts and advises on (ADR-0021).

A profile holds the target (titer in g/L for the bioreactor, yield in % for the API and
tablet processes), the levers and how much of a lever's whole-batch effect a change can
still have at a given point (`exposure`), the gate's minimum change and gain, when
predictions run, and the mid-batch features.

The bioreactor keeps its original functions (`ai.yield_.features`, `rsm.exposure`,
`optimize.open_levers`). A train batch moves between units (ADR-0018), so its input is
collected per batch across the units: `TrainCollector` follows it through the train,
keyed `<equipment class>.<name>`, with time in minutes since the batch started.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import psycopg
from pydantic import BaseModel

from ai.yield_ import features as bio
from common import models as m
from common import uns
from common.levers import actual_levers
from common.models import Operation
from common.plant import get_plant
from common.uns import TopicClass


@dataclass(frozen=True, slots=True)
class At:
    """Where a batch is when advice is asked for."""

    day: float  # days since the alignment reference (inoculation, or batch start)
    operation: str | None
    frac: float = 0.0  # how far through that operation (trains; estimated)


@dataclass(frozen=True)
class YieldProfile:
    process: str
    target: str  # the lab result that reports it
    unit: str
    levers: tuple[str, ...]
    features: tuple[str, ...]
    measured: str | None  # feature holding what is already measured of the target
    min_gain: float
    min_change: dict[str, float]
    exposure: Callable[[str, At, BaseModel], float]
    every_days: float  # prediction cadence
    advise_ops: tuple[str, ...]  # operations in which a recommendation is worth making
    surface_transform: str = "identity"  # "loss": fit the DoE surface to ln(100 - yield)

    def open_levers(self, at: At, levers: BaseModel) -> set[str]:
        return {k for k in self.levers if self.exposure(k, at, levers) > 0.0}


# --- bioreactor --------------------------------------------------------------------------------


def _bio_exposure(lever: str, at: At, levers: BaseModel) -> float:
    from ai.yield_.optimize import open_levers
    from ai.yield_.rsm import exposure

    if lever not in open_levers(at.day, at.operation, levers):
        return 0.0
    return exposure(lever, at.day, levers.shift_day)


BIOREACTOR = YieldProfile(
    process="bioreactor",
    target="titer",
    unit="g/L",
    levers=bio.LEVERS,
    features=bio.FEATURES,
    measured="titer_so_far",
    min_gain=0.1,
    min_change={"shift_day": 0.1, "prod_temp": 0.1, "ph_sp": 0.01, "do_sp": 1.0, "feed_mult": 0.02},
    exposure=_bio_exposure,
    every_days=0.25,
    advise_ops=("Growth", "TempShift", "Production"),
)


# --- trains: the input -------------------------------------------------------------------------


@dataclass
class TrainInput:
    """A train batch so far, across its units (minutes since batch start)."""

    values: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    lab: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    ops: list[list] = field(default_factory=list)  # [name, start, end or None]
    levers: BaseModel | None = None
    alert_minutes: list[float] = field(default_factory=list)
    last_minute: float = 0.0

    def last(self, key: str, minute: float) -> float:
        vals = [v for t, v in self.values.get(key, []) if t <= minute]
        return vals[-1] if vals else math.nan

    def mean(self, key: str, t0: float | None, t1: float) -> float:
        if t0 is None:
            return math.nan
        vals = [v for t, v in self.values.get(key, []) if t0 <= t <= t1]
        return float(np.mean(vals)) if vals else math.nan

    def bounds(self, name: str) -> tuple[float | None, float | None]:
        for op, start, end in self.ops:
            if op == name:
                return start, end
        return None, None

    def operation_at(self, minute: float) -> str | None:
        for op, start, end in reversed(self.ops):
            if start <= minute and (end is None or minute < end):
                return op
        return None

    def hours_in(self, name: str, minute: float) -> float:
        start, end = self.bounds(name)
        if start is None or start > minute:
            return 0.0
        return (min(end if end is not None else minute, minute) - start) / 60.0

    def lab_at(self, name: str, minute: float) -> float:
        vals = [v for t, v in self.lab.get(name, []) if t <= minute]
        return vals[-1] if vals else math.nan


class TrainCollector:
    """Accumulates the UNS messages of one train batch, from every unit it runs on."""

    def __init__(self, origin: datetime) -> None:
        self.origin = origin
        self.inp = TrainInput()
        self.planned: BaseModel | None = None
        self.operator_events: list[m.OperatorEvent] = []
        self.recipe_id: str | None = None
        self.status: m.BatchStatus | None = None
        self.classes = {c: u.cls for c, u in get_plant().units.items()}

    def minute(self, ts: datetime) -> float:
        return (ts - self.origin).total_seconds() / 60.0

    def value(self, topic: str, ts: datetime, value: float) -> None:
        p = uns.parse(topic)
        t = self.minute(ts)
        self.inp.last_minute = max(self.inp.last_minute, t)
        if p.cls is TopicClass.LAB:
            self.inp.lab.setdefault(p.name, []).append((t, value))
        elif p.cls in (TopicClass.PV, TopicClass.SP):
            prefix = "sp_" if p.cls is TopicClass.SP else ""
            key = f"{self.classes[p.unit.cell]}.{prefix}{p.name}"
            self.inp.values.setdefault(key, []).append((t, value))

    def message(self, topic: str, payload: BaseModel) -> None:
        if isinstance(payload, m.ScalarPayload):
            self.value(topic, payload.ts, payload.v)
        elif isinstance(payload, m.BatchEventPayload):
            ev, t = payload.v, self.minute(payload.ts)
            match ev:
                case m.BatchStarted():
                    self.recipe_id = ev.recipe
                    self.planned = ev.planned_levers
                case m.OperationChanged():
                    if ev.previous is not Operation.IDLE:
                        for op in self.inp.ops:
                            if op[0] == ev.previous.value and op[2] is None:
                                op[2] = t
                    if ev.current is not Operation.IDLE:
                        self.inp.ops.append([ev.current.value, t, None])
                case m.BatchEnded():
                    self.status = ev.status
        elif isinstance(payload, m.OperatorEventPayload):
            self.operator_events.append(payload.v)
        self.inp.levers = (
            actual_levers(self.planned, self.operator_events) if self.planned else None
        )


def train_from_history(conn: psycopg.Connection, batch_id: str) -> TrainCollector:
    events = conn.execute(
        "SELECT ts, topic, payload FROM uns_events WHERE batch_id = %s ORDER BY ts, seq",
        (batch_id,),
    ).fetchall()
    start = next(
        ts
        for ts, _, p in events
        if isinstance(p["v"], dict) and p["v"].get("kind") == "BATCH_START"
    )
    col = TrainCollector(start)
    for _, topic, p in events:
        payload = m.decode(topic, json.dumps(p).encode())
        if payload is not None:
            col.message(topic, payload)
    for ts, topic, value in conn.execute(
        "SELECT ts, topic, value FROM tag_values WHERE batch_id = %s "
        "AND (topic LIKE %s OR topic LIKE %s OR topic LIKE %s) ORDER BY ts",
        (batch_id, "%/pv/%", "%/sp/%", "%/lab/%"),
    ):
        col.value(topic, ts, value)
    return col


# --- API (Tuas) ----------------------------------------------------------------------------------

API_OPS = (
    "Charge",
    "Reaction",
    "Crystallization",
    "Transfer",
    "Filtration",
    "Washing",
    "Drying",
    "Discharge",
)
API_LEVERS = ("rxn_temp", "ac2o_ratio", "rxn_time", "cool_rate", "dry_temp")
API_FEATURES = (
    "hours",
    "stage",
    *API_LEVERS,
    "conversion",
    "conversion_ipc",
    "rxn_temp_err",
    "dose_total",
    "chord",
    "cryst_hours",
    "filtration_hours",
    "filtrate_total",
    "moisture",
    "vacuum_mean",
    "drying_hours",
    "alerts_so_far",
)


def api_features(inp: TrainInput, day: float) -> np.ndarray:
    t = day * 1440.0
    op = inp.operation_at(t)
    lv = inp.levers
    rxn0, rxn1 = inp.bounds("Reaction")
    dry0, _ = inp.bounds("Drying")
    return np.array(
        [
            t / 60.0,
            float(API_OPS.index(op)) if op in API_OPS else math.nan,
            *(getattr(lv, k) for k in API_LEVERS),
            inp.last("reactor.conversion", t),
            inp.lab_at("conversion_ipc", t),
            inp.mean("reactor.temperature", rxn0, min(t, rxn1 or t))
            - inp.mean("reactor.sp_temperature", rxn0, min(t, rxn1 or t)),
            inp.last("reactor.dose_total", t),
            inp.last("reactor.chord_length", t),
            inp.hours_in("Crystallization", t),
            inp.hours_in("Filtration", t),
            inp.last("filter_dryer.filtrate_total", t),
            inp.last("filter_dryer.moisture", t) if dry0 is not None and dry0 <= t else math.nan,
            inp.mean("filter_dryer.vacuum", dry0, t)
            if dry0 is not None and dry0 <= t
            else math.nan,
            inp.hours_in("Drying", t),
            float(sum(1 for a in inp.alert_minutes if a <= t)),
        ]
    )


def _api_frac(op: str | None, inp_levers: BaseModel, hours_in: float) -> float:
    lv = inp_levers
    duration = {
        "Reaction": 1.0 + lv.rxn_time,
        "Crystallization": (lv.rxn_temp - 5.0) / lv.cool_rate + 1.0,
        "Drying": 10.0,
    }.get(op or "", 1.0)
    return float(np.clip(hours_in / duration, 0.0, 1.0))


def _api_exposure(lever: str, at: At, levers: BaseModel) -> float:
    op = at.operation
    i = API_OPS.index(op) if op in API_OPS else -1
    window = {
        "ac2o_ratio": "Charge",
        "rxn_temp": "Reaction",
        "rxn_time": "Reaction",
        "cool_rate": "Crystallization",
        "dry_temp": "Drying",
    }[lever]
    w = API_OPS.index(window)
    if lever == "ac2o_ratio":
        return 1.0 if i <= w else 0.0  # the charge is dosed at the start of Reaction
    if lever == "rxn_time":
        return 1.0 if i <= w else 0.0  # the hold is a duration still ahead
    if i < w:
        return 1.0
    return 1.0 - at.frac if i == w else 0.0


API = YieldProfile(
    process="api",
    target="yield",
    unit="%",
    levers=API_LEVERS,
    features=API_FEATURES,
    measured=None,
    min_gain=0.5,
    min_change={
        "rxn_temp": 0.5,
        "ac2o_ratio": 0.01,
        "rxn_time": 0.1,
        "cool_rate": 0.5,
        "dry_temp": 1.0,
    },
    exposure=_api_exposure,
    every_days=1.0 / 24.0,
    advise_ops=(
        "Charge",
        "Reaction",
        "Crystallization",
        "Transfer",
        "Filtration",
        "Washing",
        "Drying",
    ),
)


# --- OSD (Freiburg) ------------------------------------------------------------------------------

OSD_OPS = ("Charge", "Blending", "Lubrication", "Discharge", "Compaction", "Compression")
OSD_LEVERS = ("lube_time", "roll_force", "comp_force", "turret_speed", "feed_frame")
OSD_FEATURES = (
    "hours",
    "stage",
    *OSD_LEVERS,
    "blend_uniformity",
    "ribbon_density",
    "granule_d50",
    "bulk_density",
    "tablets",
    "reject_rate",
    "weight_rsd",
    "hardness",
    "ejection",
    "room_rh",
    "alerts_so_far",
)


def osd_features(inp: TrainInput, day: float) -> np.ndarray:
    t = day * 1440.0
    op = inp.operation_at(t)
    lv = inp.levers
    rc0, _ = inp.bounds("Compaction")
    tp0, _ = inp.bounds("Compression")
    in_tp = tp0 is not None and tp0 <= t
    tablets = inp.last("tablet_press.tablets_total", t) if in_tp else math.nan
    rejects = inp.last("tablet_press.rejects_total", t) if in_tp else math.nan
    rh = [v for k in ("blender", "roller_compactor", "tablet_press")
          for tt, v in inp.values.get(f"{k}.room_rh", []) if tt <= t]  # fmt: skip
    return np.array(
        [
            t / 60.0,
            float(OSD_OPS.index(op)) if op in OSD_OPS else math.nan,
            *(getattr(lv, k) for k in OSD_LEVERS),
            inp.lab_at("blend_uniformity", t),
            inp.mean("roller_compactor.ribbon_density", rc0, t) if rc0 is not None else math.nan,
            inp.lab_at("granule_d50", t),
            inp.lab_at("bulk_density", t),
            tablets,
            100.0 * rejects / tablets if in_tp and tablets > 1.0 else math.nan,
            inp.mean("tablet_press.weight_rsd", tp0, t) if in_tp else math.nan,
            inp.mean("tablet_press.hardness", tp0, t) if in_tp else math.nan,
            inp.mean("tablet_press.ejection_force", tp0, t) if in_tp else math.nan,
            float(np.mean(rh)) if rh else math.nan,
            float(sum(1 for a in inp.alert_minutes if a <= t)),
        ]
    )


def _osd_exposure(lever: str, at: At, levers: BaseModel) -> float:
    op = at.operation
    i = OSD_OPS.index(op) if op in OSD_OPS else -1
    if lever == "lube_time":
        return 1.0 if i <= OSD_OPS.index("Blending") else 0.0
    w = OSD_OPS.index("Compaction" if lever == "roll_force" else "Compression")
    if i < w:
        return 1.0
    return 1.0 - at.frac if i == w else 0.0


def _osd_frac(op: str | None, levers: BaseModel, hours_in: float) -> float:
    duration = {
        "Compaction": 4.0,
        "Compression": 396.0 / (36 * levers.turret_speed * 60 / 1000.0),
    }.get(op or "", 1.0)
    return float(np.clip(hours_in / duration, 0.0, 1.0))


OSD = YieldProfile(
    process="osd",
    target="yield",
    unit="%",
    levers=OSD_LEVERS,
    features=OSD_FEATURES,
    measured=None,
    min_gain=0.5,
    min_change={
        "lube_time": 0.2,
        "roll_force": 0.2,
        "comp_force": 0.3,
        "turret_speed": 1.0,
        "feed_frame": 1.0,
    },
    exposure=_osd_exposure,
    every_days=0.5 / 24.0,
    advise_ops=("Charge", "Blending", "Discharge", "Compaction", "Compression"),
    surface_transform="loss",  # tablet losses (rejects) compound multiplicatively
)

PROFILES: dict[str, YieldProfile] = {p.process: p for p in (BIOREACTOR, API, OSD)}
TRAIN_FEATURES: dict[str, Callable[[TrainInput, float], np.ndarray]] = {
    "api": api_features,
    "osd": osd_features,
}
FRAC: dict[str, Callable[[str | None, BaseModel, float], float]] = {
    "api": _api_frac,
    "osd": _osd_frac,
}
# Rows and predictions run from the start of the first of these operations to the end
# of the last.
TRAIN_WINDOW: dict[str, tuple[str, str]] = {
    "api": ("Reaction", "Drying"),
    "osd": ("Compaction", "Compression"),
}


def at_for(process: str, inp: TrainInput, day: float) -> At:
    t = day * 1440.0
    op = inp.operation_at(t)
    return At(day, op, FRAC[process](op, inp.levers, inp.hours_in(op, t) if op else 0.0))


def train_days(process: str, inp: TrainInput) -> list[float]:
    """The days (since batch start) at which rows are taken from a finished batch."""
    first, last = TRAIN_WINDOW[process]
    t0, _ = inp.bounds(first)
    _, t1 = inp.bounds(last)
    if t0 is None or t1 is None:
        return []
    step = PROFILES[process].every_days * 1440.0
    return [float(t / 1440.0) for t in np.arange(t0, t1 - 1e-6, step)]
