"""Anomaly profiles: what each equipment class is scored on (ADR-0021).

A profile holds everything the detector used to take from module constants: the
signals and 30-minute window features, which feature speaks for which tag, the scored
operations and the alignment axis, the settling rule, the rules layer's limit features
and rate-of-change check, and the fault-class heuristic.

The bioreactor profile is the original constants, unchanged. The API and OSD units align
on hours since their operation started: their operations are recipe steps, not a
culture's age. Every window that reaches back into the previous operation settles
(layers 2-3 and the limit rules wait for a full window of the new operation).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ai import features as bio
from ai.context import SHIFT_SETTLE_MIN, Context
from ai.features import WINDOW_MIN, Series, grid, slope
from common.models import FaultType

Compute = Callable[[Series, int, Context], np.ndarray | None]
SETTLE_EXTRA_MIN = 15  # after a full window, a start-up transient (pump-down, ramp) settles


@dataclass(frozen=True, slots=True)
class Roc:
    """Rate-of-change rule: |feature| over `limit` outside `exempt` operations."""

    feature: str
    tag: str
    limit: float
    exempt: tuple[str, ...]
    fault: FaultType | None


@dataclass(frozen=True)
class AnomalyProfile:
    cls: str
    pv: tuple[str, ...]
    sp: tuple[str, ...]  # "sp_<name>"
    analog: tuple[str, ...]  # checked for sticking
    features: tuple[str, ...]
    feature_tag: dict[str, str]
    compute: Compute
    scored_ops: tuple[str, ...]
    stats_features: tuple[str, ...]
    limit_features: dict[str, str]  # signal -> feature holding its window mean or error
    suggest: Callable[[dict[str, float]], FaultType | None]
    align: str = "operation"  # or "inoculation" (the bioreactor)
    rules_skip_ops: tuple[str, ...] = ()
    roc: Roc | None = None
    settle_new_operation: bool = True
    feature_index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_index", {f: i for i, f in enumerate(self.features)})

    @property
    def signals(self) -> tuple[str, ...]:
        return self.pv + self.sp

    def signal_of(self, topic: str) -> str | None:
        return bio.signal_of_topic(topic, self.pv, self.sp)

    def series(self, origin: datetime) -> Series:
        return Series(origin, self.signals)

    def age_h(self, ctx: Context, minute: float) -> float:
        return ctx.age_h(minute) if self.align == "inoculation" else ctx.operation_age_h(minute)

    def settling(self, ctx: Context, minute: float) -> bool:
        """No statistical scoring (and, off the bioreactor, no limit rules) here."""
        if self.align == "inoculation":
            return ctx.settling(minute)
        return self.settle_new_operation and ctx.new_operation(
            minute, WINDOW_MIN + SETTLE_EXTRA_MIN
        )


def _window(series: Series, end: int) -> tuple[np.ndarray, dict[str, int]] | None:
    w = grid(series, end - WINDOW_MIN, end)
    if np.isnan(w).any():
        return None
    return w, {s: i for i, s in enumerate(series.signals)}


def _rate(y: np.ndarray) -> float:
    """Change over the window of a totaliser, per hour."""
    return float((y[-1] - y[0]) / (WINDOW_MIN / 60.0))


# --- bioreactor (the original constants) ----------------------------------------------------


def _bio_compute(series: Series, end: int, ctx: Context) -> np.ndarray | None:
    return bio.window_features(series, end, ctx.inoculated_at(end))


def _bio_suggest(z: dict[str, float]) -> FaultType | None:
    if z["hours_since_bolus"] > 3 or z["feed_rate"] < -3:
        return FaultType.FEED_PUMP_FAILURE
    if z["do_err_mean"] < -3 and z["ph_err_mean"] < -2:
        return FaultType.CONTAMINATION
    if z["temp_err_std"] > 3 or abs(z["temp_slope"]) > 3:
        return FaultType.TEMP_CONTROL_LOSS
    if (z["agitation_mean"] > 2 or z["o2_mean"] > 2) and z["do_err_mean"] < 0:
        return FaultType.DO_SPARGER_FOULING
    if z["co2_mean"] > 2 or z["base_rate"] < -2:
        return FaultType.PH_PROBE_DRIFT
    return None


BIOREACTOR = AnomalyProfile(
    cls="bioreactor",
    pv=bio.PV,
    sp=bio.SP,
    analog=bio.ANALOG,
    features=bio.FEATURES,
    feature_tag=bio.FEATURE_TAG,
    compute=_bio_compute,
    scored_ops=("Growth", "Production"),
    stats_features=(
        "temp_err_mean",
        "temp_err_std",
        "ph_err_mean",
        "do_err_mean",
        "agitation_mean",
        "o2_mean",
        "co2_mean",
        "base_rate",
        "hours_since_bolus",
        "pressure_mean",
        "weight_slope",
    ),
    limit_features={
        "temperature": "temp_err_mean",
        "ph": "ph_err_mean",
        "do": "do_err_mean",
        "pressure": "pressure_mean",
    },
    suggest=_bio_suggest,
    align="inoculation",
    rules_skip_ops=("Setup", "Harvest"),
    roc=Roc("temp_slope", "temperature", 1.5, ("TempShift",), FaultType.TEMP_CONTROL_LOSS),
    settle_new_operation=False,
)
assert SHIFT_SETTLE_MIN > 0  # the bioreactor settles through TempShift (ai.context)


# --- reactor-crystallizer (RX-201) ------------------------------------------------------------

RX_FEATURES = (
    "temp_err_mean",
    "temp_err_std",
    "jacket_mean",
    "jacket_delta",
    "jacket_slope",
    "power_mean",
    "power_per_speed",
    "agitation_mean",
    "dose_rate",
    "dose_flow_err_mean",
    "conversion_mean",
    "conversion_slope",
    "chord_mean",
    "chord_slope",
    "pressure_mean",
    "level_mean",
)


def _rx_compute(series: Series, end: int, ctx: Context) -> np.ndarray | None:
    got = _window(series, end)
    if got is None:
        return None
    w, i = got
    temp, jacket = w[:, i["temperature"]], w[:, i["jacket_temperature"]]
    speed = np.maximum(w[:, i["agitation"]], 10.0)
    power = w[:, i["agitator_power"]]
    return np.array(
        [
            (temp - w[:, i["sp_temperature"]]).mean(),
            (temp - w[:, i["sp_temperature"]]).std(),
            jacket.mean(),
            (jacket - temp).mean(),
            slope(jacket),
            power.mean(),
            (power / (speed / 120.0) ** 3).mean(),
            speed.mean(),
            _rate(w[:, i["dose_total"]]),
            (w[:, i["dose_flow"]] - w[:, i["sp_dose_flow"]]).mean(),
            w[:, i["conversion"]].mean(),
            slope(w[:, i["conversion"]]),
            w[:, i["chord_length"]].mean(),
            slope(w[:, i["chord_length"]]),
            w[:, i["pressure"]].mean(),
            w[:, i["level"]].mean(),
        ]
    )


def _rx_suggest(z: dict[str, float]) -> FaultType | None:
    if z["level_mean"] < -3 or (z["conversion_mean"] < -2 and z["power_per_speed"] > -3):
        return FaultType.DOSING_METER_DRIFT  # less in the vessel than the meter says
    if z["power_per_speed"] < -3:
        return FaultType.AGITATOR_DEGRADATION
    if z["jacket_delta"] < -3 or (z["jacket_mean"] < -3 and z["temp_err_mean"] > 0):
        return FaultType.JACKET_FOULING
    return None


REACTOR = AnomalyProfile(
    cls="reactor",
    pv=(
        "temperature",
        "jacket_temperature",
        "agitation",
        "agitator_power",
        "dose_flow",
        "dose_total",
        "pressure",
        "level",
        "conversion",
        "chord_length",
    ),
    sp=("sp_temperature", "sp_agitation", "sp_dose_flow"),
    analog=(
        "temperature",
        "jacket_temperature",
        "agitation",
        "agitator_power",
        "pressure",
        "level",
        "conversion",
        "chord_length",
    ),
    features=RX_FEATURES,
    feature_tag={
        "temp_err_mean": "temperature",
        "temp_err_std": "temperature",
        "jacket_mean": "jacket_temperature",
        "jacket_delta": "jacket_temperature",
        "jacket_slope": "jacket_temperature",
        "power_mean": "agitator_power",
        "power_per_speed": "agitator_power",
        "agitation_mean": "agitation",
        "dose_rate": "dose_total",
        "dose_flow_err_mean": "dose_flow",
        "conversion_mean": "conversion",
        "conversion_slope": "conversion",
        "chord_mean": "chord_length",
        "chord_slope": "chord_length",
        "pressure_mean": "pressure",
        "level_mean": "level",
    },
    compute=_rx_compute,
    scored_ops=("Reaction", "Crystallization"),
    stats_features=(
        "temp_err_mean",
        "jacket_delta",
        "power_per_speed",
        "conversion_mean",
        "conversion_slope",
        "chord_mean",
        "pressure_mean",
        "level_mean",
    ),
    limit_features={"temperature": "temp_err_mean"},
    suggest=_rx_suggest,
)


# --- filter-dryer (FD-202) ----------------------------------------------------------------------

FD_FEATURES = (
    "fp_err_mean",
    "flow_mean",
    "flow_slope",
    "filtrate_rate",
    "jacket_err_mean",
    "cake_mean",
    "cake_slope",
    "vacuum_mean",
    "vacuum_slope",
    "moisture_mean",
    "moisture_slope",
    "agitation_mean",
)


def _fd_compute(series: Series, end: int, ctx: Context) -> np.ndarray | None:
    got = _window(series, end)
    if got is None:
        return None
    w, i = got
    return np.array(
        [
            (w[:, i["filter_pressure"]] - w[:, i["sp_filter_pressure"]]).mean(),
            w[:, i["filtrate_flow"]].mean(),
            slope(w[:, i["filtrate_flow"]]),
            _rate(w[:, i["filtrate_total"]]),
            (w[:, i["jacket_temperature"]] - w[:, i["sp_jacket_temperature"]]).mean(),
            w[:, i["cake_temperature"]].mean(),
            slope(w[:, i["cake_temperature"]]),
            w[:, i["vacuum"]].mean(),
            slope(w[:, i["vacuum"]]),
            w[:, i["moisture"]].mean(),
            slope(w[:, i["moisture"]]),
            w[:, i["agitation"]].mean(),
        ]
    )


def _fd_suggest(z: dict[str, float]) -> FaultType | None:
    if z["vacuum_mean"] > 3 or z["vacuum_slope"] > 3:
        return FaultType.VACUUM_LEAK
    if z["flow_mean"] < -2 or z["flow_slope"] < -2 or z["filtrate_rate"] < -2:
        return FaultType.FILTER_BLINDING
    return None


FILTER_DRYER = AnomalyProfile(
    cls="filter_dryer",
    pv=(
        "filter_pressure",
        "filtrate_flow",
        "filtrate_total",
        "jacket_temperature",
        "cake_temperature",
        "vacuum",
        "agitation",
        "moisture",
    ),
    sp=("sp_filter_pressure", "sp_jacket_temperature", "sp_vacuum"),
    analog=(
        "filter_pressure",
        "filtrate_flow",
        "jacket_temperature",
        "cake_temperature",
        "vacuum",
        "moisture",
    ),
    features=FD_FEATURES,
    feature_tag={
        "fp_err_mean": "filter_pressure",
        "flow_mean": "filtrate_flow",
        "flow_slope": "filtrate_flow",
        "filtrate_rate": "filtrate_total",
        "jacket_err_mean": "jacket_temperature",
        "cake_mean": "cake_temperature",
        "cake_slope": "cake_temperature",
        "vacuum_mean": "vacuum",
        "vacuum_slope": "vacuum",
        "moisture_mean": "moisture",
        "moisture_slope": "moisture",
        "agitation_mean": "agitation",
    },
    compute=_fd_compute,
    scored_ops=("Filtration", "Drying"),
    stats_features=(
        "fp_err_mean",
        "flow_mean",
        "flow_slope",
        "filtrate_rate",
        "cake_mean",
        "vacuum_mean",
        "moisture_mean",
        "moisture_slope",
    ),
    limit_features={"filter_pressure": "fp_err_mean", "vacuum": "vacuum_mean"},
    suggest=_fd_suggest,
)


# --- roller compactor (RC-302) --------------------------------------------------------------------

RC_FEATURES = (
    "force_err_mean",
    "force_std",
    "gap_mean",
    "gap_std",
    "roll_speed_mean",
    "screw_mean",
    "screw_slope",
    "density_mean",
    "density_slope",
    "output_rate",
    "mill_mean",
    "rh_mean",
    "rh_slope",
)


def _rc_compute(series: Series, end: int, ctx: Context) -> np.ndarray | None:
    got = _window(series, end)
    if got is None:
        return None
    w, i = got
    force = w[:, i["roll_force"]]
    return np.array(
        [
            (force - w[:, i["sp_roll_force"]]).mean(),
            force.std(),
            w[:, i["roll_gap"]].mean(),
            w[:, i["roll_gap"]].std(),
            w[:, i["roll_speed"]].mean(),
            w[:, i["screw_speed"]].mean(),
            slope(w[:, i["screw_speed"]]),
            w[:, i["ribbon_density"]].mean(),
            slope(w[:, i["ribbon_density"]]),
            _rate(w[:, i["granule_mass"]]),
            w[:, i["mill_speed"]].mean(),
            w[:, i["room_rh"]].mean(),
            slope(w[:, i["room_rh"]]),
        ]
    )


def _rc_suggest(z: dict[str, float]) -> FaultType | None:
    if z["rh_mean"] > 3 or z["rh_slope"] > 3:
        return FaultType.HVAC_HUMIDITY
    if z["density_mean"] < -3 or z["screw_mean"] < -3 or z["gap_mean"] > 3:
        return FaultType.ROLL_FORCE_DRIFT
    return None


ROLLER_COMPACTOR = AnomalyProfile(
    cls="roller_compactor",
    pv=(
        "roll_force",
        "roll_gap",
        "roll_speed",
        "screw_speed",
        "mill_speed",
        "ribbon_density",
        "granule_mass",
        "room_rh",
    ),
    sp=("sp_roll_force",),
    analog=(
        "roll_force",
        "roll_gap",
        "roll_speed",
        "screw_speed",
        "mill_speed",
        "ribbon_density",
        "room_rh",
    ),
    features=RC_FEATURES,
    feature_tag={
        "force_err_mean": "roll_force",
        "force_std": "roll_force",
        "gap_mean": "roll_gap",
        "gap_std": "roll_gap",
        "roll_speed_mean": "roll_speed",
        "screw_mean": "screw_speed",
        "screw_slope": "screw_speed",
        "density_mean": "ribbon_density",
        "density_slope": "ribbon_density",
        "output_rate": "granule_mass",
        "mill_mean": "mill_speed",
        "rh_mean": "room_rh",
        "rh_slope": "room_rh",
    },
    compute=_rc_compute,
    scored_ops=("Compaction",),
    stats_features=(
        "force_err_mean",
        "gap_mean",
        "screw_mean",
        "density_mean",
        "output_rate",
        "rh_mean",
        "rh_slope",
    ),
    limit_features={"roll_force": "force_err_mean", "room_rh": "rh_mean"},
    suggest=_rc_suggest,
)


# --- rotary tablet press (TP-303) ---------------------------------------------------------------

TP_FEATURES = (
    "force_err_mean",
    "force_std",
    "precomp_mean",
    "turret_err_mean",
    "feed_frame_err_mean",
    "ejection_mean",
    "ejection_slope",
    "weight_mean",
    "weight_std",
    "weight_rsd_mean",
    "hardness_mean",
    "production_rate",
    "reject_rate",
    "rh_mean",
    "rh_slope",
)


def _tp_compute(series: Series, end: int, ctx: Context) -> np.ndarray | None:
    got = _window(series, end)
    if got is None:
        return None
    w, i = got
    force, weight = w[:, i["comp_force"]], w[:, i["tablet_weight"]]
    made = _rate(w[:, i["tablets_total"]])
    return np.array(
        [
            (force - w[:, i["sp_comp_force"]]).mean(),
            force.std(),
            w[:, i["precomp_force"]].mean(),
            (w[:, i["turret_speed"]] - w[:, i["sp_turret_speed"]]).mean(),
            (w[:, i["feed_frame"]] - w[:, i["sp_feed_frame"]]).mean(),
            w[:, i["ejection_force"]].mean(),
            slope(w[:, i["ejection_force"]]),
            weight.mean(),
            weight.std(),
            w[:, i["weight_rsd"]].mean(),
            w[:, i["hardness"]].mean(),
            made,
            _rate(w[:, i["rejects_total"]]) / max(made, 1.0) * 100.0,
            w[:, i["room_rh"]].mean(),
            slope(w[:, i["room_rh"]]),
        ]
    )


def _tp_suggest(z: dict[str, float]) -> FaultType | None:
    if z["rh_mean"] > 3 or z["rh_slope"] > 3:
        return FaultType.HVAC_HUMIDITY
    if z["ejection_mean"] > 3 or z["ejection_slope"] > 3:
        return FaultType.PUNCH_STICKING
    if z["weight_std"] > 3 or z["force_std"] > 3 or z["weight_rsd_mean"] > 3:
        return FaultType.HOPPER_BRIDGING
    return None


TABLET_PRESS = AnomalyProfile(
    cls="tablet_press",
    pv=(
        "comp_force",
        "precomp_force",
        "turret_speed",
        "feed_frame",
        "ejection_force",
        "tablet_weight",
        "weight_rsd",
        "hardness",
        "tablets_total",
        "rejects_total",
        "room_rh",
    ),
    sp=("sp_comp_force", "sp_turret_speed", "sp_feed_frame"),
    analog=(
        "comp_force",
        "precomp_force",
        "turret_speed",
        "feed_frame",
        "ejection_force",
        "tablet_weight",
        "weight_rsd",
        "hardness",
        "room_rh",
    ),
    features=TP_FEATURES,
    feature_tag={
        "force_err_mean": "comp_force",
        "force_std": "comp_force",
        "precomp_mean": "precomp_force",
        "turret_err_mean": "turret_speed",
        "feed_frame_err_mean": "feed_frame",
        "ejection_mean": "ejection_force",
        "ejection_slope": "ejection_force",
        "weight_mean": "tablet_weight",
        "weight_std": "tablet_weight",
        "weight_rsd_mean": "weight_rsd",
        "hardness_mean": "hardness",
        "production_rate": "tablets_total",
        "reject_rate": "rejects_total",
        "rh_mean": "room_rh",
        "rh_slope": "room_rh",
    },
    compute=_tp_compute,
    scored_ops=("Compression",),
    stats_features=(
        "force_std",
        "ejection_mean",
        "ejection_slope",
        "weight_std",
        "weight_rsd_mean",
        "hardness_mean",
        "reject_rate",
        "rh_mean",
        "rh_slope",
    ),
    limit_features={
        "comp_force": "force_err_mean",
        "tablet_weight": "weight_mean",
        "weight_rsd": "weight_rsd_mean",
        "ejection_force": "ejection_mean",
        "room_rh": "rh_mean",
    },
    suggest=_tp_suggest,
)


PROFILES: dict[str, AnomalyProfile] = {
    p.cls: p for p in (BIOREACTOR, REACTOR, FILTER_DRYER, ROLLER_COMPACTOR, TABLET_PRESS)
}


def profile_for(cls: str) -> AnomalyProfile:
    """The profile for an equipment class; KeyError for a class that is not scored
    (the blender: its operations are shorter than one window)."""
    return PROFILES[cls]
