"""Window features: the one function training and live scoring share (plan task 4.1).

Input is what the UNS carries after the deadband: per signal, the published times and
values (irregular). `window_features` holds each signal's last value forward on a
1-minute grid over a 30-minute window and computes the features from that. Training
calls it for every window of a historical batch; the live service calls it for the
window that just closed. Same function, same inputs, same numbers (parity test).

Time is minutes since the batch's origin (its BATCH_START, or for a unit of a train,
when the batch arrived on it).

The constants and `window_features` here are the bioreactor's. The other equipment
classes bring their own signals and feature functions (`ai.profiles`, ADR-0021); the
series, the grid and the stuck/quality checks are shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from common import uns
from common.uns import TopicClass

WINDOW_MIN = 30
STEP_MIN = 5

PV = (
    "temperature", "ph", "do", "agitation", "air_flow", "o2_flow", "co2_flow",
    "feed_total", "base_total", "pressure", "weight",
)  # fmt: skip
SP = ("sp_temperature", "sp_ph", "sp_do")
SIGNALS = PV + SP
ANALOG = tuple(s for s in PV if not s.endswith("_total"))  # noisy: can be checked for sticking
BOLUS_STEP = 0.2  # L: a feed_total rise larger than this between publishes is a bolus

FEATURES = (
    "temp_err_mean", "temp_err_std", "temp_slope",
    "ph_err_mean", "ph_std",
    "do_err_mean", "do_std",
    "agitation_mean", "agitation_slope",
    "o2_mean", "co2_mean", "co2_slope",
    "base_rate", "feed_rate",
    "pressure_mean", "pressure_std",
    "weight_slope", "agit_do_ratio", "hours_since_bolus",
)  # fmt: skip

# Which UNS tag each feature speaks for (alert contributions name tags, ADR-0013).
FEATURE_TAG = {
    "temp_err_mean": "temperature", "temp_err_std": "temperature", "temp_slope": "temperature",
    "ph_err_mean": "ph", "ph_std": "ph",
    "do_err_mean": "do", "do_std": "do",
    "agitation_mean": "agitation", "agitation_slope": "agitation", "agit_do_ratio": "agitation",
    "o2_mean": "o2_flow", "co2_mean": "co2_flow", "co2_slope": "co2_flow",
    "base_rate": "base_total", "feed_rate": "feed_total", "hours_since_bolus": "feed_total",
    "pressure_mean": "pressure", "pressure_std": "pressure", "weight_slope": "weight",
}  # fmt: skip

_IDX = {s: i for i, s in enumerate(SIGNALS)}
_T_H = (np.arange(WINDOW_MIN) - (WINDOW_MIN - 1) / 2) / 60.0  # centred window time, hours
_T_SS = float((_T_H**2).sum())


def signal_of_topic(topic: str, pv: tuple[str, ...] = PV, sp: tuple[str, ...] = SP) -> str | None:
    """pv/ph -> "ph", sp/ph -> "sp_ph"; None for anything that is not a feature input."""
    try:
        p = uns.parse(topic)
    except uns.TopicError:
        return None
    if p.cls is TopicClass.PV and p.name in pv:
        return p.name
    if p.cls is TopicClass.SP and f"sp_{p.name}" in sp:
        return f"sp_{p.name}"
    return None


@dataclass
class Series:
    """Published values of one batch on one unit, per signal, in arrival order."""

    origin: datetime
    signals: tuple[str, ...] = SIGNALS
    t: dict[str, list[float]] = field(default_factory=dict)
    v: dict[str, list[float]] = field(default_factory=dict)
    uncertain: dict[str, list[bool]] = field(default_factory=dict)
    _arrays: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for s in self.signals:
            self.t.setdefault(s, [])
            self.v.setdefault(s, [])
            self.uncertain.setdefault(s, [])

    def add(self, signal: str, ts: datetime, value: float, uncertain: bool = False) -> None:
        self.t[signal].append((ts - self.origin).total_seconds() / 60.0)
        self.v[signal].append(value)
        self.uncertain[signal].append(uncertain)
        self._arrays.pop(signal, None)

    def arrays(self, signal: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._arrays.get(signal)
        if cached is None:
            cached = (np.asarray(self.t[signal]), np.asarray(self.v[signal]))
            self._arrays[signal] = cached
        return cached

    @property
    def last_minute(self) -> float:
        return max((ts[-1] for ts in self.t.values() if ts), default=0.0)


def grid(series: Series, start: int, stop: int) -> np.ndarray:
    """Values held forward on minutes [start, stop): shape (stop-start, len(signals)).
    A minute's value is the last one published at or before the end of that minute."""
    minutes = np.arange(start, stop, dtype=float) + 1.0
    out = np.full((stop - start, len(series.signals)), np.nan)
    for j, s in enumerate(series.signals):
        t, v = series.arrays(s)
        if len(t) == 0:
            continue
        idx = np.searchsorted(t, minutes, side="right") - 1
        ok = idx >= 0
        out[ok, j] = v[idx[ok]]
    return out


def last_bolus(series: Series, end: int) -> float | None:
    """Minute of the last feed bolus published before `end` (None if none yet)."""
    t, v = series.arrays("feed_total")
    n = int(np.searchsorted(t, end, side="left"))
    if n < 2:
        return None
    rises = np.nonzero(np.diff(v[:n]) > BOLUS_STEP)[0]
    return float(t[rises[-1] + 1]) if len(rises) else None


def slope(y: np.ndarray) -> float:
    """Least-squares slope per hour over a 30-minute window."""
    return float(((y - y.mean()) * _T_H).sum() / _T_SS)


def _slope(y: np.ndarray) -> float:
    return float(((y - y.mean()) * _T_H).sum() / _T_SS)


def window_features(series: Series, end: int, inoculation_min: float) -> np.ndarray | None:
    """Features of the window [end-30, end) minutes; None if any signal has no value yet."""
    w = grid(series, end - WINDOW_MIN, end)
    if np.isnan(w).any():
        return None

    def col(s: str) -> np.ndarray:
        return w[:, _IDX[s]]

    temp_err = col("temperature") - col("sp_temperature")
    ph_err = col("ph") - col("sp_ph")
    do_err = col("do") - col("sp_do")
    agitation, do = col("agitation"), col("do")
    half_h = WINDOW_MIN / 60.0
    bolus = last_bolus(series, end)
    since = (end - (bolus if bolus is not None else inoculation_min)) / 60.0
    return np.array(
        [
            temp_err.mean(),
            temp_err.std(),
            _slope(col("temperature")),
            ph_err.mean(),
            col("ph").std(),
            do_err.mean(),
            do.std(),
            agitation.mean(),
            _slope(agitation),
            col("o2_flow").mean(),
            col("co2_flow").mean(),
            _slope(col("co2_flow")),
            (col("base_total")[-1] - col("base_total")[0]) / half_h,
            (col("feed_total")[-1] - col("feed_total")[0]) / half_h,
            col("pressure").mean(),
            col("pressure").std(),
            _slope(col("weight")),
            agitation.mean() / max(do.mean(), 1.0),
            since,
        ]
    )


def stuck_signals(
    series: Series, end: int, repeats: int = 3, analog: tuple[str, ...] = ANALOG
) -> list[str]:
    """Analog signals whose last `repeats` published values before `end` are identical.
    A healthy sensor always carries noise, so exact repeats mean a stuck reading (Q8)."""
    out = []
    for s in analog:
        t, v = series.arrays(s)
        n = int(np.searchsorted(t, end, side="left"))
        if n >= repeats and np.all(v[n - repeats : n] == v[n - 1]):
            out.append(s)
    return out


def uncertain_signals(series: Series, end: int) -> list[str]:
    """Signals whose latest published quality before `end` is not GOOD."""
    out = []
    for s in series.signals:
        t, _ = series.arrays(s)
        n = int(np.searchsorted(t, end, side="left"))
        if n and series.uncertain[s][n - 1]:
            out.append(s)
    return out
