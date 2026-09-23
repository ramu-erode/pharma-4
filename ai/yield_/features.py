"""Mid-batch features for titer prediction (ADR-0014). One function, `features_at`, for
training rows and live predictions alike.

A row at batch day d holds what is known by then (lab results, process summaries,
alerts so far) plus the batch's *actual whole-batch lever values*. The optimizer varies
only the lever columns; everything else is the observed trajectory, held fixed.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from ai.context import Context
from ai.features import SIGNALS, Series, grid
from common.models import Levers

LEVERS = ("shift_day", "prod_temp", "ph_sp", "do_sp", "feed_mult")
LAB = ("vcd", "viability", "glucose", "lactate", "titer")
FEATURES = (
    "day",
    *LEVERS,
    "vcd", "viability", "glucose", "lactate", "titer_so_far",
    "titer_rate", "titer_projected",
    "ivc", "growth_rate",
    "temp_err_abs_mean", "ph_err_mean", "do_err_std", "feed_total", "base_total",
    "alerts_so_far", "shifted",
)  # fmt: skip
FIRST_DAY, LAST_DAY, STEP_DAY = 3.0, 12.0, 0.5
PLANNED_HARVEST_DAY = 14.0
_S = {s: i for i, s in enumerate(SIGNALS)}


@dataclass
class YieldInput:
    series: Series
    ctx: Context
    lab: dict[str, list[tuple[float, float]]]
    levers: Levers
    alert_minutes: list[float] = field(default_factory=list)


def _latest(points: list[tuple[float, float]], minute: float) -> float:
    vals = [v for t, v in points if t <= minute]
    return vals[-1] if vals else math.nan


def features_at(inp: YieldInput, day: float) -> np.ndarray:
    inoc = inp.ctx.inoculated_at(inp.series.last_minute)
    minute = inoc + day * 1440.0
    lab = {k: inp.lab.get(k, []) for k in LAB}
    vcd_pts = [(t, v) for t, v in lab["vcd"] if t <= minute]
    ivc = 0.0
    for (t0, v0), (t1, v1) in itertools.pairwise(vcd_pts):
        ivc += (v0 + v1) / 2 * (t1 - t0) / 1440.0
    growth = math.nan
    if len(vcd_pts) >= 2 and vcd_pts[-2][1] > 0 and vcd_pts[-1][1] > 0:
        (t0, v0), (t1, v1) = vcd_pts[-2], vcd_pts[-1]
        growth = math.log(v1 / v0) / max((t1 - t0) / 1440.0, 1e-6)

    titer_pts = [(t, v) for t, v in lab["titer"] if t <= minute]
    titer_rate = projected = math.nan
    if len(titer_pts) >= 2:
        (t0, v0), (t1, v1) = titer_pts[-2], titer_pts[-1]
        titer_rate = (v1 - v0) / max((t1 - t0) / 1440.0, 1e-6)
        remaining = PLANNED_HARVEST_DAY - (t1 - inoc) / 1440.0
        projected = v1 + titer_rate * max(remaining, 0.0)

    g = grid(inp.series, int(inoc), max(int(minute), int(inoc) + 1))
    temp_err = g[:, _S["temperature"]] - g[:, _S["sp_temperature"]]
    ph_err = g[:, _S["ph"]] - g[:, _S["sp_ph"]]
    do_err = g[:, _S["do"]] - g[:, _S["sp_do"]]
    last = g[-1]
    shift_start = next(
        (start for name, start, _ in inp.ctx.operations if name == "TempShift"), math.inf
    )
    lv = inp.levers
    with np.errstate(all="ignore"):
        return np.array(
            [
                day,
                lv.shift_day, lv.prod_temp, lv.ph_sp, lv.do_sp, lv.feed_mult,
                _latest(lab["vcd"], minute), _latest(lab["viability"], minute),
                _latest(lab["glucose"], minute), _latest(lab["lactate"], minute),
                _latest(lab["titer"], minute),
                titer_rate, projected,
                ivc, growth,
                float(np.nanmean(np.abs(temp_err))), float(np.nanmean(ph_err)),
                float(np.nanstd(do_err)),
                float(last[_S["feed_total"]]), float(last[_S["base_total"]]),
                float(sum(1 for a in inp.alert_minutes if a <= minute)),
                1.0 if shift_start <= minute else 0.0,
            ]
        )  # fmt: skip


def with_levers(x: np.ndarray, levers: dict[str, float]) -> np.ndarray:
    """A copy of feature row(s) with some lever columns replaced (the optimizer's move)."""
    out = np.array(x, dtype=float, copy=True)
    for k, v in levers.items():
        out[..., FEATURES.index(k)] = v
    return out


def training_days(harvest_day: float | None) -> list[float]:
    last = min(LAST_DAY, (harvest_day or 14.0) - 0.5)
    return [float(d) for d in np.arange(FIRST_DAY, last + 1e-9, STEP_DAY)]
