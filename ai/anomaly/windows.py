"""Build detector windows from a batch's published series and context.

`window_at` is the one place a window is assembled. Offline code loops it over a whole
batch (`batch_windows`); the live service calls it for the window that just closed.
"""

from __future__ import annotations

import math

from ai.anomaly.detector import Window
from ai.context import Context
from ai.features import STEP_MIN, WINDOW_MIN, Series, grid, stuck_signals, uncertain_signals
from ai.profiles import BIOREACTOR, AnomalyProfile


def window_at(
    series: Series,
    ctx: Context,
    controlled: dict[str, set[str]],
    end: int,
    profile: AnomalyProfile = BIOREACTOR,
) -> Window | None:
    x = profile.compute(series, end, ctx)
    if x is None:
        return None
    held: set[str] = set()
    for phase in ctx.held_phases(end - 1):
        held |= controlled.get(phase, set())
    last = grid(series, end - 1, end)[0]
    sp = {name.removeprefix("sp_"): float(last[series.signals.index(name)]) for name in profile.sp}
    return Window(
        end=end,
        operation=ctx.operation_at(end - 1),
        age_h=profile.age_h(ctx, end),
        settling=profile.settling(ctx, end - 1),
        held_signals=frozenset(held),
        x=x,
        stuck=stuck_signals(series, end, analog=profile.analog),
        uncertain=uncertain_signals(series, end),
        sp=sp,
    )


def batch_windows(
    series: Series,
    ctx: Context,
    controlled: dict[str, set[str]],
    until: float | None = None,
    profile: AnomalyProfile = BIOREACTOR,
) -> list[Window]:
    """Every window of a batch on one unit, one per STEP_MIN, in time order."""
    stop = until if until is not None else series.last_minute
    out = []
    for end in range(STEP_MIN * math.ceil(WINDOW_MIN / STEP_MIN), int(stop) + 1, STEP_MIN):
        w = window_at(series, ctx, controlled, end, profile)
        if w is not None:
            out.append(w)
    return out
