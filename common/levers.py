"""The levers a batch actually ran with (ADR-0014: levers are explicit model inputs).

Actual = the levers planned at BATCH_START, updated by every operator change in time
order. Both come from the UNS (`events/batch`, `events/operator`), so graph-sync (live)
and yield training (from `uns_events`) derive them with this one function.
"""

from __future__ import annotations

from collections.abc import Iterable

from common.models import Levers, OperatorEvent


def actual_levers(planned: Levers, operator_events: Iterable[OperatorEvent]) -> Levers:
    levers = planned
    for ev in operator_events:
        if ev.parameter in Levers.model_fields:
            levers = levers.model_copy(update={ev.parameter: ev.new})
    return levers


# Which UNS tag an operator change of each lever acts on (for Event-[:ON_TAG]->Tag).
LEVER_TAG: dict[str, tuple[str, str]] = {
    "shift_day": ("sp", "temperature"),
    "prod_temp": ("sp", "temperature"),
    "ph_sp": ("sp", "ph"),
    "do_sp": ("sp", "do"),
    "feed_mult": ("pv", "feed_total"),
}
