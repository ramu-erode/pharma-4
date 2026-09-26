"""The levers a batch actually ran with (ADR-0015: levers are explicit model inputs).

Actual = the levers planned at BATCH_START, updated by every operator change in time
order. Both come from the UNS (`events/batch`, `events/operator`), so graph-sync (live)
and yield training (from `uns_events`) derive them with this one function.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

from common.models import OperatorEvent


def actual_levers[L: BaseModel](planned: L, operator_events: Iterable[OperatorEvent]) -> L:
    """Works for any process's levers (ADR-0018): changes to other fields are ignored."""
    levers = planned
    for ev in operator_events:
        if ev.parameter in type(planned).model_fields:
            levers = levers.model_copy(update={ev.parameter: ev.new})
    return levers


# Which UNS tag an operator change of each lever acts on (for Event-[:ON_TAG]->Tag), on
# the unit the change is published on. Lever names are unique across processes.
LEVER_TAG: dict[str, tuple[str, str]] = {
    # bioreactor
    "shift_day": ("sp", "temperature"),
    "prod_temp": ("sp", "temperature"),
    "ph_sp": ("sp", "ph"),
    "do_sp": ("sp", "do"),
    "feed_mult": ("pv", "feed_total"),
    # API: RX-201, except the dryer jacket on FD-202
    "rxn_temp": ("sp", "temperature"),
    "ac2o_ratio": ("sp", "dose_flow"),
    "rxn_time": ("sp", "temperature"),
    "cool_rate": ("sp", "temperature"),
    "dry_temp": ("sp", "jacket_temperature"),
    # OSD: BL-301, RC-302, TP-303
    "lube_time": ("sp", "speed"),
    "roll_force": ("sp", "roll_force"),
    "comp_force": ("sp", "comp_force"),
    "turret_speed": ("sp", "turret_speed"),
    "feed_frame": ("sp", "feed_frame"),
}
