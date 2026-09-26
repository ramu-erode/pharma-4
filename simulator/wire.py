"""Engine messages -> (topic, payload) on the wire. The live runner publishes them;
backfill and the harness feed the same pairs to the historian and AI cores instead.

A message that names its `cell` (a unit of a train, ADR-0018) goes to that unit's
topics; otherwise to `unit`, the unit the batch started on."""

from __future__ import annotations

from pydantic import BaseModel

from common import models as m
from common import uns
from common.models import Src
from common.plant import get_plant
from common.uns import UnitPath
from simulator.messages import (
    EventOut,
    LabelOut,
    LabOut,
    RawOut,
    SimMessage,
    StateOut,
    TruthOut,
)


def to_wire(unit: UnitPath, batch_id: str | None, msg: SimMessage) -> tuple[str, BaseModel] | None:
    """Topic and payload for one engine message; None for test-only TruthOut."""
    if msg.cell is not None and msg.cell != unit.cell:
        unit = get_plant().path(msg.cell)
    match msg:
        case RawOut(tag=tag, value=value, t=t, q=q):
            return uns.edge_raw(unit.device, tag), m.RawSample(tag=tag, value=value, t=t, q=q)
        case LabOut(name=name, value=value, unit=eu, t=t):
            return uns.lab(unit, name), m.ScalarPayload(
                v=value, ts=t, unit=eu, batch=batch_id, src=Src.SIM
            )
        case StateOut(name="batch", value=value, t=t):
            return uns.state_batch(unit), m.BatchStatePayload(
                v=value, ts=t, unit=None, batch=value, src=Src.SIM
            )
        case StateOut(name="operation", value=value, t=t):
            return uns.state_operation(unit), m.OperationPayload(
                v=m.Operation(value), ts=t, unit=None, batch=batch_id, src=Src.SIM
            )
        case StateOut(name=name, value=value, t=t) if name.startswith("phase/"):
            return uns.state_phase(unit, name.removeprefix("phase/")), m.PhaseStatePayload(
                v=m.PhaseState(value), ts=t, unit=None, batch=batch_id, src=Src.SIM
            )
        case EventOut(name="batch", v=v, t=t):
            return uns.events(unit, "batch"), m.BatchEventPayload(
                v=v, ts=t, unit=None, batch=batch_id, src=Src.SIM
            )
        case EventOut(name="material", v=v, t=t):
            return uns.events(unit, "material"), m.MaterialEventPayload(
                v=v, ts=t, unit="kg", batch=batch_id, src=Src.SIM
            )
        case EventOut(name="operator", v=v, t=t):
            return uns.events(unit, "operator"), m.OperatorEventPayload(
                v=v, ts=t, unit=None, batch=batch_id, src=Src.OPERATOR
            )
        case LabelOut(label=label, t=t):
            return uns.sim_faults(unit.cell), m.FaultLabelPayload(
                v=label, ts=t, unit=None, batch=label.batch_id, src=Src.SIM
            )
        case TruthOut():
            return None
    raise TypeError(f"unhandled engine message {msg!r}")
