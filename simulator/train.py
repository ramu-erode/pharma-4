"""A batch that runs through a train of units (ADR-0018): the engine shell the API and
OSD processes share. The process itself (true state, kinetics, control) lives in the
subclasses; this class owns what every train batch does the same way:

- `state/batch` on the unit that holds the batch, `Idle` operations everywhere else;
- sequential operations per unit and their parallel phases (ADR-0011);
- `BATCH_START` on the first unit, `BATCH_END` on the unit that holds the batch last;
- raw DCS samples for every unit of the train on each publish tick, the units the batch
  is not on reading their idle values;
- faults armed relative to an operation's start, and their ground-truth labels;
- operator lever changes (ADR-0012), with the process deciding which are still open.

It has the same interface as the bioreactor's `BatchRun`: `advance`, `run_to_end`,
`inject`, `clear`, `abort`, `change_lever`, `snapshot`, `done`, `status`.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from common.models import (
    BatchEnded,
    BatchStarted,
    BatchStatus,
    Campaign,
    FaultLabel,
    FaultType,
    MaterialConsumed,
    MaterialProduced,
    Operation,
    OperationChanged,
    OperatorEvent,
    PhaseChanged,
    PhaseClass,
    PhaseState,
    Quality,
)
from common.plant import Plant, get_plant
from simulator.messages import (
    EventOut,
    LabelOut,
    LabOut,
    RawOut,
    SimMessage,
    StateOut,
    TruthOut,
    load_dcs_tags,
)
from simulator.sensors import SensorBank


@dataclass(frozen=True, slots=True)
class TrainFaultSpec:
    """A fault that starts `after_h` hours into `operation` (operation start times vary
    from batch to batch, so faults are planned against them)."""

    kind: FaultType
    operation: Operation
    after_h: float
    params: dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LotUse:
    """An input lot and how much of it a batch draws (ADR-0020)."""

    lot: str
    material: str
    quantity_kg: float
    properties: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainSpec:
    batch_id: str
    cell: str  # the first unit of the train
    start: datetime
    recipe_id: str
    campaign: Campaign
    levers: BaseModel
    seed: int
    traits: dict[str, float] = field(default_factory=dict)  # batch-to-batch variation
    faults: tuple[TrainFaultSpec, ...] = ()
    lots: tuple[LotUse, ...] = ()


@dataclass(slots=True)
class TrainFault:
    kind: FaultType
    cell: str
    onset_h: float
    params: dict[str, float | str]
    defaults: dict[str, float | str]
    label_id: str
    end_h: float | None = None
    noticed: bool = False  # stuck sensor: did the device-side check notice?
    onset_sent: bool = False
    end_sent: bool = False

    def active(self, t_h: float) -> bool:
        return t_h >= self.onset_h and (self.end_h is None or t_h < self.end_h)

    def param(self, key: str) -> float:
        return float(self.params.get(key, self.defaults[key]))

    def age(self, t_h: float) -> float:
        return t_h - self.onset_h

    @property
    def stuck_var(self) -> str:
        return str(self.params.get("tag", self.defaults.get("tag", "")))


class TrainRun:
    """One batch on one train. Subclasses implement the process."""

    process: ClassVar[str]
    material: ClassVar[str]  # what the batch produces, e.g. "Acetylsalicylic acid API"
    phases: ClassVar[dict[Operation, tuple[PhaseClass, ...]]]
    noise: ClassVar[dict[str, dict[str, float]]]  # equipment class -> variable -> sigma
    quantum: ClassVar[dict[str, dict[str, float]]]  # equipment class -> totalizer step
    idle: ClassVar[dict[str, dict[str, float]]]  # equipment class -> idle readings
    fault_class: ClassVar[dict[FaultType, str]]  # which equipment class a fault acts on
    fault_defaults: ClassVar[dict[FaultType, dict[str, float | str]]]
    lever_class: ClassVar[dict[str, str]]  # lever -> equipment class it acts on

    def __init__(
        self,
        spec: TrainSpec,
        dt_s: float = 5.0,
        publish_period_s: float = 60.0,
        record_truth: bool = False,
        quiet: bool = False,
        plant: Plant | None = None,
    ) -> None:
        steps = publish_period_s / dt_s
        if steps < 1 or abs(steps - round(steps)) > 1e-9:
            raise ValueError("publish period must be a whole multiple of the integration step")
        plant = plant or get_plant()
        self.spec = spec
        self.record_truth, self.quiet = record_truth, quiet
        self.cells: tuple[str, ...] = plant.train_of(spec.cell)
        if self.cells[0] != spec.cell:
            raise ValueError(f"a {self.process} batch starts on {self.cells[0]}, not {spec.cell}")
        self.cls_of = {c: plant.unit(c).cls for c in self.cells}
        self.cell_of_cls = {cls: c for c, cls in self.cls_of.items()}
        self.raw_tags = {
            c: [
                (f"{plant.path(c).device}.{suffix}", var)
                for suffix, var in load_dcs_tags(self.cls_of[c]).items()
            ]
            for c in self.cells
        }
        self.sensors = {
            c: SensorBank(noise=self.noise[self.cls_of[c]], quantum=self.quantum[self.cls_of[c]])
            for c in self.cells
        }
        self.rng = np.random.default_rng(spec.seed)
        self.levers = spec.levers
        self.dt_h = dt_s / 3600.0
        self.steps_per_publish = round(steps)
        self.k = 0
        self.op: dict[str, Operation] = {c: Operation.IDLE for c in self.cells}
        self.op_started_h: dict[str, float] = {c: 0.0 for c in self.cells}
        self.op_ended_h: dict[Operation, float] = {}
        self.op_started: dict[Operation, float] = {}
        self.running: dict[str, dict[PhaseClass, PhaseState]] = {c: {} for c in self.cells}
        self.active: str | None = None
        self.status = BatchStatus.RUNNING
        self.disposition: str | None = None
        self.done = False
        self.coa: dict[str, float] = {}  # true outcome, filled at the end
        self._out: list[SimMessage] = []
        self._faults: list[TrainFault] = []
        self._armed: list[TrainFaultSpec] = list(spec.faults)
        self.init_process()
        self._start()

    # -- to implement -----------------------------------------------------------------------

    def init_process(self) -> None:
        raise NotImplementedError

    def begin(self) -> None:
        """Put the batch on the first unit and start its first operation."""
        raise NotImplementedError

    def transition(self) -> None:
        """Decide operation changes at the current time."""
        raise NotImplementedError

    def integrate(self, dt: float) -> None:
        """Advance the true process by `dt` hours on the active unit."""
        raise NotImplementedError

    def read(self, cell: str, var: str) -> float:
        """The published reading of `var` on `cell` while the batch is on it."""
        raise NotImplementedError

    def truth(self) -> Any:
        """A copy of the true state, for tests (TruthOut)."""
        raise NotImplementedError

    def setpoints(self) -> dict[str, float]:
        return {}

    def lever_open(self, name: str) -> bool:
        """Whether an operator may still change this lever (ADR-0021 frozen levers)."""
        raise NotImplementedError

    def target(self) -> float:
        """The process's yield target, once the batch has ended (simulator.truth)."""
        raise NotImplementedError

    # -- public API -----------------------------------------------------------------------------

    @property
    def t_h(self) -> float:
        return self.k * self.dt_h

    @property
    def batch_day(self) -> float:
        return self.t_h / 24.0

    @property
    def operation(self) -> Operation:
        return self.op[self.active] if self.active else Operation.IDLE

    def t_of_day(self, day: float) -> float:
        """Hours since batch start at batch day `day` (days count from batch start)."""
        return day * 24.0

    def ts(self, t_h: float | None = None) -> datetime:
        return self.spec.start + timedelta(
            seconds=round((self.t_h if t_h is None else t_h) * 3600, 3)
        )

    def advance(self, until_h: float) -> list[SimMessage]:
        while not self.done and self.t_h < until_h - 1e-9:
            self._step()
        out, self._out = self._out, []
        return out

    def run_to_end(self) -> list[SimMessage]:
        return self.advance(math.inf)

    def inject(self, kind: FaultType, cell: str | None = None, **params: float | str) -> str:
        """Inject a fault now (live demo). Returns the label id."""
        kind = FaultType(kind)
        if kind not in self.fault_defaults:
            raise ValueError(f"{kind.value} is not a {self.process} fault")
        return self._add_fault(kind, self.t_h, dict(params), cell)

    def clear(self, kind: FaultType) -> None:
        for f in self._faults:
            if f.kind is kind and f.active(self.t_h):
                f.end_h = self.t_h

    def abort(self, reason: str = "operator abort") -> list[SimMessage]:
        if not self.done:
            self._end(BatchStatus.ABORTED, reason, "REJECTED")
        out, self._out = self._out, []
        return out

    def change_lever(
        self,
        name: str,
        value: float,
        operator: str,
        recommendation_id: str | None = None,
        reason: str | None = None,
    ) -> list[SimMessage]:
        """An operator changes a recipe parameter mid-batch (ADR-0012). Raises ValueError
        for an unknown lever or one whose window has passed."""
        if name not in type(self.levers).model_fields:
            raise ValueError(f"unknown parameter {name!r}")
        if not self.lever_open(name):
            raise ValueError(f"{name} can no longer be changed")
        old = float(getattr(self.levers, name))
        self.levers = self.levers.model_copy(update={name: float(value)})
        event = OperatorEvent(
            operator=operator,
            parameter=name,
            old=old,
            new=float(value),
            recommendation_id=recommendation_id,
            reason=reason,
        )
        self._emit(EventOut("operator", event, self.ts(), self.cell_of_cls[self.lever_class[name]]))
        out, self._out = self._out, []
        return out

    def snapshot(self) -> TrainRun:
        return copy.deepcopy(self)

    # -- lifecycle ------------------------------------------------------------------------------

    def _start(self) -> None:
        t, first = self.ts(0.0), self.cells[0]
        self._emit(
            EventOut(
                "batch",
                BatchStarted(
                    batch_id=self.spec.batch_id,
                    recipe=self.spec.recipe_id,
                    campaign=self.spec.campaign,
                    planned_levers=self.levers,
                    process=self.process,
                ),
                t,
                first,
            )
        )
        self.begin()

    def enter(self, cell: str) -> None:
        """The batch arrives on `cell` (the previous unit, if any, goes idle)."""
        if self.active is not None and self.active != cell:
            self.leave(self.active)
        self.active = cell
        self._emit(StateOut("batch", self.spec.batch_id, self.ts(), cell))

    def leave(self, cell: str) -> None:
        self.set_operation(cell, Operation.IDLE)
        self._emit(StateOut("batch", None, self.ts(), cell))
        if self.active == cell:
            self.active = None

    def set_operation(self, cell: str, new: Operation) -> None:
        t, old = self.ts(), self.op[cell]
        if old is new:
            return
        for phase in list(self.running[cell]):
            self.set_phase(cell, phase, PhaseState.COMPLETE, old)
        if old is not Operation.IDLE:
            self.op_ended_h[old] = self.t_h
        self.op[cell], self.op_started_h[cell] = new, self.t_h
        if new is not Operation.IDLE:
            self.op_started[new] = self.t_h
        self._emit(StateOut("operation", new.value, t, cell))
        self._emit(
            EventOut(
                "batch",
                OperationChanged(batch_id=self.spec.batch_id, previous=old, current=new),
                t,
                cell,
            )
        )
        for phase in self.phases.get(new, ()):
            self.set_phase(cell, phase, PhaseState.RUNNING, new)
        for fs in [f for f in self._armed if f.operation is new]:
            self._armed.remove(fs)
            cell_for = self.cell_of_cls.get(self.fault_class.get(fs.kind, ""), cell)
            self._add_fault(fs.kind, self.t_h + fs.after_h, dict(fs.params), cell_for)

    def set_phase(
        self, cell: str, phase: PhaseClass, state: PhaseState, operation: Operation | None = None
    ) -> None:
        operation = operation or self.op[cell]
        if state is PhaseState.COMPLETE:
            if phase not in self.running[cell]:
                return
            self.running[cell].pop(phase)
        else:
            self.running[cell][phase] = state
        t = self.ts()
        self._emit(StateOut(f"phase/{phase.value.lower()}", state.value, t, cell))
        self._emit(
            EventOut(
                "batch",
                PhaseChanged(
                    batch_id=self.spec.batch_id, operation=operation, phase=phase, state=state
                ),
                t,
                cell,
            )
        )

    def op_age(self, cell: str | None = None) -> float:
        """Hours since the current operation on `cell` (default: the active unit) began."""
        return self.t_h - self.op_started_h[cell or self.active or self.cells[0]]

    def lab(self, cell: str, values: dict[str, float], units: dict[str, str]) -> None:
        if self.quiet:
            return
        t = self.ts()
        for name, value in values.items():
            self._out.append(LabOut(name, float(value), units[name], t, cell))

    def produced(self, cell: str, quantity_kg: float) -> None:
        self._emit(
            EventOut(
                "material",
                MaterialProduced(
                    batch_id=self.spec.batch_id,
                    lot=self.spec.batch_id,
                    material=self.material,
                    quantity_kg=max(0.0, quantity_kg),
                ),
                self.ts(),
                cell,
            )
        )

    def consumed(self, cell: str, lot: LotUse) -> None:
        self._emit(
            EventOut(
                "material",
                MaterialConsumed(
                    batch_id=self.spec.batch_id,
                    lot=lot.lot,
                    material=lot.material,
                    quantity_kg=lot.quantity_kg,
                ),
                self.ts(),
                cell,
            )
        )

    def finish(self, disposition: str, reason: str | None = None) -> None:
        self._end(BatchStatus.COMPLETE, reason, disposition)

    def _end(self, status: BatchStatus, reason: str | None, disposition: str) -> None:
        cell = self.active or self.cells[-1]
        self.status, self.disposition = status, disposition
        if self.active is not None:
            self.leave(self.active)
        for f in self._faults:
            if f.onset_sent and not f.end_sent:
                self._send_label(f, end=True)
        self._emit(
            EventOut(
                "batch",
                BatchEnded(
                    batch_id=self.spec.batch_id,
                    status=status,
                    reason=reason,
                    disposition=disposition,
                ),
                self.ts(),
                cell,
            )
        )
        self.done = True

    # -- the integration step -----------------------------------------------------------------

    def active_faults(self, kind: FaultType | None = None) -> list[TrainFault]:
        return [f for f in self._faults if f.active(self.t_h) and (kind is None or f.kind is kind)]

    def _step(self) -> None:
        self.transition()
        if self.done:
            return
        t = self.t_h
        for f in self._faults:
            if f.active(t):
                if not f.onset_sent:
                    self._send_label(f, end=False)
                if f.kind is FaultType.STUCK_SENSOR:
                    bank = self.sensors[f.cell]
                    if f.stuck_var not in bank.stuck:
                        bank.stick(f.stuck_var)
            elif f.onset_sent and not f.end_sent and f.end_h is not None and t >= f.end_h:
                if f.kind is FaultType.STUCK_SENSOR:
                    self.sensors[f.cell].unstick(f.stuck_var)
                self._send_label(f, end=True)
        self.integrate(self.dt_h)
        self.k += 1
        if self.k % self.steps_per_publish == 0 and not self.done:
            self._publish_tick()

    def _publish_tick(self) -> None:
        if self.quiet:
            return
        t = self.ts()
        uncertain = {
            (f.cell, f.stuck_var) for f in self.active_faults(FaultType.STUCK_SENSOR) if f.noticed
        }
        for cell in self.cells:
            bank, on = self.sensors[cell], cell == self.active
            idle = self.idle[self.cls_of[cell]]
            for tag, var in self.raw_tags[cell]:
                value = self.read(cell, var) if on else bank.sample(var, idle[var], self.rng)
                q = Quality.UNCERTAIN if (cell, var) in uncertain else Quality.GOOD
                self._out.append(RawOut(tag, value, t, q, cell))
        if self.record_truth:
            self._out.append(
                TruthOut(t, self.t_h, self.truth(), self.operation, self.setpoints(), self.active)
            )

    # -- outputs ----------------------------------------------------------------------------------

    def _emit(self, msg: SimMessage) -> None:
        if not self.quiet:
            self._out.append(msg)

    def _add_fault(
        self, kind: FaultType, onset_h: float, params: dict[str, float | str], cell: str | None
    ) -> str:
        defaults = self.fault_defaults[kind]
        if cell is None:
            cell = self.cell_of_cls.get(self.fault_class.get(kind, ""), self.active)
        if cell not in self.cells:
            raise ValueError(f"{cell} is not part of this batch's train")
        fault = TrainFault(
            kind=kind,
            cell=cell,
            onset_h=onset_h,
            params=params,
            defaults=defaults,
            label_id=f"{self.spec.batch_id}-F{len(self._faults) + 1}",
        )
        duration = params.get("duration_h", defaults.get("duration_h"))
        if duration is not None:
            fault.end_h = onset_h + float(duration)
        if kind is FaultType.STUCK_SENSOR:
            if not fault.stuck_var:  # default: the unit's first analog variable
                fault.params = {**params, "tag": next(iter(self.noise[self.cls_of[cell]]))}
            if fault.stuck_var not in self.noise[self.cls_of[cell]]:
                raise ValueError(f"{cell} has no analog variable {fault.stuck_var!r}")
            fault.noticed = bool(self.rng.random() < fault.param("notice_probability"))
        self._faults.append(fault)
        return fault.label_id

    def _send_label(self, f: TrainFault, end: bool) -> None:
        if end:
            f.end_sent = True
        else:
            f.onset_sent = True
        label = FaultLabel(
            id=f.label_id,
            fault=f.kind,
            cell=f.cell,
            batch_id=self.spec.batch_id,
            onset=self.ts(f.onset_h),
            end=self.ts(min(self.t_h, f.end_h) if f.end_h is not None else self.t_h)
            if end
            else None,
            params=dict(f.params),
        )
        self._emit(LabelOut(label, self.ts(), f.cell))


STUCK_DEFAULTS: dict[str, float | str] = {"notice_probability": 0.3, "duration_h": 6.0}
