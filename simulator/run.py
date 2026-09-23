"""Live simulator service: drives one `BatchRun` per bioreactor on the simulated clock
and obeys `_sim/cmd/*` (ADR-0012).

    python -m simulator.run

Publishes raw DCS tags to edge/raw, and lab, state and events straight into the UNS
(as the LIMS/MES stand-in, ADR-0005), plus `_sim/clock` and `_sim/faults`.
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import yaml
from pydantic import BaseModel

from common import models as m
from common import uns
from common.models import Operation, PhaseState, Src
from common.mqtt import UnsClient, connect
from common.settings import Settings, get_settings
from common.uns import UnitPath
from simulator import batches as b
from simulator import recipes
from simulator.clock import SimClock
from simulator.engine import DCS_TAGS_FILE, BatchRun, load_dcs_tags
from simulator.sensors import SensorBank
from simulator.wire import to_wire

SERVICE = "simulator"
log = logging.getLogger(SERVICE)

TICK_WALL_S = 0.05
CLOCK_PUBLISH_WALL_S = 1.0
RETAINED_WAIT_S = 1.5

# What an empty, idle vessel's instruments read between batches.
IDLE_VALUES = {
    "temp": 22.0, "temp_sp": 22.0, "ph": 7.4, "ph_sp": 7.0, "do": 100.0, "do_sp": 40.0,
    "agitation": 0.0, "air": 0.0, "o2": 0.0, "co2": 0.0, "feed_total": 0.0,
    "base_total": 0.0, "pressure": 0.0, "weight": 0.0, "spare_rtd": 22.0,
}  # fmt: skip


@dataclass
class Unit:
    path: UnitPath
    run: BatchRun | None = None
    next_idle_sample: datetime | None = None
    idle_sensors: SensorBank = field(default_factory=SensorBank)


@dataclass
class Retained:
    """What the broker remembers from a previous simulator life."""

    latest_ts: list[datetime] = field(default_factory=list)
    last_batch_seq: int | None = None
    batch_of: dict[str, str | None] = field(default_factory=dict)
    operation_of: dict[str, Operation] = field(default_factory=dict)
    phases_of: dict[str, dict[str, PhaseState]] = field(default_factory=dict)

    def on_message(self, topic: str, payload: BaseModel | None) -> None:
        if payload is None:
            return
        if isinstance(payload, m.ClockStatusPayload):
            self.latest_ts.append(payload.v.sim_time)
            self.last_batch_seq = payload.v.last_batch_seq
            return
        ts = getattr(payload, "ts", None)
        if ts is not None:
            self.latest_ts.append(ts)
        p = uns.parse(topic)
        if p.unit is None:
            return
        cell = p.unit.cell
        if isinstance(payload, m.BatchStatePayload):
            self.batch_of[cell] = payload.v
        elif isinstance(payload, m.OperationPayload):
            self.operation_of[cell] = payload.v
        elif isinstance(payload, m.PhaseStatePayload):
            self.phases_of.setdefault(cell, {})[p.name.removeprefix("phase/")] = payload.v


class Simulator:
    def __init__(self, client: UnsClient, settings: Settings, retained: Retained) -> None:
        self.client = client
        self.settings = settings
        self.clock = SimClock.starting_at(m.now_utc(), retained.latest_ts, settings.sim_speed)
        self.next_seq = max(
            settings.backfill_batches,
            (retained.last_batch_seq + 1) if retained.last_batch_seq is not None else 0,
        )
        cells = yaml.safe_load(DCS_TAGS_FILE.read_text())["units"]
        self.units = {
            c: Unit(UnitPath(settings.site, settings.area, settings.line, c)) for c in cells
        }
        self.tags = load_dcs_tags()
        self.commands: queue.Queue[BaseModel] = queue.Queue()
        self.rng = np.random.default_rng(settings.seed)
        self._close_orphans(retained)

    # -- startup --------------------------------------------------------------------------

    def _close_orphans(self, retained: Retained) -> None:
        """A batch the previous simulator life left running cannot be resumed: close it as
        ABORTED so the UNS and graph do not show a phantom batch."""
        t = self.clock.now
        for cell, unit in self.units.items():
            batch_id = retained.batch_of.get(cell)
            op = retained.operation_of.get(cell, Operation.IDLE)
            if batch_id and op is not Operation.IDLE:
                log.warning("%s: closing orphaned batch %s", cell, batch_id)
                for phase, state in retained.phases_of.get(cell, {}).items():
                    if state is not PhaseState.COMPLETE:
                        self._pub(
                            uns.state_phase(unit.path, phase),
                            m.PhaseStatePayload(
                                v=PhaseState.COMPLETE, ts=t, unit=None, batch=batch_id, src=Src.SIM
                            ),
                        )
                self._pub(
                    uns.events(unit.path, "batch"),
                    m.BatchEventPayload(
                        v=m.BatchEnded(
                            batch_id=batch_id,
                            status=m.BatchStatus.ABORTED,
                            reason="simulator restarted",
                        ),
                        ts=t,
                        unit=None,
                        batch=batch_id,
                        src=Src.SIM,
                    ),
                )
            if op is not Operation.IDLE or cell not in retained.operation_of:
                self._pub(
                    uns.state_operation(unit.path),
                    m.OperationPayload(v=Operation.IDLE, ts=t, unit=None, batch=None, src=Src.SIM),
                )
            if retained.batch_of.get(cell) is not None or cell not in retained.batch_of:
                self._pub(
                    uns.state_batch(unit.path),
                    m.BatchStatePayload(v=None, ts=t, unit=None, batch=None, src=Src.SIM),
                )

    # -- main loop ------------------------------------------------------------------------

    def serve(self, stop: threading.Event) -> None:
        self.client.subscribe(uns.SUB_SIM_CMD, lambda _t, p: p is not None and self.commands.put(p))
        last = time.monotonic()
        last_clock_pub = 0.0
        while not stop.is_set():
            time.sleep(TICK_WALL_S)
            self._drain_commands()
            now = time.monotonic()
            self._tick(now - last)
            last = now
            if now - last_clock_pub >= CLOCK_PUBLISH_WALL_S:
                self._publish_clock()
                last_clock_pub = now

    def _tick(self, wall_elapsed: float) -> None:
        if not self.clock.advance(wall_elapsed):
            return
        target = self.clock.now
        armed = self.units.get(self.clock.run_to_cell or "")
        if armed and armed.run and self.clock.run_to_day is not None:
            target = min(target, armed.run.ts(b.t_of_day(self.clock.run_to_day)))
        self.clock.now = target
        for unit in self.units.values():
            self._advance(unit, target)
        if (
            armed
            and armed.run
            and self.clock.check_run_to_day(armed.path.cell, armed.run.batch_day)
        ):
            log.info("%s reached the requested day; paused", armed.path.cell)
            self._publish_clock()

    def _advance(self, unit: Unit, until: datetime) -> None:
        run = unit.run
        if run is not None:
            for msg in run.advance((until - run.spec.start).total_seconds() / 3600.0):
                self._emit(unit, run.spec.batch_id, msg)
            if run.done:
                unit.run = None
                unit.next_idle_sample = run.ts() + timedelta(seconds=self.settings.publish_period_s)
            return
        period = timedelta(seconds=self.settings.publish_period_s)
        if unit.next_idle_sample is None:
            unit.next_idle_sample = until
        while unit.next_idle_sample <= until:
            t = unit.next_idle_sample
            for suffix, var in self.tags.items():
                value = unit.idle_sensors.sample(var, IDLE_VALUES[var], self.rng)
                tag = f"{unit.path.device}.{suffix}"
                self._pub(
                    uns.edge_raw(unit.path.device, tag), m.RawSample(tag=tag, value=value, t=t)
                )
            unit.next_idle_sample = t + period

    # -- commands -------------------------------------------------------------------------

    def _drain_commands(self) -> None:
        while True:
            try:
                payload = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle(payload.v)
            except (ValueError, KeyError, LookupError) as exc:
                log.warning("command rejected: %s (%s)", payload.v, exc)

    def _handle(self, cmd: BaseModel) -> None:
        match cmd:
            case m.BatchCommand(action="start"):
                self._start_batch(cmd)
            case m.BatchCommand(action="abort"):
                unit = self._running(cmd.cell)
                for msg in unit.run.abort("operator abort"):
                    self._emit(unit, unit.run.spec.batch_id, msg)
            case m.FaultCommand(action="inject"):
                unit = self._running(cmd.cell)
                label = unit.run.inject(cmd.fault, **cmd.params)
                log.info("%s: injected %s (%s)", cmd.cell, cmd.fault, label)
            case m.FaultCommand(action="clear"):
                self._running(cmd.cell).run.clear(cmd.fault)
            case m.ClockCommand(action="pause"):
                self.clock.paused = True
            case m.ClockCommand(action="resume"):
                self.clock.paused = False
            case m.ClockCommand(action="speed"):
                self.clock.set_speed(cmd.speed or self.settings.sim_speed)
            case m.ClockCommand(action="run_to_day"):
                if cmd.cell is None or cmd.day is None:
                    raise ValueError("run_to_day needs a cell and a day")
                self._running(cmd.cell)
                self.clock.arm_run_to_day(cmd.cell, cmd.day)
            case m.SetpointCommand():
                unit = self._running(cmd.cell)
                msgs = unit.run.change_lever(
                    cmd.parameter, cmd.value, cmd.operator, cmd.recommendation_id, cmd.reason
                )
                for msg in msgs:
                    self._emit(unit, unit.run.spec.batch_id, msg)
        self._publish_clock()

    def _running(self, cell: str) -> Unit:
        unit = self.units[cell]
        if unit.run is None:
            raise ValueError(f"{cell} has no running batch")
        return unit

    def _start_batch(self, cmd: m.BatchCommand) -> None:
        unit = self.units[cmd.cell]
        if unit.run is not None:
            raise ValueError(f"{cmd.cell} is already running {unit.run.spec.batch_id}")
        start = self.clock.now
        recipe = recipes.get(cmd.recipe) if cmd.recipe else recipes.current(start.date())
        levers = recipe.clip(cmd.levers) if cmd.levers else recipe.nominal
        seq, self.next_seq = self.next_seq, self.next_seq + 1
        rng = np.random.default_rng([self.settings.seed, seq])
        spec = b.BatchSpec(
            batch_id=b.batch_id(start, seq),
            cell=cmd.cell,
            start=start,
            recipe_id=recipe.id,
            campaign=cmd.campaign,
            levers=levers,
            seed=int(rng.integers(2**31)),
            traits=b.draw_traits(rng),
        )
        unit.run = BatchRun(
            spec,
            dt_s=self.settings.integration_step_s,
            publish_period_s=self.settings.publish_period_s,
            tags=self.tags,
        )
        unit.idle_sensors = SensorBank()
        for msg in unit.run.advance(0.0):
            self._emit(unit, spec.batch_id, msg)
        log.info("%s: started %s (recipe %s)", cmd.cell, spec.batch_id, recipe.id)

    # -- publishing -----------------------------------------------------------------------

    def _emit(self, unit: Unit, batch_id: str, msg: object) -> None:
        wire = to_wire(unit.path, batch_id, msg)
        if wire is not None:
            self._pub(*wire)

    def _pub(self, topic: str, payload: BaseModel) -> None:
        self.client.publish(topic, payload)

    def _publish_clock(self) -> None:
        c = self.clock
        self._pub(uns.sim_clock(), m.ClockStatusPayload(
            v=m.ClockStatus(sim_time=c.now, speed=c.speed, paused=c.paused,
                            run_to_day=c.run_to_day, run_to_cell=c.run_to_cell,
                            last_batch_seq=self.next_seq - 1),
            ts=c.now, unit=None, batch=None, src=Src.SIM))  # fmt: skip


def read_retained(client: UnsClient) -> Retained:
    retained = Retained()
    client.subscribe(uns.sub_class(uns.TopicClass.STATE), retained.on_message)
    client.subscribe(uns.sim_clock(), retained.on_message)
    time.sleep(RETAINED_WAIT_S)
    return retained


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    client = connect(SERVICE, Src.SIM, settings)
    sim = Simulator(client, settings, read_retained(client))
    log.info(
        "clock at %s, speed %sx, next batch seq %d", sim.clock.now, sim.clock.speed, sim.next_seq
    )
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    sim.serve(stop)
    client.close()


if __name__ == "__main__":
    main()
