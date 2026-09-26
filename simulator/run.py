"""Live simulator service: drives the batches of every site on the simulated clock and
obeys `_sim/cmd/*` (ADR-0012, ADR-0018).

    python -m simulator.run

A bioreactor batch runs on one unit; an API or OSD batch holds every unit of its train
until it ends. Publishes raw DCS tags to edge/raw, and lab, state and events straight
into the UNS (as the LIMS/MES stand-in, ADR-0005), plus `_sim/clock` and `_sim/faults`.
It also keeps Freiburg's API stock on `_sim/inventory` (ADR-0020): Tuas lots go in as
they are released, tablet batches draw on them first in, first out.
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
from pydantic import BaseModel

from common import models as m
from common import uns
from common.models import Operation, PhaseState, Src
from common.mqtt import UnsClient, connect
from common.plant import get_plant
from common.settings import Settings, get_settings
from common.uns import UnitPath
from simulator import batches as b
from simulator import recipes
from simulator import train_campaign as tc
from simulator.api.engine import ApiRun
from simulator.clock import SimClock
from simulator.engine import BatchRun
from simulator.messages import load_dcs_tags
from simulator.osd.engine import OsdRun
from simulator.processes import TRAIN_RUNS, Run
from simulator.sensors import SensorBank
from simulator.train import LotUse, TrainRun, TrainSpec
from simulator.wire import to_wire

SERVICE = "simulator"
log = logging.getLogger(SERVICE)

TICK_WALL_S = 0.05
CLOCK_PUBLISH_WALL_S = 1.0
RETAINED_WAIT_S = 1.5

# What an empty, idle unit's instruments read between batches, per equipment class.
BIO_IDLE_VALUES = {
    "temp": 22.0, "temp_sp": 22.0, "ph": 7.4, "ph_sp": 7.0, "do": 100.0, "do_sp": 40.0,
    "agitation": 0.0, "air": 0.0, "o2": 0.0, "co2": 0.0, "feed_total": 0.0,
    "base_total": 0.0, "pressure": 0.0, "weight": 0.0, "spare_rtd": 22.0,
}  # fmt: skip
IDLE_VALUES: dict[str, dict[str, float]] = {
    "bioreactor": BIO_IDLE_VALUES,
    **ApiRun.idle,
    **OsdRun.idle,
}
IDLE_NOISE: dict[str, dict[str, float]] = {**ApiRun.noise, **OsdRun.noise}
IDLE_QUANTUM: dict[str, dict[str, float]] = {**ApiRun.quantum, **OsdRun.quantum}


@dataclass
class Unit:
    path: UnitPath
    cls: str
    process: str
    run: Run | None = None  # a train batch is held by every unit of its train
    next_idle_sample: datetime | None = None
    idle_sensors: SensorBank = field(default_factory=SensorBank)

    def reset_idle(self) -> None:
        self.idle_sensors = (
            SensorBank()
            if self.cls == "bioreactor"
            else SensorBank(noise=IDLE_NOISE[self.cls], quantum=IDLE_QUANTUM[self.cls])
        )


@dataclass
class Retained:
    """What the broker remembers from a previous simulator life."""

    latest_ts: list[datetime] = field(default_factory=list)
    last_batch_seq: int | None = None
    batch_of: dict[str, str | None] = field(default_factory=dict)
    operation_of: dict[str, Operation] = field(default_factory=dict)
    phases_of: dict[str, dict[str, PhaseState]] = field(default_factory=dict)
    inventory: m.Inventory | None = None
    opening_stock: m.Inventory | None = None

    def on_message(self, topic: str, payload: BaseModel | None) -> None:
        if payload is None:
            return
        if isinstance(payload, m.ClockStatusPayload):
            self.latest_ts.append(payload.v.sim_time)
            self.last_batch_seq = payload.v.last_batch_seq
            return
        if isinstance(payload, m.InventoryPayload):
            if topic == uns.sim_inventory():
                self.inventory = payload.v
            else:
                self.opening_stock = payload.v
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
        self.plant = get_plant()
        self.clock = SimClock.starting_at(m.now_utc(), retained.latest_ts, settings.sim_speed)
        self.next_seq = max(
            settings.backfill_total,
            (retained.last_batch_seq + 1) if retained.last_batch_seq is not None else 0,
        )
        self.units: dict[str, Unit] = {}
        for cell, u in self.plant.units.items():
            unit = Unit(u.path, u.cls, u.process)
            unit.reset_idle()
            self.units[cell] = unit
        self.tags = {c: load_dcs_tags(u.cls) for c, u in self.units.items()}
        self.commands: queue.Queue[BaseModel] = queue.Queue()
        self.rng = np.random.default_rng(settings.seed)
        self.inventory: m.Inventory | None = retained.inventory
        if self.inventory is None and retained.opening_stock is not None:
            self._adopt(retained.opening_stock)
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
        self.client.subscribe(
            uns.sim_opening_stock(), lambda _t, p: p is not None and self.commands.put(p)
        )
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

    def _runs(self) -> list[Run]:
        out: list[Run] = []
        for unit in self.units.values():
            if unit.run is not None and all(unit.run is not r for r in out):
                out.append(unit.run)
        return out

    def _tick(self, wall_elapsed: float) -> None:
        if not self.clock.advance(wall_elapsed):
            return
        target = self.clock.now
        armed = self.units.get(self.clock.run_to_cell or "")
        if armed and armed.run and self.clock.run_to_day is not None:
            target = min(target, armed.run.ts(armed.run.t_of_day(self.clock.run_to_day)))
        self.clock.now = target
        for run in self._runs():
            self._advance(run, target)
        for unit in self.units.values():
            if unit.run is None:
                self._idle(unit, target)
        if (
            armed
            and armed.run
            and self.clock.check_run_to_day(armed.path.cell, armed.run.batch_day)
        ):
            log.info("%s reached the requested day; paused", armed.path.cell)
            self._publish_clock()

    def _advance(self, run: Run, until: datetime) -> None:
        start_unit = self.plant.path(run.spec.cell)
        for msg in run.advance((until - run.spec.start).total_seconds() / 3600.0):
            self._emit(start_unit, run.spec.batch_id, msg)
        if run.done:
            nxt = run.ts() + timedelta(seconds=self.settings.publish_period_s)
            for unit in self.units.values():
                if unit.run is run:
                    unit.run, unit.next_idle_sample = None, nxt
            if isinstance(run, ApiRun):
                self._stock_lot(run)

    def _idle(self, unit: Unit, until: datetime) -> None:
        period = timedelta(seconds=self.settings.publish_period_s)
        if unit.next_idle_sample is None:
            unit.next_idle_sample = until
        idle = IDLE_VALUES[unit.cls]
        while unit.next_idle_sample <= until:
            t = unit.next_idle_sample
            for suffix, var in self.tags[unit.path.cell].items():
                value = unit.idle_sensors.sample(var, idle[var], self.rng)
                tag = f"{unit.path.device}.{suffix}"
                self._pub(
                    uns.edge_raw(unit.path.device, tag), m.RawSample(tag=tag, value=value, t=t)
                )
            unit.next_idle_sample = t + period

    # -- inventory (ADR-0020) -------------------------------------------------------------------

    def _adopt(self, stock: m.Inventory) -> None:
        log.info("adopting the opening stock: %d lots, %.0f kg", len(stock.lots),
                 sum(lot.quantity_kg for lot in stock.lots))  # fmt: skip
        self.inventory = stock
        self._publish_inventory()

    def _stock_lot(self, run: ApiRun) -> None:
        """A released Tuas lot goes to Freiburg (the demo skips QC release and shipping)."""
        if run.disposition != "ACCEPTED":
            return
        inv = self.inventory or m.Inventory(site="freiburg")
        lot = m.LotStock(
            lot=run.spec.batch_id,
            material=run.material,
            quantity_kg=round(run.coa["product_kg"], 3),
            released=run.ts(),
            properties={
                "d50_um": run.coa["d50"],
                "free_sa_pct": run.coa["free_sa"],
                "assay_pct": run.coa["assay"],
            },
        )
        self.inventory = inv.model_copy(update={"lots": [*inv.lots, lot]})
        self._publish_inventory()
        log.info("%s: %.0f kg of API released to Freiburg", lot.lot, lot.quantity_kg)

    def _draw_api(self, now: datetime) -> tuple[LotUse, ...]:
        """200 kg of released API, first in, first out, or ValueError if short."""
        lots = sorted(
            (lot for lot in (self.inventory.lots if self.inventory else []) if lot.released <= now),
            key=lambda lot: lot.released,
        )
        have = sum(lot.quantity_kg for lot in lots)
        if have < tc.API_PER_BATCH_KG - 1e-6:
            raise ValueError(f"Freiburg holds only {have:.0f} kg of released API")
        need, uses, left = tc.API_PER_BATCH_KG, [], {}
        for lot in lots:
            take = min(lot.quantity_kg, need)
            if take > 0:
                uses.append(LotUse(lot.lot, lot.material, round(take, 3), dict(lot.properties)))
                need -= take
            left[lot.lot] = lot.quantity_kg - take
        assert self.inventory is not None
        self.inventory = self.inventory.model_copy(
            update={
                "lots": [
                    lot.model_copy(
                        update={"quantity_kg": round(left.get(lot.lot, lot.quantity_kg), 3)}
                    )
                    for lot in self.inventory.lots
                    if left.get(lot.lot, lot.quantity_kg) > 1e-6
                ]
            }
        )
        return tuple(uses)

    # -- commands -------------------------------------------------------------------------

    def _drain_commands(self) -> None:
        while True:
            try:
                payload = self.commands.get_nowait()
            except queue.Empty:
                return
            if isinstance(payload, m.InventoryPayload):  # _sim/opening_stock
                if self.inventory is None:
                    self._adopt(payload.v)
                continue
            try:
                self._handle(payload.v)
            except (ValueError, KeyError, LookupError) as exc:
                log.warning("command rejected: %s (%s)", payload.v, exc)

    def _handle(self, cmd: BaseModel) -> None:
        match cmd:
            case m.BatchCommand(action="start"):
                self._start_batch(cmd)
            case m.BatchCommand(action="abort"):
                run = self._running(cmd.cell).run
                for msg in run.abort("operator abort"):
                    self._emit(self.plant.path(run.spec.cell), run.spec.batch_id, msg)
            case m.FaultCommand(action="inject"):
                run = self._running(cmd.cell).run
                if isinstance(run, TrainRun):
                    cell = None if cmd.fault in run.fault_class else cmd.cell
                    label = run.inject(cmd.fault, cell=cell, **cmd.params)
                else:
                    label = run.inject(cmd.fault, **cmd.params)
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
                run = self._running(cmd.cell).run
                msgs = run.change_lever(
                    cmd.parameter, cmd.value, cmd.operator, cmd.recommendation_id, cmd.reason
                )
                for msg in msgs:
                    self._emit(self.plant.path(run.spec.cell), run.spec.batch_id, msg)
        self._publish_clock()

    def _running(self, cell: str) -> Unit:
        unit = self.units[cell]
        if unit.run is None:
            raise ValueError(f"{cell} has no running batch")
        return unit

    def _start_batch(self, cmd: m.BatchCommand) -> None:
        unit = self.units[cmd.cell]
        train = self.plant.train_of(cmd.cell)
        if train[0] != cmd.cell:
            raise ValueError(f"a {unit.process} batch starts on {train[0]}")
        busy = [c for c in train if self.units[c].run is not None]
        if busy:
            raise ValueError(
                f"{', '.join(busy)} still running {self.units[busy[0]].run.spec.batch_id}"
            )
        start = self.clock.now
        recipe = (
            recipes.get(cmd.recipe) if cmd.recipe else recipes.current(start.date(), unit.process)
        )
        if recipe.process != unit.process:
            raise ValueError(f"recipe {recipe.id} is for {recipe.process}, not {unit.process}")
        levers = (
            recipe.clip(recipe.levers_model.model_validate(cmd.levers.model_dump()))
            if cmd.levers
            else recipe.nominal
        )
        seq = self.next_seq
        rng = np.random.default_rng([self.settings.seed, seq])
        batch_id = b.batch_id(start, seq)
        run: Run
        if unit.process in TRAIN_RUNS:
            lots = self._draw_api(start) if unit.process == "osd" else ()
            spec = TrainSpec(
                batch_id=batch_id,
                cell=cmd.cell,
                start=start,
                recipe_id=recipe.id,
                campaign=cmd.campaign,
                levers=levers,
                seed=int(rng.integers(2**31)),
                traits=tc.draw_traits(unit.process, rng),
                lots=lots,
            )
            run = TRAIN_RUNS[unit.process](
                spec,
                dt_s=self.settings.integration_step_s,
                publish_period_s=self.settings.publish_period_s,
                plant=self.plant,
            )
            if lots:
                self._publish_inventory()
        else:
            spec = b.BatchSpec(
                batch_id=batch_id,
                cell=cmd.cell,
                start=start,
                recipe_id=recipe.id,
                campaign=cmd.campaign,
                levers=levers,
                seed=int(rng.integers(2**31)),
                traits=b.draw_traits(rng),
            )
            run = BatchRun(
                spec,
                dt_s=self.settings.integration_step_s,
                publish_period_s=self.settings.publish_period_s,
                tags=self.tags[cmd.cell],
            )
        self.next_seq += 1
        for cell in train:
            self.units[cell].run = run
            self.units[cell].reset_idle()
        for msg in run.advance(0.0):
            self._emit(self.plant.path(cmd.cell), spec.batch_id, msg)
        log.info("%s: started %s (recipe %s)", cmd.cell, spec.batch_id, recipe.id)

    # -- publishing -----------------------------------------------------------------------

    def _emit(self, unit: UnitPath, batch_id: str, msg: object) -> None:
        wire = to_wire(unit, batch_id, msg)
        if wire is not None:
            self._pub(*wire)

    def _pub(self, topic: str, payload: BaseModel) -> None:
        self.client.publish(topic, payload)

    def _publish_inventory(self) -> None:
        if self.inventory is not None:
            payload = m.InventoryPayload(
                v=self.inventory, ts=self.clock.now, unit="kg", batch=None, src=Src.SIM
            )
            self._pub(uns.sim_inventory(), payload)

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
    client.subscribe(uns.sim_inventory(), retained.on_message)
    client.subscribe(uns.sim_opening_stock(), retained.on_message)
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
