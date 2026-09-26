"""The aspirin API batch on the Tuas train: RX-201 reactor-crystallizer, then FD-202
filter-dryer (ADR-0019).

    RX-201  Charge (2 h) -> Reaction (dose 1 h + hold) -> Crystallization -> Transfer (1 h)
    FD-202  Filtration -> Washing -> Drying (to NIR moisture < 0.4%) -> Discharge (1 h)

Control loops act on measured values, as a DCS would: a drifting Ac2O flowmeter makes
the dosing loop stop at the right *metered* total with too little anhydride in the
vessel, and a dryer ends on its NIR reading.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, ClassVar

from common.models import FaultType, Operation, PhaseClass, PhaseState
from simulator.api import process as pr
from simulator.train import STUCK_DEFAULTS, TrainRun

RX, FD = "reactor", "filter_dryer"
MATERIAL = "Acetylsalicylic acid API"

CHARGE_H = 2.0
CHARGE_FILL_H = 0.5
HEAT_RATE = 60.0  # °C/h, heat-up ramp during Charge
DOSE_H = 1.0
DOSE_THROTTLE_C = 1.0  # dosing slows above SP + this, and stops at twice it
TRANSFER_H = 1.0
DISCHARGE_H = 1.0
MAX_FILTRATION_H = 14.0
MAX_DRYING_H = 30.0
WASH_L = 400.0
FILTER_PRESSURE = 1.5  # barg
VACUUM_SP = 50.0  # mbar a
DRY_END_MOISTURE = 0.4  # % (NIR), below the 0.5% LOD spec
DRY_CRITICAL = 5.0  # % moisture where drying leaves the constant-rate period

LAB_UNITS = {
    "conversion_ipc": "%",
    "assay": "%",
    "free_sa": "%",
    "related": "%",
    "lod": "%",
    "d50": "µm",
    "yield": "%",
}
SPEC = {"free_sa": 0.10, "related": 0.30, "lod": 0.5, "assay": 99.5}


@dataclass(slots=True)
class TempLoop:
    """TIC: jacket command = SP + PI correction, clamped to the utility's range, with
    conditional integration so it does not wind up while saturated."""

    kp: float = 20.0
    ki: float = 15.0
    lo: float = -15.0
    hi: float = 140.0
    integral: float = 0.0

    def update(self, sp: float, measured: float, dt: float) -> float:
        err = sp - measured
        out = sp + self.kp * err + self.integral + self.ki * err * dt
        if self.lo < out < self.hi or (out >= self.hi and err < 0) or (out <= self.lo and err > 0):
            self.integral += self.ki * err * dt
        return min(self.hi, max(self.lo, sp + self.kp * err + self.integral))


class ApiRun(TrainRun):
    process: ClassVar[str] = "api"
    material: ClassVar[str] = MATERIAL
    phases: ClassVar[dict[Operation, tuple[PhaseClass, ...]]] = {
        Operation.CHARGE: (PhaseClass.TEMP_CTRL, PhaseClass.AGIT_CTRL),
        Operation.REACTION: (PhaseClass.TEMP_CTRL, PhaseClass.AGIT_CTRL, PhaseClass.DOSE_ADD),
        Operation.CRYSTALLIZATION: (PhaseClass.TEMP_CTRL, PhaseClass.AGIT_CTRL),
        Operation.TRANSFER: (PhaseClass.AGIT_CTRL,),
        Operation.FILTRATION: (PhaseClass.FILTER_CTRL,),
        Operation.WASHING: (PhaseClass.FILTER_CTRL,),
        Operation.DRYING: (PhaseClass.TEMP_CTRL, PhaseClass.VAC_CTRL, PhaseClass.AGIT_CTRL),
        Operation.DISCHARGE: (PhaseClass.AGIT_CTRL,),
    }
    noise: ClassVar[dict[str, dict[str, float]]] = {
        RX: {
            "temp": 0.03, "jacket": 0.05, "agitation": 0.2, "power": 0.03, "dose_flow": 1.5,
            "pressure": 0.4, "level": 0.1, "conversion": 0.15, "chord": 0.8,
        },
        FD: {
            "filter_pressure": 0.004, "filtrate_flow": 0.15, "jacket": 0.05, "cake_temp": 0.05,
            "vacuum": 0.4, "agitation": 0.05, "moisture": 0.02,
        },
    }  # fmt: skip
    quantum: ClassVar[dict[str, dict[str, float]]] = {
        RX: {"dose_total": 0.1},
        FD: {"filtrate_total": 0.5},
    }
    idle: ClassVar[dict[str, dict[str, float]]] = {
        RX: {
            "temp": 25.0, "temp_sp": 25.0, "jacket": 25.0, "agitation": 0.0,
            "agitation_sp": 0.0, "power": 0.0, "dose_flow": 0.0, "dose_flow_sp": 0.0,
            "dose_total": 0.0, "pressure": 5.0, "level": 0.0, "conversion": 0.0, "chord": 2.0,
        },
        FD: {
            "filter_pressure": 0.0, "filter_pressure_sp": 0.0, "filtrate_flow": 0.0,
            "filtrate_total": 0.0, "jacket": 25.0, "jacket_sp": 25.0, "cake_temp": 25.0,
            "vacuum": 1013.0, "vacuum_sp": 1013.0, "agitation": 0.0, "moisture": 0.0,
        },
    }  # fmt: skip
    fault_class: ClassVar[dict[FaultType, str]] = {
        FaultType.JACKET_FOULING: RX,
        FaultType.DOSING_METER_DRIFT: RX,
        FaultType.AGITATOR_DEGRADATION: RX,
        FaultType.FILTER_BLINDING: FD,
        FaultType.VACUUM_LEAK: FD,
    }
    fault_defaults: ClassVar[dict[FaultType, dict[str, float | str]]] = {
        FaultType.JACKET_FOULING: {"floor": 0.06, "tau_h": 1.0},
        FaultType.DOSING_METER_DRIFT: {"rate": 0.6, "max": 0.3},  # fraction over-read
        FaultType.AGITATOR_DEGRADATION: {"floor": 0.45, "tau_h": 0.4},
        FaultType.FILTER_BLINDING: {"tau_h": 0.4},
        FaultType.VACUUM_LEAK: {"rate": 40.0, "max": 200.0, "duration_h": 8.0},  # mbar/h
        FaultType.STUCK_SENSOR: {"tag": "", **STUCK_DEFAULTS},
    }
    lever_class: ClassVar[dict[str, str]] = {
        "rxn_temp": RX, "ac2o_ratio": RX, "rxn_time": RX, "cool_rate": RX, "dry_temp": FD,
    }  # fmt: skip

    # -- setup ----------------------------------------------------------------------------------

    def init_process(self) -> None:
        self.params = pr.Params()
        self.rx = pr.Reactor()
        self.fd = pr.Dryer(impurity_kg={})
        self.loop = TempLoop()
        self.fd_loop = TempLoop(kp=3.0, ki=2.0, lo=15.0, hi=90.0)
        self.tsp = 25.0
        self.agit_sp = 0.0
        self.dose_sp = 0.0
        self.dose_target_kg = 0.0
        self.dose_end_h: float | None = None
        self.final_reached_h: float | None = None
        self.ratio_dosed = 0.0
        self.moisture0 = 0.0
        self.vacuum_sp = 1013.0
        self.dry_jacket_sp = 25.0
        tr = self.spec.traits
        self.k1_trait = tr.get("reactivity", 1.0)
        self.nuc_trait = tr.get("nucleation", 1.0)
        self.cloth_trait = tr.get("cloth", 1.0)

    def begin(self) -> None:
        self.enter(self.cells[0])
        self.set_operation(self.cells[0], Operation.CHARGE)

    # -- operations -----------------------------------------------------------------------------

    def transition(self) -> None:
        if self.active is None:
            return
        op, age, t = self.operation, self.op_age(), self.t_h
        rx_cell, fd_cell = self.cells
        lv = self.levers
        if op is Operation.CHARGE and age >= CHARGE_H - 1e-9:
            self.dose_target_kg = lv.ac2o_ratio * pr.N_SA0 * pr.MW_AC2O
            self.set_operation(rx_cell, Operation.REACTION)
        elif op is Operation.REACTION:
            if self.dose_end_h is None and self.rx.ac2o_metered_kg >= self.dose_target_kg:
                self.dose_end_h = t
                self.ratio_dosed = self.rx.ac2o_dosed_kg / pr.MW_AC2O / pr.N_SA0
                self.set_phase(rx_cell, PhaseClass.DOSE_ADD, PhaseState.COMPLETE)
            if self.dose_end_h is not None and t - self.dose_end_h >= lv.rxn_time - 1e-9:
                self.lab(
                    rx_cell,
                    {"conversion_ipc": self.rx.conversion + float(self.rng.normal(0, 0.05))},
                    LAB_UNITS,
                )
                pr.quench(self.rx, self.params)
                self.set_operation(rx_cell, Operation.CRYSTALLIZATION)
        elif op is Operation.CRYSTALLIZATION:
            if self.final_reached_h is not None and t - self.final_reached_h >= pr.AGEING_H:
                self.set_operation(rx_cell, Operation.TRANSFER)
        elif op is Operation.TRANSFER and age >= TRANSFER_H - 1e-9:
            self._to_dryer()
            self.enter(fd_cell)
            self.set_operation(fd_cell, Operation.FILTRATION)
        elif op is Operation.FILTRATION and (
            self.fd.filtered_l >= self.fd.liquor_l or age >= MAX_FILTRATION_H
        ):
            self.moisture0 = pr.initial_moisture(self.fd.d32)
            self.fd.moisture = self.moisture0
            self.set_operation(fd_cell, Operation.WASHING)
        elif op is Operation.WASHING and self.fd.wash_l >= WASH_L:
            self.set_operation(fd_cell, Operation.DRYING)
        elif op is Operation.DRYING and (
            (age >= 1.0 and self.sensors[fd_cell].control("moisture", self.fd.moisture)
             < DRY_END_MOISTURE)
            or age >= MAX_DRYING_H
        ):  # fmt: skip
            self.fd.drying_h = age
            self.set_operation(fd_cell, Operation.DISCHARGE)
        elif op is Operation.DISCHARGE and age >= DISCHARGE_H - 1e-9:
            self._release(fd_cell)

    def _to_dryer(self) -> None:
        """The slurry moves to the filter-dryer: crystals, liquor and what is dissolved."""
        rx, fd = self.rx, self.fd
        fd.liquor_l = rx.liquor_l
        fd.d32 = rx.d32
        fd.fines = pr.fines_loss(fd.d32)
        fd.crystal_kg = rx.crystal_kg * (1.0 - fd.fines)  # fines pass the cloth
        fd.losses_kg += rx.crystal_kg - fd.crystal_kg
        fd.impurity_kg = {
            "sa": rx.n_sa * pr.MW_SA,
            "assa": rx.n_assa * 318.28,
            "asan": rx.n_asan * 342.30,
            "asa": rx.n_asa * pr.MW_ASA,
        }

    def _release(self, cell: str) -> None:
        """Discharge done: the CoA, the lot into stock, the batch end (ADR-0020)."""
        fd, rng = self.fd, self.rng
        wash_loss = 3.0e-3 * WASH_L
        lumps = fd.crystal_kg * 0.004 * math.exp((self.levers.dry_temp - 55.0) / 4.0)
        product = max(fd.crystal_kg - wash_loss - fd.dust_kg - lumps, 1.0)
        imp = fd.impurity_kg or {}
        cake_liquor = self.moisture0 / (100.0 - self.moisture0) * fd.crystal_kg
        residual = cake_liquor / max(fd.liquor_l, 1.0) * (1.0 - 0.92 * fd.d32 / (fd.d32 + 25.0))
        free_sa = imp.get("sa", 0.0) * (0.008 + residual) + fd.free_sa_kg
        related = (imp.get("assa", 0.0) + imp.get("asan", 0.0)) * (0.03 + residual)
        self.coa = {
            "yield": 100.0 * product / pr.THEORETICAL_KG,
            "free_sa": 100.0 * free_sa / product,
            "related": 100.0 * related / product,
            "lod": fd.moisture,
            "d50": fd.d32,
            "product_kg": product,
        }
        self.coa["assay"] = 100.0 - self.coa["free_sa"] - self.coa["related"] - 0.2 * fd.moisture
        measured = {
            "assay": self.coa["assay"] + rng.normal(0, 0.05),
            "free_sa": max(0.0, self.coa["free_sa"] * (1 + rng.normal(0, 0.03))),
            "related": max(0.0, self.coa["related"] * (1 + rng.normal(0, 0.03))),
            "lod": max(0.0, fd.moisture + rng.normal(0, 0.02)),
            "d50": fd.d32 * (1 + rng.normal(0, 0.03)),
            "yield": self.coa["yield"] + rng.normal(0, 0.1),
        }
        self.lab(cell, measured, LAB_UNITS)
        ok = (
            measured["free_sa"] <= SPEC["free_sa"]
            and measured["related"] <= SPEC["related"]
            and measured["lod"] <= SPEC["lod"]
            and measured["assay"] >= SPEC["assay"]
        )
        self.produced(cell, product)
        self.finish("ACCEPTED" if ok else "REJECTED", None if ok else "CoA out of specification")

    # -- process ----------------------------------------------------------------------------------

    def _fault_factor(self, kind: FaultType) -> float:
        """Exponential decay towards a floor, 1.0 when inactive."""
        out = 1.0
        for f in self.active_faults(kind):
            floor, tau = f.param("floor"), f.param("tau_h")
            out *= floor + (1.0 - floor) * math.exp(-f.age(self.t_h) / tau)
        return out

    def _meter_drift(self) -> float:
        d = 0.0
        for f in self.active_faults(FaultType.DOSING_METER_DRIFT):
            d += min(f.param("max"), f.param("rate") * f.age(self.t_h))
        return d

    def integrate(self, dt: float) -> None:
        if self.active == self.cells[0]:
            self._reactor(dt)
        elif self.active == self.cells[1]:
            self._dryer(dt)

    def _reactor(self, dt: float) -> None:
        rx, p, lv, op, age = self.rx, self.params, self.levers, self.operation, self.op_age()
        sensors = self.sensors[self.cells[0]]
        if op is Operation.CHARGE:
            fill = min(1.0, age / CHARGE_FILL_H)
            rx.liquor_l = pr.ACOH_L * fill
            rx.solids_l = pr.SA_VOLUME_L * fill
            if fill >= 1.0 and rx.n_sa == 0.0:
                rx.n_sa = pr.N_SA0
            if age >= CHARGE_FILL_H:  # heat up on a ramp, not a step
                self.tsp = min(lv.rxn_temp, self.tsp + HEAT_RATE * dt)
            self.agit_sp = 120.0 if age >= 0.05 else 0.0
        elif op is Operation.REACTION:
            self.tsp = lv.rxn_temp
            self.agit_sp = 120.0
        elif op is Operation.CRYSTALLIZATION:
            self.agit_sp = 80.0
            if self.tsp > pr.FINAL_TEMP:
                self.tsp = max(pr.FINAL_TEMP, self.tsp - lv.cool_rate * dt)
            elif self.final_reached_h is None:
                self.final_reached_h = self.t_h
        elif op is Operation.TRANSFER:
            self.agit_sp = 60.0

        # dosing: the loop holds the *metered* flow at SP
        drift = self._meter_drift()
        dosing = op is Operation.REACTION and self.dose_end_h is None
        self.dose_sp = self.dose_target_kg / DOSE_H if dosing else 0.0
        if dosing:  # the exotherm interlock throttles dosing while the reactor runs hot
            hot = sensors.control("temp", rx.temp) - self.tsp - DOSE_THROTTLE_C
            self.dose_sp *= min(1.0, max(0.0, 1.0 - hot / DOSE_THROTTLE_C))
        rx.dose_flow = self.dose_sp / (1.0 + drift)
        if dosing:
            kg = rx.dose_flow * dt
            rx.n_ac2o += kg / pr.MW_AC2O
            rx.liquor_l += kg / pr.AC2O_DENSITY
            rx.ac2o_dosed_kg += kg
            rx.ac2o_metered_kg += kg * (1.0 + drift)

        eff = self._fault_factor(FaultType.AGITATOR_DEGRADATION)
        rx.agitation = self.agit_sp
        mass = rx.mass_kg
        fill = max(rx.volume_l, 1.0) / 2000.0
        rx.power = 4.0 * (rx.agitation / 120.0) ** 3 * fill**0.3 * eff
        ua = p.ua * (max(rx.agitation, 10.0) / 120.0) ** 0.5 * eff**0.3
        ua *= self._fault_factor(FaultType.JACKET_FOULING)
        ua *= min(1.0, rx.volume_l / 1500.0 + 0.2)  # wetted area grows with fill

        heat = 0.0
        if op in (Operation.REACTION, Operation.CRYSTALLIZATION, Operation.TRANSFER):
            heat = pr.react(rx, p, dt, mix=eff**0.7, k1_trait=self.k1_trait)
        if op in (Operation.CRYSTALLIZATION, Operation.TRANSFER):
            pr.crystallise(rx, p, dt, max(self.ratio_dosed, 1.0), self.nuc_trait)

        cmd = self.loop.update(self.tsp, sensors.control("temp", rx.temp), dt)
        rx.jacket += (cmd - rx.jacket) * min(1.0, dt / p.tau_jacket)
        cp = max(mass, 300.0) * p.cp
        q = ua * (rx.jacket - rx.temp) * 3600.0 * dt + heat
        q -= rx.dose_flow * dt * 2.0 * (rx.temp - 25.0)  # cold anhydride
        q -= 0.05 * (rx.temp - 25.0) * 3600.0 * dt  # losses, kW/K
        rx.temp += q / cp
        rx.max_temp = max(rx.max_temp, rx.temp)

    def _dryer(self, dt: float) -> None:
        fd, op, lv = self.fd, self.operation, self.levers
        cell = self.cells[1]
        for f in self.active_faults(FaultType.FILTER_BLINDING):
            fd.blind = 1.0 + f.age(self.t_h) / f.param("tau_h")
        if op in (Operation.FILTRATION, Operation.WASHING):
            fd.filter_pressure = FILTER_PRESSURE
            if op is Operation.FILTRATION:
                fd.filtrate_flow = pr.filtrate_flow(fd, fd.filter_pressure, self.cloth_trait)
                fd.filtered_l += fd.filtrate_flow * 60.0 * dt
            else:
                full = pr.Dryer(d32=fd.d32, filtered_l=fd.liquor_l, blind=fd.blind)
                fd.filtrate_flow = pr.filtrate_flow(full, fd.filter_pressure, self.cloth_trait)
                fd.wash_l += fd.filtrate_flow * 60.0 * dt
        else:
            fd.filter_pressure = 0.0
            fd.filtrate_flow = 0.0
        if op is Operation.DRYING:
            self.vacuum_sp = VACUUM_SP
            self.dry_jacket_sp = lv.dry_temp
            leak = 0.0
            for f in self.active_faults(FaultType.VACUUM_LEAK):
                leak = max(leak, min(f.param("max"), f.param("rate") * f.age(self.t_h)))
            pumped = fd.vacuum + (self.vacuum_sp - fd.vacuum) * min(1.0, dt / 0.08)
            fd.vacuum = max(pumped, self.vacuum_sp + leak)
            cmd = self.fd_loop.update(self.dry_jacket_sp, self.sensors[cell].control(
                "jacket", fd.jacket), dt)  # fmt: skip
            fd.jacket += (cmd - fd.jacket) * min(1.0, dt / 0.3)
            tb = pr.boiling_point(fd.vacuum)
            drive = max(fd.jacket - tb, 0.0)
            rate = 0.2 * drive * min(1.0, fd.moisture / DRY_CRITICAL)
            fd.moisture = max(0.02, fd.moisture - rate * dt)
            if fd.moisture > DRY_CRITICAL:
                target = min(tb, fd.jacket)
            else:
                target = tb + (fd.jacket - tb) * (1.0 - fd.moisture / DRY_CRITICAL)
            fd.cake_temp += (target - fd.cake_temp) * min(1.0, dt / 0.3)
            hydrolysis = 4.0e-4 * fd.moisture / 100.0 * fd.crystal_kg
            fd.free_sa_kg += (
                hydrolysis * math.exp(0.07 * (fd.cake_temp - 40.0)) * dt * (pr.MW_SA / pr.MW_ASA)
            )
            fd.dust_kg += 3.0e-4 * fd.crystal_kg * dt
        else:
            self.vacuum_sp = 1013.0
            fd.vacuum += (1013.0 - fd.vacuum) * min(1.0, dt / 0.1)
            if op is not Operation.DISCHARGE:
                self.dry_jacket_sp = 25.0
                fd.jacket += (25.0 - fd.jacket) * min(1.0, dt / 0.3)
                fd.cake_temp += (8.0 - fd.cake_temp) * min(1.0, dt / 0.5)

    # -- readings -------------------------------------------------------------------------------

    def read(self, cell: str, var: str) -> float:
        bank, rng = self.sensors[cell], self.rng
        if cell == self.cells[0]:
            rx = self.rx
            match var:
                case "temp":
                    return bank.sample("temp", rx.temp, rng)
                case "temp_sp":
                    return self.tsp
                case "jacket":
                    return bank.sample("jacket", rx.jacket, rng)
                case "agitation":
                    return bank.sample("agitation", rx.agitation, rng)
                case "agitation_sp":
                    return self.agit_sp
                case "power":
                    return bank.sample("power", rx.power, rng)
                case "dose_flow":
                    return bank.sample("dose_flow", rx.dose_flow * (1 + self._meter_drift()), rng)
                case "dose_flow_sp":
                    return self.dose_sp
                case "dose_total":
                    return bank.sample("dose_total", rx.ac2o_metered_kg, rng)
                case "pressure":
                    vap = 2.5 * math.exp(0.04 * (rx.temp - 25.0)) if rx.liquor_l > 0 else 0.0
                    return bank.sample("pressure", 20.0 + vap, rng)
                case "level":
                    return bank.sample("level", 100.0 * rx.volume_l / pr.WORKING_VOLUME_L, rng)
                case "conversion":
                    return bank.sample("conversion", rx.conversion, rng)
                case "chord":
                    chord = 0.55 * rx.d32 if rx.m0 > 0 else 2.0
                    return bank.sample("chord", chord, rng)
        else:
            fd = self.fd
            match var:
                case "filter_pressure":
                    return bank.sample("filter_pressure", fd.filter_pressure, rng)
                case "filter_pressure_sp":
                    return fd.filter_pressure
                case "filtrate_flow":
                    return bank.sample("filtrate_flow", fd.filtrate_flow, rng)
                case "filtrate_total":
                    return bank.sample("filtrate_total", fd.filtered_l + fd.wash_l, rng)
                case "jacket":
                    return bank.sample("jacket", fd.jacket, rng)
                case "jacket_sp":
                    return self.dry_jacket_sp
                case "cake_temp":
                    return bank.sample("cake_temp", fd.cake_temp, rng)
                case "vacuum":
                    return bank.sample("vacuum", fd.vacuum, rng)
                case "vacuum_sp":
                    return self.vacuum_sp
                case "agitation":
                    on = self.operation in (Operation.DRYING, Operation.DISCHARGE)
                    return bank.sample("agitation", 5.0 if on else 0.0, rng)
                case "moisture":
                    return bank.sample("moisture", fd.moisture, rng)
        raise KeyError(var)

    # -- context for the harness, truth and operator ----------------------------------------------

    def truth(self) -> Any:
        return {"rx": copy.copy(self.rx), "fd": copy.copy(self.fd)}

    def setpoints(self) -> dict[str, float]:
        return {"temperature": self.tsp, "vacuum": self.vacuum_sp}

    def lever_open(self, name: str) -> bool:
        op = self.operation
        before_reaction = op is Operation.CHARGE
        reacting = op in (Operation.CHARGE, Operation.REACTION)
        match name:
            case "ac2o_ratio":
                return before_reaction
            case "rxn_temp" | "rxn_time":
                return reacting
            case "cool_rate":
                cooling = op is Operation.CRYSTALLIZATION and self.final_reached_h is None
                return reacting or cooling
            case "dry_temp":
                return self.active is not None and op is not Operation.DISCHARGE
        return False

    def target(self) -> float:
        return self.coa.get("yield", 0.0)
