"""The aspirin 500 mg tablet batch on the Freiburg line: BL-301 bin blender, RC-302
roller compactor, TP-303 rotary tablet press (ADR-0019).

    BL-301  Charge (0.5 h, consumes API lots) -> Blending (to NIR RSD < 3%) ->
            Lubrication (the recipe's minutes) -> Discharge (0.25 h)
    RC-302  Compaction (60 kg/h until the blend is used)
    TP-303  Compression (until the granules are used; the tablets go into stock)

The press's weight control holds the *measured* compression force at SP, and the
compactor's force loop its *measured* roll force: a drifting transducer shows a
perfect PV while the ribbons go soft.
"""

from __future__ import annotations

import copy
import math
from typing import Any, ClassVar

from common.models import FaultType, Operation, PhaseClass
from simulator.osd import process as pr
from simulator.train import STUCK_DEFAULTS, LotUse, TrainRun

BL, RC, TP = "blender", "roller_compactor", "tablet_press"
MATERIAL = "Acetylsalicylic acid 500 mg tablets"
API_MATERIAL = "Acetylsalicylic acid API"

CHARGE_H = 0.5
DISCHARGE_H = 0.25
BLEND_RPM = 12.0
MIN_REVOLUTIONS, MAX_REVOLUTIONS = 150.0, 480.0
BLEND_END_RSD = 3.0
COMPACTION_KG_H = 60.0
ROLL_SPEED = 4.0
MILL_SPEED = 800.0
GAP_SP = 2.5
BLEND_LOSS, COMPACTION_LOSS, PRESS_SETUP_LOSS = 0.003, 0.01, 0.005
ROOM_RH = 30.0
DEFAULT_LOT = LotUse(
    "B2026-0001", API_MATERIAL, pr.API_KG, {"d50_um": 215.0, "free_sa_pct": 0.05, "assay_pct": 99.7}
)

LAB_UNITS = {
    "blend_uniformity": "%",
    "granule_d50": "µm",
    "bulk_density": "g/mL",
    "assay": "%",
    "av": "",
    "dissolution": "%",
    "hardness": "N",
    "friability": "%",
    "free_sa": "%",
    "yield": "%",
}
SPEC = {"dissolution": 80.0, "av": 15.0, "free_sa": 0.3, "friability": 1.0}


class OsdRun(TrainRun):
    process: ClassVar[str] = "osd"
    material: ClassVar[str] = MATERIAL
    phases: ClassVar[dict[Operation, tuple[PhaseClass, ...]]] = {
        Operation.BLENDING: (PhaseClass.BLEND_CTRL,),
        Operation.LUBRICATION: (PhaseClass.BLEND_CTRL,),
        Operation.COMPACTION: (PhaseClass.COMPACT_CTRL, PhaseClass.MILL_CTRL),
        Operation.COMPRESSION: (PhaseClass.TABLET_CTRL, PhaseClass.FEED_CTRL),
    }
    noise: ClassVar[dict[str, dict[str, float]]] = {
        BL: {"speed": 0.05, "blend_rsd": 0.15, "room_rh": 0.3},
        RC: {
            "roll_force": 0.04, "roll_gap": 0.015, "roll_speed": 0.02, "screw_speed": 0.2,
            "mill_speed": 3.0, "ribbon_density": 0.004, "room_rh": 0.3,
        },
        TP: {
            "comp_force": 0.15, "precomp_force": 0.05, "turret_speed": 0.1, "feed_frame": 0.1,
            "ejection_force": 4.0, "tablet_weight": 1.2, "weight_rsd": 0.06, "hardness": 2.5,
            "room_rh": 0.3,
        },
    }  # fmt: skip
    quantum: ClassVar[dict[str, dict[str, float]]] = {
        BL: {"revolutions": 1.0},
        RC: {"granule_mass": 0.5},
        TP: {"tablets_total": 1.0, "rejects_total": 0.1},
    }
    idle: ClassVar[dict[str, dict[str, float]]] = {
        BL: {"speed": 0.0, "speed_sp": 0.0, "revolutions": 0.0, "blend_rsd": 0.0,
             "room_rh": ROOM_RH},
        RC: {
            "roll_force": 0.0, "roll_force_sp": 0.0, "roll_gap": 0.0, "roll_speed": 0.0,
            "screw_speed": 0.0, "mill_speed": 0.0, "ribbon_density": 0.0, "granule_mass": 0.0,
            "room_rh": ROOM_RH,
        },
        TP: {
            "comp_force": 0.0, "comp_force_sp": 0.0, "precomp_force": 0.0, "turret_speed": 0.0,
            "turret_speed_sp": 0.0, "feed_frame": 0.0, "feed_frame_sp": 0.0,
            "ejection_force": 0.0, "tablet_weight": 0.0, "weight_rsd": 0.0, "hardness": 0.0,
            "tablets_total": 0.0, "rejects_total": 0.0, "room_rh": ROOM_RH,
        },
    }  # fmt: skip
    fault_class: ClassVar[dict[FaultType, str]] = {
        FaultType.ROLL_FORCE_DRIFT: RC,
        FaultType.PUNCH_STICKING: TP,
        FaultType.HOPPER_BRIDGING: TP,
        FaultType.HVAC_HUMIDITY: RC,  # one HVAC zone serves the line; labelled where injected
    }
    fault_defaults: ClassVar[dict[FaultType, dict[str, float | str]]] = {
        FaultType.ROLL_FORCE_DRIFT: {"rate": 0.08, "max": 0.4},  # fraction over-read per h
        FaultType.PUNCH_STICKING: {"rate": 70.0},  # N/h of ejection force
        FaultType.HOPPER_BRIDGING: {"probability": 0.4, "depth": 0.09},
        FaultType.HVAC_HUMIDITY: {"rate": 6.0, "max": 62.0},  # % RH per h
        FaultType.STUCK_SENSOR: {"tag": "", **STUCK_DEFAULTS},
    }
    lever_class: ClassVar[dict[str, str]] = {
        "lube_time": BL, "roll_force": RC, "comp_force": TP, "turret_speed": TP,
        "feed_frame": TP,
    }  # fmt: skip

    # -- setup ----------------------------------------------------------------------------------

    def init_process(self) -> None:
        self.bl, self.rc, self.tp = pr.Blender(), pr.Compactor(), pr.Press()
        lots = self.spec.lots or (DEFAULT_LOT,)
        kg = sum(lot.quantity_kg for lot in lots)
        self.api_d50 = sum(lot.quantity_kg * lot.properties.get("d50_um", 215.0) for lot in lots)
        self.api_d50 /= kg
        self.api_free_sa = sum(
            lot.quantity_kg * lot.properties.get("free_sa_pct", 0.05) for lot in lots
        ) / kg  # fmt: skip
        tr = self.spec.traits
        self.tab_trait = tr.get("tabletability", 1.0)
        self.flow_trait = tr.get("flow", 1.0)
        self.exposure = 0.0  # ∫ humidity driving hydrolysis dt
        self.bridge = 0.0  # this tick's underfill
        self.rh = ROOM_RH

    def begin(self) -> None:
        bl = self.cells[0]
        self.enter(bl)
        self.set_operation(bl, Operation.CHARGE)
        for lot in self.spec.lots or (DEFAULT_LOT,):
            self.consumed(bl, lot)

    # -- operations -----------------------------------------------------------------------------

    def transition(self) -> None:
        if self.active is None:
            return
        bl_cell, rc_cell, tp_cell = self.cells
        op, age, bl = self.operation, self.op_age(), self.bl
        if op is Operation.CHARGE and age >= CHARGE_H - 1e-9:
            self.set_operation(bl_cell, Operation.BLENDING)
        elif op is Operation.BLENDING:
            measured = self.sensors[bl_cell].control("blend_rsd", bl.rsd)
            if (bl.revolutions >= MIN_REVOLUTIONS and measured < BLEND_END_RSD) or (
                bl.revolutions >= MAX_REVOLUTIONS
            ):
                self.lab(
                    bl_cell,
                    {"blend_uniformity": bl.rsd * (1 + float(self.rng.normal(0, 0.08)))},
                    LAB_UNITS,
                )
                self.set_operation(bl_cell, Operation.LUBRICATION)
        elif op is Operation.LUBRICATION and bl.lube_min >= self.levers.lube_time - 1e-6:
            self.set_operation(bl_cell, Operation.DISCHARGE)
        elif op is Operation.DISCHARGE and age >= DISCHARGE_H - 1e-9:
            self.enter(rc_cell)
            self.set_operation(rc_cell, Operation.COMPACTION)
        elif op is Operation.COMPACTION and self.rc.processed_kg >= bl.loaded_kg - 1e-6:
            sf = self.rc.sf_mean
            self.lab(
                rc_cell,
                {
                    "granule_d50": (150.0 + 900.0 * (sf - 0.55)) * (1 + self.rng.normal(0, 0.04)),
                    "bulk_density": 0.45 + 0.4 * (sf - 0.55) + self.rng.normal(0, 0.005),
                },
                LAB_UNITS,
            )
            self.tp.target_k = self.rc.granule_kg * (1 - PRESS_SETUP_LOSS) / pr.TABLET_MG * 1000
            self.enter(tp_cell)
            self.set_operation(tp_cell, Operation.COMPRESSION)
        elif op is Operation.COMPRESSION and self.tp.tablets_k >= self.tp.target_k:
            self._release(tp_cell)

    def _release(self, cell: str) -> None:
        """Compression done: the CoA, the tablets into stock, the batch end (ADR-0020)."""
        tp, rng = self.tp, self.rng
        n = max(tp.stats.get("n", 0.0), 1e-9)
        h = tp.stats["h"] / n
        rsd = tp.stats["rsd"] / n
        good_k = tp.tablets_k - tp.rejects_k
        blend_rsd = self.bl.rsd
        self.coa = {
            "yield": 100.0 * good_k * 1000.0 / pr.TARGET_TABLETS,
            "hardness": h,
            "friability": pr.friability(h),
            "dissolution": pr.dissolution(h, self.levers.lube_time, self.api_d50),
            "av": 2.4 * math.sqrt(blend_rsd**2 + rsd**2),
            "free_sa": self.api_free_sa + self.exposure,
            "good_kg": good_k * pr.TABLET_MG / 1000.0,
        }
        self.coa["assay"] = 100.0 - 0.5 * self.coa["free_sa"] - 0.2
        measured = {
            "assay": self.coa["assay"] + rng.normal(0, 0.4),
            "av": self.coa["av"] * (1 + rng.normal(0, 0.05)),
            "dissolution": self.coa["dissolution"] + rng.normal(0, 1.0),
            "hardness": h + rng.normal(0, 1.5),
            "friability": max(0.0, self.coa["friability"] * (1 + rng.normal(0, 0.1))),
            "free_sa": max(0.0, self.coa["free_sa"] * (1 + rng.normal(0, 0.04))),
            "yield": self.coa["yield"] + rng.normal(0, 0.05),
        }
        self.lab(cell, measured, LAB_UNITS)
        ok = (
            measured["dissolution"] >= SPEC["dissolution"]
            and measured["av"] <= SPEC["av"]
            and measured["free_sa"] <= SPEC["free_sa"]
            and measured["friability"] <= SPEC["friability"]
        )
        self.produced(cell, self.coa["good_kg"])
        self.finish("ACCEPTED" if ok else "REJECTED", None if ok else "CoA out of specification")

    # -- process ----------------------------------------------------------------------------------

    def _force_drift(self) -> float:
        d = 0.0
        for f in self.active_faults(FaultType.ROLL_FORCE_DRIFT):
            d += min(f.param("max"), f.param("rate") * f.age(self.t_h))
        return d

    def integrate(self, dt: float) -> None:
        op, lv = self.operation, self.levers
        # the line's room humidity (one HVAC zone)
        rh = ROOM_RH + 1.5 * math.sin(2.0 * math.pi * self.t_h / 24.0)
        for f in self.active_faults(FaultType.HVAC_HUMIDITY):
            rh = min(f.param("max"), rh + f.param("rate") * f.age(self.t_h))
        self.rh = rh
        self.exposure += 5.0e-5 * max(rh - 20.0, 0.0) ** 2 * dt

        if self.active == self.cells[0]:
            bl = self.bl
            if op is Operation.CHARGE:
                bl.loaded_kg = pr.BLEND_KG * min(1.0, self.op_age() / CHARGE_H)
                bl.speed = 0.0
            elif op in (Operation.BLENDING, Operation.LUBRICATION):
                bl.speed = BLEND_RPM
                bl.revolutions += BLEND_RPM * 60.0 * dt
                bl.rsd = pr.blend_rsd(bl.revolutions, self.api_d50)
                if op is Operation.LUBRICATION:
                    bl.lube_min += 60.0 * dt
            else:
                bl.speed = 0.0
                if op is Operation.DISCHARGE:
                    bl.loaded_kg = min(bl.loaded_kg, pr.BLEND_KG * (1.0 - BLEND_LOSS))
        elif self.active == self.cells[1]:
            rc = self.rc
            rc.force = lv.roll_force / (1.0 + self._force_drift())
            rc.solid_fraction = pr.solid_fraction(rc.force, self.tab_trait)
            rc.screw = 30.0 * (rc.force / 7.0) ** 0.5
            rc.gap = GAP_SP * (lv.roll_force / rc.force) ** 0.15
            rc.roll_speed, rc.mill = ROLL_SPEED, MILL_SPEED
            kg = min(COMPACTION_KG_H * dt, self.bl.loaded_kg - rc.processed_kg)
            rc.processed_kg += kg
            rc.sf_sum += kg * rc.solid_fraction
            rc.granule_kg += kg * (1.0 - COMPACTION_LOSS)
        elif self.active == self.cells[2]:
            self._compress(dt)

    def _compress(self, dt: float) -> None:
        tp, lv, rng = self.tp, self.levers, self.rng
        sf = self.rc.sf_mean
        ratio = lv.feed_frame / lv.turret_speed
        for f in self.active_faults(FaultType.PUNCH_STICKING):
            tp.buildup = f.param("rate") * f.age(self.t_h)
        bridge = 0.0
        for f in self.active_faults(FaultType.HOPPER_BRIDGING):
            if rng.random() < f.param("probability"):
                bridge = f.param("depth") * float(rng.random())
        self.bridge = bridge
        tp.turret, tp.feed_frame = lv.turret_speed, lv.feed_frame
        tp.force = lv.comp_force * (1.0 - 2.5 * bridge)
        tp.hardness = pr.hardness(tp.force, sf, lv.lube_time, ratio, self.tab_trait)
        tp.weight_rsd = pr.weight_rsd(pr.fines(sf, self.flow_trait), tp.turret, ratio)
        tp.weight_rsd += 12.0 * bridge
        tp.weight = pr.TABLET_MG * (1.0 - bridge)
        tp.ejection = pr.ejection(lv.lube_time, tp.force, tp.buildup)
        reject = (
            pr.weight_reject(tp.weight_rsd)
            + pr.capping(tp.force, tp.turret, sf)
            + pr.chipping(tp.hardness)
            + pr.sticking(lv.lube_time, tp.buildup)
        )
        made = pr.STATIONS * tp.turret * 60.0 / 1000.0 * dt * (1.0 - bridge)
        tp.tablets_k += made
        tp.rejects_k += made * min(reject, 1.0)
        st = tp.stats
        st["n"] = st.get("n", 0.0) + made
        st["h"] = st.get("h", 0.0) + made * tp.hardness
        st["rsd"] = st.get("rsd", 0.0) + made * tp.weight_rsd

    # -- readings -------------------------------------------------------------------------------

    def read(self, cell: str, var: str) -> float:
        bank, rng, lv = self.sensors[cell], self.rng, self.levers
        if var == "room_rh":
            return bank.sample("room_rh", self.rh, rng)
        if cell == self.cells[0]:
            bl = self.bl
            match var:
                case "speed":
                    return bank.sample("speed", bl.speed, rng)
                case "speed_sp":
                    return bl.speed
                case "revolutions":
                    return bank.sample("revolutions", bl.revolutions, rng)
                case "blend_rsd":
                    rsd = bl.rsd if self.operation is not Operation.CHARGE else 25.0
                    return bank.sample("blend_rsd", rsd, rng)
        elif cell == self.cells[1]:
            rc = self.rc
            match var:
                case "roll_force":
                    return bank.sample("roll_force", rc.force * (1 + self._force_drift()), rng)
                case "roll_force_sp":
                    return lv.roll_force
                case "roll_gap":
                    return bank.sample("roll_gap", rc.gap, rng)
                case "roll_speed":
                    return bank.sample("roll_speed", rc.roll_speed, rng)
                case "screw_speed":
                    return bank.sample("screw_speed", rc.screw, rng)
                case "mill_speed":
                    return bank.sample("mill_speed", rc.mill, rng)
                case "ribbon_density":
                    return bank.sample("ribbon_density", rc.solid_fraction * pr.TRUE_DENSITY, rng)
                case "granule_mass":
                    return bank.sample("granule_mass", rc.granule_kg, rng)
        else:
            tp = self.tp
            match var:
                case "comp_force":
                    return bank.sample("comp_force", tp.force, rng)
                case "comp_force_sp":
                    return lv.comp_force
                case "precomp_force":
                    return bank.sample("precomp_force", 0.2 * tp.force, rng)
                case "turret_speed":
                    return bank.sample("turret_speed", tp.turret, rng)
                case "turret_speed_sp":
                    return lv.turret_speed
                case "feed_frame":
                    return bank.sample("feed_frame", tp.feed_frame, rng)
                case "feed_frame_sp":
                    return lv.feed_frame
                case "ejection_force":
                    return bank.sample("ejection_force", tp.ejection, rng)
                case "tablet_weight":  # checkweigher: the mean of a 10-tablet sample
                    sd = tp.weight * tp.weight_rsd / 100.0 / math.sqrt(10.0)
                    return bank.sample("tablet_weight", tp.weight + sd * rng.normal(), rng)
                case "weight_rsd":
                    return bank.sample("weight_rsd", tp.weight_rsd, rng)
                case "hardness":
                    return bank.sample("hardness", tp.hardness, rng)
                case "tablets_total":
                    return bank.sample("tablets_total", tp.tablets_k, rng)
                case "rejects_total":
                    return bank.sample("rejects_total", tp.rejects_k, rng)
        raise KeyError(var)

    # -- context for the harness, truth and operator ----------------------------------------------

    def truth(self) -> Any:
        return {
            "bl": copy.copy(self.bl),
            "rc": copy.copy(self.rc),
            "tp": copy.copy(self.tp),
            "rh": self.rh,
            "force_drift": self._force_drift(),
        }

    def setpoints(self) -> dict[str, float]:
        lv = self.levers
        return {"roll_force": lv.roll_force, "comp_force": lv.comp_force}

    def lever_open(self, name: str) -> bool:
        op, cell = self.operation, self.active
        if cell is None:
            return False
        stage = self.cells.index(cell)
        match name:
            case "lube_time":
                return op in (Operation.CHARGE, Operation.BLENDING) or (op is Operation.LUBRICATION)
            case "roll_force":
                return stage <= 1
            case "comp_force" | "turret_speed" | "feed_frame":
                return True
        return False

    def target(self) -> float:
        return self.coa.get("yield", 0.0)
