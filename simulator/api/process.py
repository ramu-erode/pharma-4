"""True state and mechanisms of an aspirin API batch (ADR-0019). Pure functions on plain
dataclasses; time in hours, temperature in °C, amounts in kmol, kg and L.

What the model is for: plausible trends, faults with the right signatures, and a yield
that depends on the levers through competing mechanisms:

- Acetylation SA + Ac2O -> ASA + AcOH is second order and exothermic. Hotter and with
  more anhydride it converts faster, but two side reactions grow faster still:
  ASA + SA -> acetylsalicylsalicylic acid (ASSA), and ASA + Ac2O -> acetylsalicylic
  anhydride (ASAN), which keeps growing through the hold and while the batch cools.
- ASA solubility rises steeply with temperature and with the anhydride excess, so more
  anhydride leaves more product in the mother liquor.
- Cooling crystallisation is a moment model with seeding: fast cooling means high
  supersaturation, many nuclei and fines. Fines blind the cake, hold more liquid, wash
  worse and pass the cloth.
- The wet cake hydrolyses back to SA while it dries; a hot dryer agglomerates, a slow
  one grinds the product into dust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

R = 8.314e-3  # kJ/mol/K
MW_SA, MW_ASA, MW_AC2O = 138.12, 180.16, 102.09
SA_CHARGE_KG = 500.0
N_SA0 = SA_CHARGE_KG / MW_SA  # kmol
THEORETICAL_KG = N_SA0 * MW_ASA
ACOH_L = 1200.0  # solvent
AC2O_DENSITY = 1.08  # kg/L
SA_VOLUME_L = SA_CHARGE_KG / 1.44
WORKING_VOLUME_L = 4000.0
T_REF = 85.0

# Crystals: volume shape factor and density, in grams per µm³ of L³.
CRYSTAL_G_PER_UM3 = 0.5 * 1.40e-12
SEED_KG, SEED_UM = 3.0, 60.0
FINAL_TEMP = 5.0
AGEING_H = 1.0


@dataclass(frozen=True, slots=True)
class Params:
    k1: float = 1.8  # m³/kmol/h at 85 °C, acetylation
    ea1: float = 60.0  # kJ/mol
    k2: float = 0.0040  # ASSA
    ea2: float = 110.0
    k3: float = 0.0045  # ASAN
    ea3: float = 130.0
    kh: float = 0.006  # 1/h at 85 °C, hydrolysis of dissolved ASA once quenched
    eh: float = 95.0
    quench_water_l: float = 60.0
    dh: float = 60_000.0  # kJ/kmol released by acetylation
    cp: float = 2.0  # kJ/kg/K
    ua: float = 3.6  # kW/K jacket, clean, at 120 rpm
    tau_jacket: float = 0.10  # h
    kb: float = 5.0e11  # nuclei per m³ per h at S = 1
    nb: float = 2.4  # nucleation order
    kg: float = 300.0  # µm/h at S = 1
    ng: float = 1.0
    sol_a: float = 21.0  # g/L at 0 °C
    sol_b: float = 0.0347  # 1/°C
    sol_excess: float = 0.8  # solubility gain per unit anhydride excess


@dataclass(slots=True)
class Reactor:
    """RX-201: what is really in the vessel."""

    temp: float = 25.0
    jacket: float = 25.0
    liquor_l: float = 0.0  # AcOH + Ac2O volume (the solvent)
    solids_l: float = 0.0
    n_sa: float = 0.0
    n_ac2o: float = 0.0
    n_asa: float = 0.0  # dissolved
    n_assa: float = 0.0
    n_asan: float = 0.0
    water_l: float = 0.0  # quench water: excess anhydride gone, hydrolysis possible
    ac2o_dosed_kg: float = 0.0  # true
    ac2o_metered_kg: float = 0.0  # what the flowmeter totalises
    dose_flow: float = 0.0  # kg/h, true
    agitation: float = 0.0
    power: float = 0.0
    m0: float = 0.0  # crystal moments, totals over the liquor (#, µm·#, µm²·#, µm³·#)
    m1: float = 0.0
    m2: float = 0.0
    m3: float = 0.0
    seeded: bool = False
    max_temp: float = 25.0

    @property
    def crystal_kg(self) -> float:
        return self.m3 * CRYSTAL_G_PER_UM3 / 1000.0

    @property
    def conc(self) -> float:
        """Dissolved ASA, g/L."""
        return self.n_asa * MW_ASA * 1e6 / max(self.liquor_l, 1.0) / 1000.0

    @property
    def conversion(self) -> float:
        return 100.0 * (1.0 - self.n_sa / N_SA0) if self.n_sa or self.n_asa else 0.0

    @property
    def d32(self) -> float:
        return self.m3 / self.m2 if self.m2 > 0 else 0.0

    @property
    def mass_kg(self) -> float:
        return (
            SA_CHARGE_KG * (self.solids_l > 0)
            + ACOH_L * 1.05 * (self.liquor_l > 0)
            + (self.ac2o_dosed_kg)
        )

    @property
    def volume_l(self) -> float:
        return self.liquor_l + self.solids_l


@dataclass(slots=True)
class Dryer:
    """FD-202: the slurry, then the cake."""

    liquor_l: float = 0.0  # still to filter
    filtered_l: float = 0.0
    crystal_kg: float = 0.0
    d32: float = 0.0
    fines: float = 0.0  # mass fraction below the cloth cut
    impurity_kg: dict[str, float] | None = None  # in the liquor: sa, assa, asan
    moisture: float = 0.0  # % w/w
    cake_temp: float = 25.0
    jacket: float = 25.0
    vacuum: float = 1013.0  # mbar a
    wash_l: float = 0.0
    free_sa_kg: float = 0.0  # hydrolysis during drying
    filter_pressure: float = 0.0
    filtrate_flow: float = 0.0  # L/min
    blind: float = 1.0  # cloth resistance factor (1 = clean)
    losses_kg: float = 0.0
    dust_kg: float = 0.0
    drying_h: float = 0.0


def arrhenius(k_ref: float, ea: float, temp: float) -> float:
    return k_ref * math.exp(-ea / R * (1.0 / (temp + 273.15) - 1.0 / (T_REF + 273.15)))


def solubility(p: Params, temp: float, ratio: float) -> float:
    """ASA solubility in the liquor, g/L."""
    return p.sol_a * math.exp(p.sol_b * temp) * (1.0 + p.sol_excess * max(ratio - 1.0, 0.0))


def boiling_point(mbar: float) -> float:
    """Water's boiling point at a pressure (Antoine), °C: the constant-rate cake temperature."""
    mmhg = max(mbar, 1.0) * 0.750062
    return 1730.63 / (8.07131 - math.log10(mmhg)) - 233.426


def react(r: Reactor, p: Params, dt: float, mix: float, k1_trait: float) -> float:
    """Reaction step (all three reactions); returns the heat released, kJ."""
    v = max(r.liquor_l, 1.0) / 1000.0  # m³
    sa, ac, asa = r.n_sa / v, r.n_ac2o / v, r.n_asa / v
    r1 = arrhenius(p.k1, p.ea1, r.temp) * k1_trait * mix * sa * ac * v
    r2 = arrhenius(p.k2, p.ea2, r.temp) * sa * asa * v
    r3 = arrhenius(p.k3, p.ea3, r.temp) * asa * ac * v
    r1 = min(r1, r.n_sa / dt, r.n_ac2o / dt)
    r2 = min(r2, max(r.n_sa / dt - r1, 0.0))
    r3 = min(r3, max(r.n_ac2o / dt - r1, 0.0))
    r4 = arrhenius(p.kh, p.eh, r.temp) * asa * v if r.water_l > 0 else 0.0
    r.n_sa += r4 * dt
    r.n_asa -= r4 * dt
    r.n_sa -= (r1 + r2) * dt
    r.n_ac2o -= (r1 + r3) * dt
    r.n_asa += (r1 - r2 - r3) * dt
    r.n_assa += r2 * dt
    r.n_asan += r3 * dt
    return p.dh * r1 * dt


def quench(r: Reactor, p: Params) -> None:
    """Water destroys the excess anhydride (to acetic acid) before crystallisation."""
    r.liquor_l += p.quench_water_l
    r.water_l = p.quench_water_l
    r.n_ac2o = 0.0


def crystallise(r: Reactor, p: Params, dt: float, ratio: float, nuc_trait: float) -> None:
    csat = solubility(p, r.temp, ratio)
    c = r.conc
    if not r.seeded and c > csat * 1.02:
        seed_g_each = CRYSTAL_G_PER_UM3 * SEED_UM**3
        n = SEED_KG * 1000.0 / seed_g_each
        r.m0 += n
        r.m1 += n * SEED_UM
        r.m2 += n * SEED_UM**2
        r.m3 += n * SEED_UM**3
        r.n_asa -= SEED_KG / MW_ASA  # seeds come from the batch's own product
        r.seeded = True
    s = (c - csat) / csat
    if s <= 0.0 or r.m0 <= 0.0:
        return
    g = p.kg * s**p.ng
    b = p.kb * nuc_trait * s**p.nb * r.liquor_l / 1000.0
    dm3 = 3.0 * g * r.m2 * dt
    grown_kmol = dm3 * CRYSTAL_G_PER_UM3 / 1000.0 / MW_ASA
    if grown_kmol > 0.8 * r.n_asa:  # never crystallise more than is dissolved
        scale = 0.8 * r.n_asa / grown_kmol
        g, dm3, grown_kmol = g * scale, dm3 * scale, grown_kmol * scale
    r.m3 += dm3
    r.m2 += 2.0 * g * r.m1 * dt
    r.m1 += g * r.m0 * dt
    r.m0 += b * dt
    r.n_asa -= grown_kmol


def fines_loss(d32: float) -> float:
    """Fraction of the crystal mass that passes the filter cloth. It scales with the
    specific surface (1/d32²), which is what a nucleation burst raises."""
    return min(0.5, 0.06 * (120.0 / max(d32, 10.0)) ** 2)


def filtrate_flow(d: Dryer, pressure: float, cloth_trait: float) -> float:
    """Constant-pressure cake filtration (Darcy), L/min. Finer crystals, a thicker cake
    and a blinded cloth all slow it."""
    v_char = 430.0 * (max(d.d32, 5.0) / 200.0) ** 2
    return 40.0 * (pressure / 1.5) / (d.blind * cloth_trait + d.filtered_l / v_char)


def initial_moisture(d32: float) -> float:
    """Liquid a deliquored cake holds, % w/w; fines hold more."""
    return 20.0 + 9.0 * min(100.0 / max(d32, 20.0), 5.0)
