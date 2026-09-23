"""True process state of one CHO fed-batch bioreactor and its dynamics (ADR-0008).

Pure functions on plain dataclasses; time is in hours. `step()` is the hot loop of the
whole simulator (about 240k calls per batch at a 5 s step), so it is written with local
variables and plain floats rather than NumPy.

What the model is for: plausible trends, faults with the right signatures, and a titer
that depends on the optimizer's levers through mechanisms, not a lookup:

- lower temperature arrests growth but raises specific productivity (qP), so the
  shift day and production temperature trade biomass against productive time;
- pH away from ~7.0 slows growth and productivity; higher pH drives lactate;
- DO below ~20% limits growth; high agitation (to hold a high DO SP) adds shear death;
- feed supplies glucose and a lumped nutrient (amino acids) that productivity needs;
  too little starves qP, too much raises osmolality, which kills cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Params:
    """Kinetic and physical constants. Rates are per hour."""

    mu_max: float = 0.043
    x_max: float = 32.0  # 1e6 cells/mL, logistic ceiling
    ks_glc: float = 0.5  # g/L
    kd0: float = 0.0018
    kd_lac: float = 0.0012  # per g/L lactate
    kd_starve: float = 0.02  # when glucose is exhausted
    kd_osm: float = 0.004  # per 50 mOsm above osm_tol
    osm_tol: float = 380.0
    kd_shear: float = 0.00005  # per rpm above 110
    lysis: float = 0.01  # dead cells lyse
    kd_swing: float = 0.004  # per °C/h of temperature change

    q_glc: float = 0.0062  # g/L per (1e6 cells/mL) per h
    q_lac_prod: float = 0.0030
    q_lac_cons: float = 0.0012
    q_p: float = 0.00072  # g/L titer per (1e6 cells/mL) per h at 36.5 °C
    q_nut: float = 0.0035  # nutrient use, g/L per (1e6 cells/mL) per h
    k_nut: float = 0.5  # g/L, half-saturation of qP on the nutrient

    t_ref: float = 36.5
    ph_opt_mu: float = 7.02
    ph_opt_qp: float = 6.97
    ph_width: float = 0.4

    # Oxygen
    kla0: float = 4.0
    our: float = 20.0  # %DO per h per (1e6 cells/mL)

    # pH (an integrating process: nothing pulls it back but control)
    ph_strip: float = 0.03  # per h per L/min air (CO2 stripping raises pH)
    ph_resp: float = 0.001  # per h per (1e6 cells/mL) (respiratory CO2 lowers pH)
    ph_lac: float = 0.5  # per g/L of net lactate production
    ph_co2: float = 0.3  # per h per L/min CO2
    ph_base: float = 0.17  # per (mL base / L broth)

    # Temperature
    tau_temp: float = 0.5  # h, jacket to broth
    ambient: float = 22.0
    loss: float = 0.05

    # Feed and medium
    glc_feed: float = 52.0  # g/L in feed
    nut_feed: float = 60.0  # g/L in feed
    osm_feed: float = 800.0
    osm_medium: float = 290.0


@dataclass(frozen=True, slots=True)
class Traits:
    """Batch-to-batch variation of the cells themselves (seeded per batch)."""

    growth: float = 1.0
    productivity: float = 1.0
    metabolism: float = 1.0


@dataclass(slots=True)
class TrueState:
    """What is really in the vessel. Sensors see this through a sensor model."""

    xv: float = 0.0  # viable cells, 1e6 cells/mL
    xd: float = 0.0  # dead cells
    glc: float = 4.5  # g/L
    nut: float = 3.0  # g/L, lumped amino acids
    lac: float = 0.0  # g/L
    titer: float = 0.0  # g/L
    osm: float = 290.0  # mOsm/kg
    vol: float = 1500.0  # L
    temp: float = 36.5  # °C
    ph: float = 7.0
    do: float = 60.0  # % air saturation
    contaminant: float = 0.0  # arbitrary units; 0 = sterile
    feed_total: float = 0.0  # L
    base_total: float = 0.0  # mL
    ivc: float = 0.0  # integral of viable cells, 1e6 cells·day/mL


@dataclass(frozen=True, slots=True)
class Actuators:
    jacket: float  # °C, jacket temperature from the temperature loop
    agitation: float  # rpm
    air: float  # L/min
    o2: float  # L/min
    co2: float  # L/min
    base: float  # mL/h
    feed: float  # L/h
    kla_factor: float = 1.0  # < 1 when the sparger fouls
    contaminant_growth: float = 0.0  # per h; > 0 only under contamination


def viability(s: TrueState) -> float:
    total = s.xv + s.xd
    return 100.0 * s.xv / total if total > 0 else 100.0


def do_star(air: float, o2: float) -> float:
    """Saturation DO (% air sat.) of the sparged gas mix."""
    total = air + o2
    if total <= 0:
        return 100.0
    return 100.0 * (0.21 * air + o2) / (0.21 * total)


def kla(p: Params, agitation: float, air: float, o2: float, factor: float) -> float:
    gas = max(air + o2, 0.01)
    return p.kla0 * (agitation / 100.0) ** 1.5 * (gas / 0.5) ** 0.4 * factor


def step(s: TrueState, a: Actuators, p: Params, tr: Traits, dt: float) -> TrueState:
    """Advance the true state by `dt` hours (explicit Euler)."""
    xv, xd, glc, lac, vol, nut = s.xv, s.xd, s.glc, s.lac, s.vol, s.nut
    temp, ph, do = s.temp, s.ph, s.do
    dT = p.t_ref - temp  # > 0 after the shift

    # --- specific rates -------------------------------------------------------------
    f_glc = glc / (p.ks_glc + glc) if glc > 0 else 0.0
    f_do = min(1.0, do / 20.0) if do > 0 else 0.0
    f_ph_mu = math.exp(-(((ph - p.ph_opt_mu) / p.ph_width) ** 2))
    f_t_mu = math.exp(-0.30 * dT)
    mu = p.mu_max * tr.growth * f_glc * f_do * f_ph_mu * f_t_mu * max(0.0, 1.0 - xv / p.x_max)

    kd = p.kd0 + p.kd_lac * lac
    if glc < 0.2:
        kd += p.kd_starve
    if s.osm > p.osm_tol:
        kd += p.kd_osm * (s.osm - p.osm_tol) / 50.0
    if a.agitation > 110.0:
        kd += p.kd_shear * (a.agitation - 110.0)
    if s.contaminant > 1.0:
        kd += 0.004 * math.log(s.contaminant)
    dtemp_dt = (a.jacket - temp) / p.tau_temp - p.loss * (temp - p.ambient)
    kd += p.kd_swing * abs(dtemp_dt)  # temperature swings stress the cells
    kd *= math.exp(-0.15 * dT)  # cooler cultures die slower

    s_t = math.exp(-0.40 * dT)  # metabolic activity
    q_glc = p.q_glc * tr.metabolism * math.exp(-0.20 * dT) * f_glc
    q_lac = p.q_lac_prod * tr.metabolism * s_t * (1.0 + 3.0 * (ph - 7.0)) * f_glc
    q_lac -= p.q_lac_cons * (1.0 - s_t) * lac / (lac + 0.5)
    g_t = 1.0 + 0.35 * dT - 0.05 * dT * dT
    g_ph = math.exp(-(((ph - p.ph_opt_qp) / p.ph_width) ** 2))
    g_do = min(1.0, do / 25.0) * max(0.3, 1.0 - 0.01 * max(0.0, do - 45.0)) if do > 0 else 0.0
    g_nut = nut / (p.k_nut + nut) if nut > 0 else 0.0
    q_p = p.q_p * tr.productivity * max(0.0, g_t) * g_ph * g_do * g_nut * (1.0 + p.k_nut / 3.0)

    # --- volumes and dilution -------------------------------------------------------
    feed_l = a.feed * dt
    base_l = a.base * dt / 1000.0
    new_vol = vol + feed_l + base_l
    dil = vol / new_vol  # concentration factor from added volume

    # --- balances -------------------------------------------------------------------
    cont = s.contaminant
    if a.contaminant_growth > 0.0:
        cont = max(cont, 1e-3) * math.exp(a.contaminant_growth * dt)
    lac_cont = 0.05 * cont if cont > 1e-3 else 0.0  # contaminant makes acid (as lactate)

    xv_n = (xv + (mu - kd) * xv * dt) * dil
    xd_n = (xd + (kd * xv - p.lysis * xd) * dt) * dil
    glc_n = (glc - q_glc * xv * dt) * dil + feed_l * p.glc_feed / new_vol
    nut_n = (nut - p.q_nut * tr.metabolism * xv * dt) * dil + feed_l * p.nut_feed / new_vol
    lac_rate = q_lac * xv + lac_cont
    lac_n = (lac + lac_rate * dt) * dil
    titer_n = (s.titer + q_p * xv * dt) * dil
    osm_n = (s.osm * vol + p.osm_feed * feed_l + 1000.0 * base_l) / new_vol

    # Oxygen: transfer in, uptake by cells and contaminant.
    our = p.our * (xv * (0.6 + 0.4 * s_t) + 8.0 * cont)
    k = kla(p, a.agitation, a.air, a.o2, a.kla_factor)
    do_n = do + (k * (do_star(a.air, a.o2) - do) - our) * dt

    # pH: stripping up; respiration, lactate and CO2 down; base up.
    dph = (
        p.ph_strip * a.air
        - p.ph_resp * xv
        - p.ph_lac * max(lac_rate, -0.05)
        - p.ph_co2 * a.co2
        + p.ph_base * a.base / new_vol
    )
    ph_n = ph + dph * dt

    temp_n = temp + dtemp_dt * dt

    return TrueState(
        xv=max(xv_n, 0.0),
        xd=max(xd_n, 0.0),
        glc=max(glc_n, 0.0),
        nut=max(nut_n, 0.0),
        lac=max(lac_n, 0.0),
        titer=max(titer_n, 0.0),
        osm=osm_n,
        vol=new_vol,
        temp=temp_n,
        ph=ph_n,
        do=min(max(do_n, 0.0), 400.0),
        contaminant=cont,
        feed_total=s.feed_total + feed_l,
        base_total=s.base_total + a.base * dt,
        ivc=s.ivc + xv * dt / 24.0,
    )


def inoculate(s: TrueState, xv0: float) -> TrueState:
    return replace(s, xv=xv0, xd=xv0 * 0.03)
