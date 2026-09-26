"""True state and mechanisms of an aspirin 500 mg tablet batch (ADR-0019). Pure
functions on plain dataclasses; time in hours.

Dry granulation, because aspirin hydrolyses in moisture: blend, roller-compact, mill,
compress. What the model is for: faults with the right signatures and a tablet yield
(good tablets over theoretical) that depends on the levers through rejects:

- Blend uniformity decays with revolutions, slower and to a worse floor for coarse API.
- Ribbon solid fraction rises with specific roll force. Denser ribbons give coarser
  granules with fewer fines, so they flow better and the weight varies less. But they
  work-harden, so they make softer tablets and cap more easily.
- Tablet hardness rises with compression force and falls with work-hardening and
  over-lubrication. Soft tablets chip and are rejected; high force and a fast turret
  cap tablets. Too little lubricant makes them stick and pick.
- Weight variation grows with fines, turret speed and a feed frame out of step with
  the turret. The checkweigher rejects tablets outside ±5%.
- Dissolution slows for hard tablets, long lubrication and coarse API. Free SA grows
  with room humidity and exposure time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

API_PER_TABLET_G = 0.5
TABLET_MG = 600.0
TARGET_TABLETS = 400_000
API_KG = TARGET_TABLETS * API_PER_TABLET_G / 1000.0  # 200 kg
BLEND_KG = TARGET_TABLETS * TABLET_MG / 1e6  # 240 kg
STATIONS = 36
TRUE_DENSITY = 1.35  # g/cm³, the blend
REJECT_LIMIT = 0.05  # checkweigher band, ±5% of target weight


@dataclass(slots=True)
class Blender:
    loaded_kg: float = 0.0
    revolutions: float = 0.0
    speed: float = 0.0
    rsd: float = 25.0  # % blend RSD, true
    lube_min: float = 0.0


@dataclass(slots=True)
class Compactor:
    processed_kg: float = 0.0
    granule_kg: float = 0.0
    force: float = 0.0  # kN/cm, true
    solid_fraction: float = 0.0
    sf_sum: float = 0.0  # mass-weighted, for the granule's average
    gap: float = 0.0
    screw: float = 0.0
    roll_speed: float = 0.0
    mill: float = 0.0

    @property
    def sf_mean(self) -> float:
        return self.sf_sum / self.processed_kg if self.processed_kg > 0 else 0.0


@dataclass(slots=True)
class Press:
    tablets_k: float = 0.0  # thousands made
    rejects_k: float = 0.0
    target_k: float = TARGET_TABLETS / 1000.0
    force: float = 0.0  # kN, true
    turret: float = 0.0
    feed_frame: float = 0.0
    weight: float = 0.0  # mg, mean
    weight_rsd: float = 0.0  # %
    hardness: float = 0.0  # N
    ejection: float = 0.0  # N
    buildup: float = 0.0  # N of extra ejection force from product on the punches
    stats: dict[str, float] = field(default_factory=dict)  # tablet-weighted sums


def blend_rsd(revolutions: float, api_d50: float) -> float:
    """NIR blend RSD (%) after `revolutions`: coarse API mixes slower and segregates."""
    coarse = max(api_d50 - 250.0, 0.0) / 100.0
    floor = 1.2 + 1.5 * coarse
    n0 = 45.0 * (api_d50 / 250.0)
    return floor + (25.0 - floor) * math.exp(-revolutions / n0)


def solid_fraction(force: float, tabletability: float = 1.0) -> float:
    """Ribbon solid fraction from specific roll force (kN/cm)."""
    return 0.55 + 0.20 * math.log(max(force, 0.5) / 2.0) / math.log(6.0) * tabletability**0.2


def fines(sf: float, flow_trait: float = 1.0) -> float:
    """Granule mass fraction below 75 µm."""
    return max(0.08, (0.45 - 1.2 * (sf - 0.55)) / flow_trait)


def work_hardening(sf: float) -> float:
    return 1.0 - 0.9 * max(sf - 0.55, 0.0)


def lube_factor(lube_min: float, ff_ratio: float) -> float:
    """Tablet strength lost to lubricant: long blending, or an overdriven feed frame."""
    return 1.0 - 0.035 * max(lube_min - 2.0, 0.0) - 0.12 * max(ff_ratio - 0.85, 0.0)


def hardness(force: float, sf: float, lube_min: float, ff_ratio: float, trait: float) -> float:
    """Tablet breaking force, N."""
    return (
        220.0 * trait * work_hardening(sf) * (1.0 - math.exp(-force / 12.0))
        * lube_factor(lube_min, ff_ratio)
    )  # fmt: skip


def weight_rsd(fines_frac: float, turret: float, ff_ratio: float) -> float:
    """Tablet weight RSD, %."""
    return 1.3 + 4.0 * fines_frac**2 * (turret / 45.0) ** 1.5 + 4.0 * (ff_ratio - 0.72) ** 2


def normal_tail(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def weight_reject(rsd: float) -> float:
    """Fraction outside the checkweigher band."""
    return 2.0 * normal_tail(100.0 * REJECT_LIMIT / max(rsd, 1e-3))


def capping(force: float, turret: float, sf: float) -> float:
    """Fraction of tablets that cap or laminate."""
    return min(0.5, 0.001 + 0.008 * math.exp((force - 17.0) / 2.0) * (turret / 45.0) ** 2
               * (1.0 + 4.0 * max(sf - 0.66, 0.0)))  # fmt: skip


def friability(h: float) -> float:
    """Friability, % (spec ≤ 1.0)."""
    return 0.8 * math.exp(-(h - 60.0) / 40.0)


def chipping(h: float) -> float:
    """Fraction rejected for chips and breaks: soft tablets."""
    return 0.05 * math.exp(-(h - 80.0) / 12.0)


def sticking(lube_min: float, buildup: float) -> float:
    """Fraction lost to sticking and picking: too little lubricant, or product on punches."""
    return min(0.3, 0.004 * math.exp(-(lube_min - 2.0) / 0.8) + 4.0e-5 * buildup)


def ejection(lube_min: float, force: float, buildup: float) -> float:
    return 180.0 + 140.0 * math.exp(-(lube_min - 2.0) / 1.5) + 4.0 * force + buildup


def dissolution(h: float, lube_min: float, api_d50: float) -> float:
    """Q at 30 min, % dissolved (spec ≥ 80)."""
    return 97.0 - 0.12 * (h - 100.0) - 3.0 * (lube_min - 2.0) - 0.04 * (api_d50 - 250.0)
