"""DCS-style control loops. They act on *measured* values, like a real DCS: a drifting
pH probe makes the pH loop hold the wrong true pH (architecture: pH probe drift).

Each loop is a small stateful object; `hold()` freezes its output, which is what a
HELD phase does (ADR-0011).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PI:
    kp: float
    ki: float
    lo: float
    hi: float
    integral: float = 0.0
    output: float = 0.0
    held: bool = False

    def update(self, error: float, dt: float) -> float:
        if self.held:
            return self.output
        candidate = self.integral + self.ki * error * dt
        out = self.kp * error + candidate
        # Conditional integration: stop integrating into saturation (anti-windup).
        if (
            self.lo < out < self.hi
            or (out >= self.hi and error < 0)
            or (out <= self.lo and error > 0)
        ):
            self.integral = candidate
        self.output = min(self.hi, max(self.lo, self.kp * error + self.integral))
        return self.output


@dataclass(slots=True)
class TemperatureLoop:
    """TIC-101: jacket temperature = SP + PI correction."""

    pi: PI = field(default_factory=lambda: PI(kp=2.0, ki=1.0, lo=-12.0, hi=12.0))

    def update(self, sp: float, measured: float, dt: float) -> float:
        return sp + self.pi.update(sp - measured, dt)


@dataclass(slots=True)
class PhLoop:
    """AIC-102 split range: output > 0 opens CO2 (acid side), < 0 runs the base pump."""

    pi: PI = field(default_factory=lambda: PI(kp=8.0, ki=4.0, lo=-1.0, hi=1.0))
    co2_max: float = 0.5  # L/min
    base_max: float = 2000.0  # mL/h

    def update(self, sp: float, measured: float, dt: float) -> tuple[float, float]:
        u = self.pi.update(measured - sp, dt)
        if u >= 0.0:
            return self.co2_max * u, 0.0
        return 0.0, -self.base_max * u


@dataclass(slots=True)
class DoCascade:
    """AIC-103 cascade: the first half of the output raises agitation (SIC-104) from
    its minimum to maximum, the second half opens O2 (FIC-107)."""

    pi: PI = field(default_factory=lambda: PI(kp=0.02, ki=0.04, lo=0.0, hi=1.0))
    agit_min: float = 80.0
    agit_max: float = 140.0
    o2_max: float = 1.0

    def update(self, sp: float, measured: float, dt: float) -> tuple[float, float]:
        u = self.pi.update(sp - measured, dt)
        if u <= 0.5:
            return self.agit_min + (self.agit_max - self.agit_min) * u / 0.5, 0.0
        return self.agit_max, self.o2_max * (u - 0.5) / 0.5


def air_flow(batch_day: float) -> float:
    """FIC-106 follows a fixed ramp with culture age (L/min), 0.1 → 2.0."""
    return min(2.0, max(0.1, 0.2 + 0.1 * batch_day))


def temperature_sp(
    batch_day: float,
    shift_day: float,
    prod_temp: float,
    t_growth: float = 36.5,
    ramp_h: float = 4.0,
) -> float:
    """Temperature setpoint profile: growth temperature, then a linear ramp to the
    production temperature starting on the shift day."""
    if batch_day <= shift_day:
        return t_growth
    frac = min(1.0, (batch_day - shift_day) * 24.0 / ramp_h)
    return t_growth + (prod_temp - t_growth) * frac
