"""Sensor model: the gap between the true state and what the DCS reads.

Two views of every measured variable:

- `control(name, true)` is what the control loops act on: true value plus any offset
  (probe drift), or the frozen value of a stuck sensor. No random noise, so the hot
  loop stays cheap and deterministic.
- `sample(name, true, rng)` is the published reading: the control view plus Gaussian
  noise. Every analog PV gets noise (never zero), so a healthy sensor never repeats an
  exact value; that is what the stuck-sensor rule relies on (ADR-0009, Q8).

Totalizers are quantised, not noisy: a real FQI reads flat between additions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Noise standard deviation per analog variable, in engineering units.
NOISE: dict[str, float] = {
    "temp": 0.02,
    "ph": 0.004,
    "do": 0.3,
    "agitation": 0.2,
    "air": 0.004,
    "o2": 0.003,
    "co2": 0.002,
    "pressure": 0.3,
    "weight": 0.4,
    "spare_rtd": 0.05,
}

# Totalizer resolution.
QUANTUM: dict[str, float] = {"feed_total": 0.01, "base_total": 0.1}


@dataclass(slots=True)
class SensorBank:
    offsets: dict[str, float] = field(default_factory=dict)
    stuck: dict[str, float] = field(default_factory=dict)  # name -> frozen reading
    last: dict[str, float] = field(default_factory=dict)  # last published reading
    _noise: list[float] = field(default_factory=list)  # pre-drawn standard normals

    def _normal(self, rng: np.random.Generator) -> float:
        if not self._noise:
            self._noise = rng.standard_normal(4096).tolist()
        return self._noise.pop()

    def control(self, name: str, true: float) -> float:
        frozen = self.stuck.get(name)
        if frozen is not None:
            return frozen
        return true + self.offsets.get(name, 0.0)

    def sample(self, name: str, true: float, rng: np.random.Generator) -> float:
        frozen = self.stuck.get(name)
        if frozen is not None:
            value = frozen
        elif name in NOISE:
            value = true + self.offsets.get(name, 0.0) + NOISE[name] * self._normal(rng)
        elif name in QUANTUM:
            q = QUANTUM[name]
            value = round(true / q) * q
        else:
            value = true  # setpoints and other exact values
        self.last[name] = value
        return value

    def stick(self, name: str) -> None:
        """Freeze a sensor at its last published reading (bit-identical from now on)."""
        if name not in NOISE:
            raise ValueError(f"only analog PVs can stick, not {name!r}")
        if name in self.last:
            self.stuck[name] = self.last[name]

    def unstick(self, name: str) -> None:
        self.stuck.pop(name, None)
