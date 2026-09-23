"""Simulated plant clock (ADR-0009).

- Monotonic: starts at max(wall-clock now, the latest simulated time already on the
  broker), so a restart never sends `ts` backwards.
- Speed is changed at runtime (pause, 1x … 3600x) and can pause itself once a unit's
  batch reaches a given day ("run to day N").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SimClock:
    now: datetime
    speed: float = 3600.0
    paused: bool = False
    run_to_day: float | None = None
    run_to_cell: str | None = None

    @classmethod
    def starting_at(cls, wall_now: datetime, seen: list[datetime], speed: float) -> SimClock:
        """The monotonic start rule."""
        return cls(now=max([wall_now, *seen]), speed=speed)

    def advance(self, wall_elapsed_s: float) -> timedelta:
        """Move simulated time for `wall_elapsed_s` real seconds; return the step taken."""
        if self.paused or wall_elapsed_s <= 0:
            return timedelta(0)
        step = timedelta(seconds=wall_elapsed_s * self.speed)
        self.now += step
        return step

    def set_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive; use pause")
        self.speed = speed

    def arm_run_to_day(self, cell: str, day: float) -> None:
        self.run_to_cell, self.run_to_day = cell, day
        self.paused = False

    def check_run_to_day(self, cell: str, batch_day: float) -> bool:
        """Pause if the armed unit reached its target day. Returns True when it fired."""
        if (
            self.run_to_day is not None
            and cell == self.run_to_cell
            and batch_day >= self.run_to_day
        ):
            self.paused = True
            self.run_to_day = self.run_to_cell = None
            return True
        return False
