"""Recipes (graph/recipes/*.yaml): nominal levers, PARs, jitter and spec limits.

The simulator and graph-sync read the same files. Recipes are configuration, so they
are loaded from disk rather than queried from the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml

from common.models import Levers

RECIPE_DIR = Path(__file__).resolve().parents[1] / "graph" / "recipes"
LEVER_NAMES = ("shift_day", "prod_temp", "ph_sp", "do_sp", "feed_mult")


@dataclass(frozen=True, slots=True)
class Limit:
    sp_relative: bool
    action: tuple[float, float]
    alarm: tuple[float, float]

    def action_band(self, sp: float | None) -> tuple[float, float]:
        lo, hi = self.action
        if self.sp_relative:
            if sp is None:
                raise ValueError("SP-relative limit needs the setpoint")
            return sp + lo, sp + hi
        return lo, hi


@dataclass(frozen=True, slots=True)
class Recipe:
    id: str
    name: str
    version: int
    effective_from: date
    nominal: Levers
    par: dict[str, tuple[float, float]]
    jitter: dict[str, float]
    limits: dict[str, Limit]

    def clip(self, levers: Levers) -> Levers:
        """Clamp levers into the PARs."""
        data = levers.model_dump()
        for k, (lo, hi) in self.par.items():
            data[k] = min(hi, max(lo, data[k]))
        return Levers(**data)


def _load(path: Path) -> Recipe:
    raw = yaml.safe_load(path.read_text())
    par = {k: (float(v[0]), float(v[1])) for k, v in raw["par"].items()}
    if set(par) != set(LEVER_NAMES):
        raise ValueError(f"{path.name}: PARs must cover exactly {LEVER_NAMES}")
    return Recipe(
        id=raw["id"],
        name=raw["name"],
        version=int(raw["version"]),
        effective_from=raw["effective_from"],
        nominal=Levers(**raw["nominal"]),
        par=par,
        jitter={k: float(v) for k, v in raw["jitter"].items()},
        limits={
            k: Limit(
                sp_relative=bool(v["sp_relative"]),
                action=(float(v["action"][0]), float(v["action"][1])),
                alarm=(float(v["alarm"][0]), float(v["alarm"][1])),
            )
            for k, v in raw["limits"].items()
        },
    )


@lru_cache(maxsize=1)
def load_all() -> dict[str, Recipe]:
    recipes = {r.id: r for r in (_load(p) for p in sorted(RECIPE_DIR.glob("*.yaml")))}
    if not recipes:
        raise FileNotFoundError(f"no recipes in {RECIPE_DIR}")
    return recipes


def get(recipe_id: str) -> Recipe:
    return load_all()[recipe_id]


def current(on: date) -> Recipe:
    """The recipe in effect on a date (latest `effective_from` not after it)."""
    eligible = [r for r in load_all().values() if r.effective_from <= on]
    if not eligible:
        raise LookupError(f"no recipe effective on {on}")
    return max(eligible, key=lambda r: r.effective_from)
