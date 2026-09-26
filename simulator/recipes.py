"""Recipes (graph/recipes/*.yaml): nominal levers, PARs, jitter and spec limits.

The simulator and graph-sync read the same files. Recipes are configuration, so they
are loaded from disk rather than queried from the graph. Each recipe belongs to one
process (ADR-0018); its levers are that process's lever model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from common.models import LEVER_MODELS

RECIPE_DIR = Path(__file__).resolve().parents[1] / "graph" / "recipes"
LEVERS_OF: dict[str, tuple[str, ...]] = {p: tuple(m.model_fields) for p, m in LEVER_MODELS.items()}
LEVER_NAMES = LEVERS_OF["bioreactor"]


@dataclass(frozen=True, slots=True)
class Limit:
    sp_relative: bool
    action: tuple[float, float]
    alarm: tuple[float, float]
    operations: tuple[str, ...] | None = None  # None: whenever a batch is on the unit

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
    nominal: BaseModel  # the process's lever model (common.models.LEVER_MODELS)
    par: dict[str, tuple[float, float]]
    jitter: dict[str, float]
    limits: dict[str, Limit]
    process: str = "bioreactor"

    @property
    def levers_model(self) -> type[BaseModel]:
        return LEVER_MODELS[self.process]

    def clip(self, levers: BaseModel) -> BaseModel:
        """Clamp levers into the PARs."""
        data = levers.model_dump()
        for k, (lo, hi) in self.par.items():
            data[k] = min(hi, max(lo, data[k]))
        return self.levers_model(**data)


def _load(path: Path) -> Recipe:
    raw = yaml.safe_load(path.read_text())
    process = raw.get("process", "bioreactor")
    par = {k: (float(v[0]), float(v[1])) for k, v in raw["par"].items()}
    if set(par) != set(LEVERS_OF[process]):
        raise ValueError(f"{path.name}: PARs must cover exactly {LEVERS_OF[process]}")
    return Recipe(
        id=raw["id"],
        name=raw["name"],
        version=int(raw["version"]),
        effective_from=raw["effective_from"],
        nominal=LEVER_MODELS[process](**raw["nominal"]),
        par=par,
        jitter={k: float(v) for k, v in raw["jitter"].items()},
        limits={
            k: Limit(
                sp_relative=bool(v["sp_relative"]),
                action=(float(v["action"][0]), float(v["action"][1])),
                alarm=(float(v["alarm"][0]), float(v["alarm"][1])),
                operations=tuple(v["operations"]) if "operations" in v else None,
            )
            for k, v in raw["limits"].items()
        },
        process=process,
    )


@lru_cache(maxsize=1)
def load_all() -> dict[str, Recipe]:
    recipes = {r.id: r for r in (_load(p) for p in sorted(RECIPE_DIR.glob("*.yaml")))}
    if not recipes:
        raise FileNotFoundError(f"no recipes in {RECIPE_DIR}")
    return recipes


def get(recipe_id: str) -> Recipe:
    return load_all()[recipe_id]


def current(on: date, process: str = "bioreactor") -> Recipe:
    """The process's recipe in effect on a date (latest `effective_from` not after it)."""
    eligible = [r for r in load_all().values() if r.process == process and r.effective_from <= on]
    if not eligible:
        raise LookupError(f"no {process} recipe effective on {on}")
    return max(eligible, key=lambda r: r.effective_from)


def of_process(process: str) -> list[Recipe]:
    return sorted((r for r in load_all().values() if r.process == process), key=lambda r: r.version)
