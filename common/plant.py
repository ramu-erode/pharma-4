"""The plant model: enterprise, sites, lines, units and equipment classes (ADR-0018).

The only module that reads `config/plant.yaml`. Everything that needs a unit's ISA-95
path, its equipment class or its process asks here; nothing reads site/area/line from
settings any more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from common.uns import UnitPath

PLANT_FILE = Path(__file__).resolve().parents[1] / "config" / "plant.yaml"

PROCESSES = ("bioreactor", "api", "osd")
# Processes whose batch runs through every unit of its line in order (ADR-0018). The
# others run a whole batch on one unit.
TRAIN_PROCESSES = ("api", "osd")


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    name: str
    props: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Unit:
    path: UnitPath
    type: str  # the equipment type shown to people, e.g. Bioreactor
    cls: str  # the equipment class keying modules, tags and AI profiles, e.g. bioreactor
    process: str
    props: dict[str, Any]  # the rest of its config, e.g. working_volume_l

    @property
    def cell(self) -> str:
        return self.path.cell


@dataclass(frozen=True, slots=True)
class Line:
    site: Node
    area: Node
    line: Node
    process: str
    units: tuple[str, ...]  # cells, in train order

    @property
    def train(self) -> bool:
        return self.process in TRAIN_PROCESSES


@dataclass(frozen=True, slots=True)
class Plant:
    enterprise: Node
    sites: tuple[Node, ...]
    lines: tuple[Line, ...]
    units: dict[str, Unit]  # by cell
    classes: dict[str, dict[str, Any]]  # equipment class -> modules, sensors, phase classes

    def unit(self, cell: str) -> Unit:
        try:
            return self.units[cell]
        except KeyError:
            raise KeyError(f"no unit {cell!r} in the plant model") from None

    def path(self, cell: str) -> UnitPath:
        return self.unit(cell).path

    def cells(self, process: str | None = None, cls: str | None = None) -> list[str]:
        return [
            c
            for c, u in self.units.items()
            if (process is None or u.process == process) and (cls is None or u.cls == cls)
        ]

    def line_of(self, cell: str) -> Line:
        return next(ln for ln in self.lines if cell in ln.units)

    def train_of(self, cell: str) -> tuple[str, ...]:
        """The units a batch started on `cell` runs through, in order."""
        line = self.line_of(cell)
        return line.units if line.train else (cell,)

    def device_cells(self) -> dict[str, str]:
        """DCS device name -> cell, e.g. RX201 -> RX-201."""
        return {u.path.device: c for c, u in self.units.items()}


def load(path: Path = PLANT_FILE) -> Plant:
    raw = yaml.safe_load(path.read_text())
    ent = raw["enterprise"]
    sites, lines, units = [], [], {}
    for site_id, site in raw["sites"].items():
        site_node = Node(site_id, site["name"], _rest(site, "name", "areas"))
        sites.append(site_node)
        for area_id, area in site["areas"].items():
            area_node = Node(area_id, area["name"], _rest(area, "name", "lines"))
            for line_id, line in area["lines"].items():
                process = line["process"]
                if process not in PROCESSES:
                    raise ValueError(f"{line_id}: unknown process {process!r}")
                line_node = Node(line_id, line["name"])
                for cell, spec in line["units"].items():
                    if spec["class"] not in raw["equipment_classes"]:
                        raise ValueError(f"{cell}: unknown equipment class {spec['class']!r}")
                    if cell in units:
                        raise ValueError(f"unit {cell} defined twice")
                    units[cell] = Unit(
                        path=UnitPath(site_id, area_id, line_id, cell),
                        type=spec["type"],
                        cls=spec["class"],
                        process=process,
                        props=_rest(spec, "type", "class"),
                    )
                lines.append(Line(site_node, area_node, line_node, process, tuple(line["units"])))
    # Area and line ids are graph keys, so they must be unique across the enterprise.
    areas = {(ln.site.id, ln.area.id) for ln in lines}
    line_ids = [ln.line.id for ln in lines]
    if len({a for _, a in areas}) != len(areas) or len(set(line_ids)) != len(line_ids):
        raise ValueError("area and line ids must be unique across the enterprise")
    return Plant(
        enterprise=Node(ent["id"], ent["name"]),
        sites=tuple(sites),
        lines=tuple(lines),
        units=units,
        classes=raw["equipment_classes"],
    )


def _rest(d: dict[str, Any], *drop: str) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in drop}


@lru_cache(maxsize=1)
def get_plant() -> Plant:
    return load()
