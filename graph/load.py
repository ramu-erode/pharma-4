"""Load configuration into the graph: enterprise, sites, lines, units, modules, bindings,
tags, recipes (ADR-0011, ADR-0018).

Everything here is configuration: `config/plant.yaml` (through `common.plant`),
`graph/recipes/*.yaml` and the edge tag map (the same facts the edge adapter publishes
on `_meta/tags`). Statements are MERGEs, so loading twice changes nothing.
"""

from __future__ import annotations

from pathlib import Path

from common.plant import Plant, get_plant
from common.uns import TopicClass
from edge.core import TagMap
from graph.core import Stmt, cm_of_raw_tag, tag_statement
from simulator import recipes

SCHEMA_FILE = Path(__file__).with_name("schema.cypher")


def schema_statements() -> list[Stmt]:
    text = "\n".join(
        ln for ln in SCHEMA_FILE.read_text().splitlines() if not ln.strip().startswith("//")
    )
    return [(s.strip(), {}) for s in text.split(";") if s.strip()]


def plant_statements(tag_map: TagMap, plant: Plant | None = None) -> list[Stmt]:
    plant = plant or get_plant()
    ent = plant.enterprise
    out: list[Stmt] = [
        ("MERGE (e:Enterprise {id: $id}) SET e.name = $name", {"id": ent.id, "name": ent.name})
    ]
    for site in plant.sites:
        out.append(
            (
                "MATCH (e:Enterprise {id: $ent}) MERGE (s:Site {id: $id}) "
                "SET s.name = $name, s += $props MERGE (e)-[:HAS_SITE]->(s)",
                {"ent": ent.id, "id": site.id, "name": site.name, "props": site.props},
            )
        )
    for line in plant.lines:
        out.append(
            (
                "MATCH (s:Site {id: $site}) "
                "MERGE (a:Area {id: $area}) SET a.name = $area_name "
                "MERGE (l:Line {id: $line}) SET l.name = $line_name, l.process = $process "
                "MERGE (s)-[:HAS_AREA]->(a) MERGE (a)-[:HAS_LINE]->(l)",
                {
                    "site": line.site.id,
                    "area": line.area.id,
                    "area_name": line.area.name,
                    "line": line.line.id,
                    "line_name": line.line.name,
                    "process": line.process,
                },
            )
        )
        for position, cell in enumerate(line.units):
            out += _unit_statements(plant, line.line.id, cell, position)
    out += tag_statements(tag_map)
    for cell, unit in plant.units.items():
        cls = plant.classes[unit.cls]
        for sensor in cls["sensors"]:
            pv = [
                e.topic
                for e in tag_map.entries.values()
                if e.unit_path.cell == cell
                and cm_of_raw_tag(e.raw_tag) == sensor["control_module"]
                and e.cls is TopicClass.PV
            ]
            out.append(
                (
                    "MATCH (u:Equipment {id: $cell}) "
                    "MERGE (s:Sensor {id: $id}) SET s.model = $model, "
                    "s.calibration_interval_days = $interval "
                    "MERGE (u)-[:HAS_SENSOR]->(s) "
                    "WITH s UNWIND $topics AS topic MATCH (t:Tag {topic: topic}) "
                    "MERGE (s)-[:MEASURES]->(t)",
                    {
                        "cell": cell,
                        "id": f"{cell}/{sensor['id']}",
                        "model": sensor["model"],
                        "interval": sensor["calibration_interval_days"],
                        "topics": pv,
                    },
                )
            )
        for phase, bindings in cls["phase_classes"].items():
            out.append(("MERGE (:PhaseClass {name: $name})", {"name": phase}))
            for bnd in bindings:
                label = "EquipmentModule" if bnd["module"].startswith("EM-") else "ControlModule"
                out.append(
                    (
                        f"MATCH (pc:PhaseClass {{name: $phase}}), (m:{label} {{id: $mid}}) "
                        "MERGE (pc)-[r:BOUND_TO {alias: $alias, unit: $cell}]->(m) "
                        "SET r.role = $role",
                        {
                            "phase": phase,
                            "mid": f"{cell}/{bnd['module']}",
                            "alias": bnd["alias"],
                            "cell": cell,
                            "role": bnd["role"],
                        },
                    )
                )
    return out


def _unit_statements(plant: Plant, line: str, cell: str, position: int) -> list[Stmt]:
    """A unit and its modules. `position` is its place in the line (a train's order)."""
    unit = plant.unit(cell)
    out: list[Stmt] = [
        (
            "MATCH (l:Line {id: $line}) MERGE (u:Equipment {id: $cell}) "
            "SET u.type = $type, u.class = $cls, u.process = $process, u.path = $path, "
            "u.position = $position, u += $props "
            "MERGE (l)-[:HAS_UNIT]->(u)",
            {
                "line": line,
                "cell": cell,
                "type": unit.type,
                "cls": unit.cls,
                "process": unit.process,
                "path": unit.path.prefix,
                "position": position,
                "props": unit.props,
            },
        )
    ]
    for em_id, em in plant.classes[unit.cls]["equipment_modules"].items():
        out.append(
            (
                "MATCH (u:Equipment {id: $cell}) "
                "MERGE (em:EquipmentModule {id: $id}) SET em.name = $name, em.module = $em "
                "MERGE (u)-[:HAS_EM]->(em)",
                {"cell": cell, "id": f"{cell}/{em_id}", "name": em["name"], "em": em_id},
            )
        )
        for cm in em["control_modules"]:
            out.append(
                (
                    "MATCH (em:EquipmentModule {id: $em}) "
                    "MERGE (cm:ControlModule {id: $id}) SET cm.module = $cm "
                    "MERGE (em)-[:HAS_CM]->(cm)",
                    {"em": f"{cell}/{em_id}", "id": f"{cell}/{cm}", "cm": cm},
                )
            )
    return out


def tag_statements(tag_map: TagMap) -> list[Stmt]:
    """Tag nodes and their control modules: the facts `_meta/tags` carries live."""
    return [
        tag_statement(e.topic, e.raw_tag, e.unit, e.kind.value, e.unit_path.cell)
        for e in tag_map.entries.values()
    ]


def recipe_statements(tag_map: TagMap, plant: Plant | None = None) -> list[Stmt]:
    plant = plant or get_plant()
    out: list[Stmt] = []
    pv_topics: dict[tuple[str, str], list[str]] = {}  # (process, name) -> topics
    for e in tag_map.entries.values():
        if e.cls is TopicClass.PV:
            process = plant.unit(e.unit_path.cell).process
            pv_topics.setdefault((process, e.name), []).append(e.topic)
    for r in recipes.load_all().values():
        out.append(
            (
                "MERGE (r:Recipe {id: $id}) SET r.name = $name, r.version = $version, "
                "r.process = $process, r.effective_from = date($eff), r += $nominal",
                {
                    "id": r.id,
                    "process": r.process,
                    "name": r.name,
                    "version": r.version,
                    "eff": r.effective_from.isoformat(),
                    "nominal": r.nominal.model_dump(),
                },
            )
        )
        for param, (lo, hi) in r.par.items():
            out.append(
                (
                    "MATCH (r:Recipe {id: $recipe}) "
                    "MERGE (l:SpecLimit {id: $id}) "
                    "SET l.type = 'PAR', l.parameter = $param, l.low = $lo, l.high = $hi "
                    "MERGE (r)-[:HAS_LIMIT]->(l)",
                    {
                        "recipe": r.id,
                        "id": f"{r.id}/par/{param}",
                        "param": param,
                        "lo": lo,
                        "hi": hi,
                    },
                )
            )
        for name, limit in r.limits.items():
            for kind, (lo, hi) in (("action", limit.action), ("alarm", limit.alarm)):
                out.append(
                    (
                        "MATCH (r:Recipe {id: $recipe}) "
                        "MERGE (l:SpecLimit {id: $id}) "
                        "SET l.type = $kind, l.parameter = $name, l.low = $lo, l.high = $hi, "
                        "l.sp_relative = $rel, l.operations = $ops "
                        "MERGE (r)-[:HAS_LIMIT]->(l) "
                        "WITH l UNWIND $topics AS topic MATCH (t:Tag {topic: topic}) "
                        "MERGE (l)-[:FOR_TAG]->(t)",
                        {
                            "recipe": r.id,
                            "id": f"{r.id}/{name}/{kind}",
                            "kind": kind,
                            "name": name,
                            "lo": lo,
                            "hi": hi,
                            "rel": limit.sp_relative,
                            "ops": list(limit.operations) if limit.operations else None,
                            "topics": pv_topics.get((r.process, name), []),
                        },
                    )
                )
    return out


def config_statements(plant: Plant | None = None) -> list[Stmt]:
    plant = plant or get_plant()
    tag_map = TagMap.load(plant)
    return plant_statements(tag_map, plant) + recipe_statements(tag_map, plant)
