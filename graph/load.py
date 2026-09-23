"""Load configuration into the graph: plant hierarchy, modules, bindings, tags, recipes.

Everything here is configuration (ADR-0011): `config/plant.yaml`, `graph/recipes/*.yaml`
and the edge tag map (the same facts the edge adapter publishes on `_meta/tags`).
Statements are MERGEs, so loading twice changes nothing.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from common.uns import TopicClass, UnitPath
from edge.core import TagMap
from graph.core import Stmt, cm_of_raw_tag, tag_statement
from simulator import recipes

PLANT_FILE = Path(__file__).resolve().parents[1] / "config" / "plant.yaml"
SCHEMA_FILE = Path(__file__).with_name("schema.cypher")


def schema_statements() -> list[Stmt]:
    text = "\n".join(
        ln for ln in SCHEMA_FILE.read_text().splitlines() if not ln.strip().startswith("//")
    )
    return [(s.strip(), {}) for s in text.split(";") if s.strip()]


def plant_statements(tag_map: TagMap, plant_file: Path = PLANT_FILE) -> list[Stmt]:
    plant = yaml.safe_load(plant_file.read_text())
    site, area, line = plant["site"], plant["area"], plant["line"]
    out: list[Stmt] = [
        (
            "MERGE (s:Site {id: $site}) SET s.name = $site_name "
            "MERGE (a:Area {id: $area}) SET a.name = $area_name "
            "MERGE (l:Line {id: $line}) SET l.name = $line_name "
            "MERGE (s)-[:HAS_AREA]->(a) MERGE (a)-[:HAS_LINE]->(l)",
            {
                "site": site["id"],
                "site_name": site["name"],
                "area": area["id"],
                "area_name": area["name"],
                "line": line["id"],
                "line_name": line["name"],
            },
        )
    ]
    for cell, spec in plant["units"].items():
        path = UnitPath(site["id"], area["id"], line["id"], cell)
        out.append(
            (
                "MATCH (l:Line {id: $line}) MERGE (u:Equipment {id: $cell}) "
                "SET u.type = $type, u.path = $path, u.working_volume_l = $vol "
                "MERGE (l)-[:HAS_UNIT]->(u)",
                {
                    "line": line["id"],
                    "cell": cell,
                    "type": spec["type"],
                    "path": path.prefix,
                    "vol": spec["working_volume_l"],
                },
            )
        )
        for em_id, em in plant["equipment_modules"].items():
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
    out += tag_statements(tag_map)
    for cell in plant["units"]:
        for sensor in plant["sensors"]:
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
        for phase, bindings in plant["phase_classes"].items():
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


def tag_statements(tag_map: TagMap) -> list[Stmt]:
    """Tag nodes and their control modules: the facts `_meta/tags` carries live."""
    return [
        tag_statement(e.topic, e.raw_tag, e.unit, e.kind.value, e.unit_path.cell)
        for e in tag_map.entries.values()
    ]


def recipe_statements(tag_map: TagMap) -> list[Stmt]:
    out: list[Stmt] = []
    pv_topics: dict[str, list[str]] = {}
    for e in tag_map.entries.values():
        if e.cls is TopicClass.PV:
            pv_topics.setdefault(e.name, []).append(e.topic)
    for r in recipes.load_all().values():
        out.append(
            (
                "MERGE (r:Recipe {id: $id}) SET r.name = $name, r.version = $version, "
                "r.effective_from = date($eff), r += $nominal",
                {
                    "id": r.id,
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
                        "l.sp_relative = $rel "
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
                            "topics": pv_topics.get(name, []),
                        },
                    )
                )
    return out


def config_statements(site: str, area: str, line: str) -> list[Stmt]:
    tag_map = TagMap.load(site, area, line)
    return plant_statements(tag_map) + recipe_statements(tag_map)
