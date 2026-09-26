"""The i3X address space: namespaces, types, objects and relationships (ADR-0016).

Pure: `build(catalog)` turns plain rows (read from Neo4j by `i3x.catalog`) into an
`AddressSpace`. No driver, no broker, no clock.

Shape (ADR-0016, ADR-0018, ADR-0020):

    pharmanextgen > site > area > line > unit                      HasParent / HasChildren
    unit > equipment modules > control modules > tags         HasComponent / ComponentOf
    unit > sensors, state/*, lab/*, ai/* data points          HasComponent / ComponentOf
    batches > batch > alerts, operator actions, recommendations    HasParent / HasChildren
    recipes > recipe,  phase-classes > phase class,  materials > lot    HasParent / HasChildren
    RanOn, FollowsRecipe, ConcernsTag, ActedOn, Measures, Controls, Monitors,
    Produced, Consumed                                                                graph

A data point's elementId is its UNS topic. Values come from the broker at read time
(`topic` is set); every other object carries a static `value` from the graph.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from common import models as m
from common import uns
from common.models import format_ts
from common.uns import UnitPath

NS_I3X = "https://cesmii.org/i3x"
NS_PHARMA = "urn:pharmanextgen:pharma-4"
NAMESPACES = [
    {"uri": NS_I3X, "displayName": "i3X"},
    {"uri": NS_PHARMA, "displayName": "pharma-4 PharmaNextGen POC"},
]

HAS_PARENT, HAS_CHILDREN = "HasParent", "HasChildren"
HAS_COMPONENT, COMPONENT_OF = "HasComponent", "ComponentOf"
RAN_ON, FOLLOWS_RECIPE, CONCERNS_TAG = "RanOn", "FollowsRecipe", "ConcernsTag"
ACTED_ON, MEASURES, CONTROLS, MONITORS = "ActedOn", "Measures", "Controls", "Monitors"
PRODUCED, CONSUMED = "Produced", "Consumed"

_REL_PAIRS = [
    (HAS_PARENT, HAS_CHILDREN, NS_I3X),
    (HAS_COMPONENT, COMPONENT_OF, NS_I3X),
    (RAN_ON, "HostedBatch", NS_PHARMA),
    (FOLLOWS_RECIPE, "FollowedBy", NS_PHARMA),
    (CONCERNS_TAG, "ConcernedBy", NS_PHARMA),
    (ACTED_ON, "ActedOnBy", NS_PHARMA),
    (MEASURES, "MeasuredBy", NS_PHARMA),
    (CONTROLS, "ControlledBy", NS_PHARMA),
    (MONITORS, "MonitoredBy", NS_PHARMA),
    (PRODUCED, "ProducedBy", NS_PHARMA),
    (CONSUMED, "ConsumedBy", NS_PHARMA),
]
REVERSE: dict[str, str] = {}
for _a, _b, _ in _REL_PAIRS:
    REVERSE[_a], REVERSE[_b] = _b, _a
RELATIONSHIP_TYPES = [
    {
        "elementId": rel,
        "displayName": rel,
        "namespaceUri": ns,
        "relationshipId": rel,
        "reverseOf": rev,
    }
    for a, b, ns in _REL_PAIRS
    for rel, rev in ((a, b), (b, a))
]

ROOT_BATCHES, ROOT_RECIPES, ROOT_PHASES = "batches", "recipes", "phase-classes"
ROOT_MATERIALS = "materials"

# --- object types ----------------------------------------------------------------------------

NUM: dict[str, Any] = {"type": "number"}
INT: dict[str, Any] = {"type": "integer"}
STR: dict[str, Any] = {"type": "string"}
BOOL: dict[str, Any] = {"type": "boolean"}


def _null(t: str) -> dict[str, Any]:
    return {"type": [t, "null"]}


def _object(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props)}


def _array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


QUANTILES = _object({"p10": NUM, "p50": NUM, "p90": NUM})
LEVERS = {"type": "object", "additionalProperties": NUM}  # the batch's process's levers
LEVER_ADVICE = {
    "type": "object",
    "additionalProperties": _object({"current": NUM, "recommended": NUM, "frozen": BOOL}),
}
PHASE_RUN = _object(
    {"phase": STR, "state": STR, "start": STR, "end": _null("string"), "holds": INT}
)
OPERATION_RUN = _object(
    {"name": STR, "start": STR, "end": _null("string"), "phases": _array(PHASE_RUN)}
)


@dataclass(frozen=True, slots=True)
class ObjectType:
    element_id: str
    display_name: str
    source_type_id: str  # the Neo4j label or UNS topic class the type is projected from
    schema: dict[str, Any]
    namespace: str = NS_PHARMA
    version: str = "1.0.0"

    def to_json(self) -> dict[str, Any]:
        return {
            "elementId": self.element_id,
            "displayName": self.display_name,
            "namespaceUri": self.namespace,
            "sourceTypeId": self.source_type_id,
            "version": self.version,
            "schema": self.schema,
        }


_NAMED = _object({"name": STR})


def _some(required: dict[str, Any], optional: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {**required, **optional},
        "required": list(required),
    }


UNIT_SCHEMA = _some(
    {
        "equipmentType": STR,
        "equipmentClass": STR,
        "process": STR,
        "batch": _null("string"),
        "operation": _null("string"),
    },
    {"workingVolumeL": NUM, "filterAreaM2": NUM, "rollWidthCm": NUM, "stations": INT},
)
# The equipment types in the plant model (config/plant.yaml), each an i3X object type.
UNIT_TYPES = {
    "Bioreactor": "Bioreactor (ISA-88 unit)",
    "ReactorCrystallizer": "Reactor-crystallizer (ISA-88 unit)",
    "FilterDryer": "Agitated filter-dryer (ISA-88 unit)",
    "BinBlender": "Bin blender (ISA-88 unit)",
    "RollerCompactor": "Roller compactor (ISA-88 unit)",
    "RotaryTabletPress": "Rotary tablet press (ISA-88 unit)",
}
OUTCOME = _some(
    {"disposition": _null("string")},
    {
        k: _null("number")
        for k in (
            "titer",
            "peakVcd",
            "viability",
            "harvestDay",
            "yieldPct",
            "assay",
            "freeSa",
            "related",
            "lod",
            "d50",
            "conversionIpc",
            "blendUniformity",
            "granuleD50",
            "bulkDensity",
            "dissolution",
            "hardness",
            "friability",
            "av",
        )
    },
)
TYPES: dict[str, ObjectType] = {
    t.element_id: t
    for t in [
        ObjectType("EnterpriseType", "Enterprise", "Enterprise", _NAMED),
        ObjectType(
            "SiteType", "Site", "Site", _some({"name": STR}, {"location": STR, "role": STR})
        ),
        ObjectType("AreaType", "Area", "Area", _NAMED),
        ObjectType("LineType", "Line", "Line", _some({"name": STR}, {"process": STR})),
        *[ObjectType(f"{t}Type", name, "Equipment", UNIT_SCHEMA) for t, name in UNIT_TYPES.items()],
        ObjectType(
            "EquipmentModuleType",
            "Equipment module",
            "EquipmentModule",
            _object({"module": STR, "name": STR}),
        ),
        ObjectType(
            "ControlModuleType", "Control module", "ControlModule", _object({"module": STR})
        ),
        ObjectType(
            "SensorType",
            "Sensor",
            "Sensor",
            _object({"model": STR, "calibrationIntervalDays": NUM}),
        ),
        ObjectType("ProcessValueType", "Process value", "pv", NUM),
        ObjectType("SetpointType", "Setpoint", "sp", NUM),
        ObjectType("LabResultType", "Offline lab result", "lab", NUM),
        ObjectType("BatchStateType", "Batch on the unit", "state", STR),
        ObjectType(
            "OperationStateType",
            "Current operation",
            "state",
            {"type": "string", "enum": [o.value for o in m.Operation]},
        ),
        ObjectType(
            "PhaseStateType",
            "Phase state",
            "state",
            {"type": "string", "enum": [s.value for s in m.PhaseState]},
        ),
        ObjectType("AnomalyScoreType", "Anomaly index (1.0 = threshold)", "ai", NUM),
        ObjectType(
            "YieldPredictionType",
            "Predicted batch outcome (titer, or yield)",
            "ai",
            _object({"target": STR, "value": QUANTILES, "batchDay": NUM, "modelVersion": STR}),
        ),
        ObjectType(
            "YieldRecommendationType",
            "Open setpoint recommendation",
            "ai",
            _object(
                {
                    "id": STR,
                    "target": STR,
                    "levers": LEVER_ADVICE,
                    "predictedCurrent": QUANTILES,
                    "predictedRecommended": QUANTILES,
                    "gain": QUANTILES,
                    "modelVersion": STR,
                }
            ),
        ),
        ObjectType("FolderType", "Folder", "Folder", _object({"description": STR, "count": INT})),
        ObjectType(
            "BatchType",
            "Batch",
            "Batch",
            _object(
                {
                    "batchId": STR,
                    "process": STR,
                    "unit": STR,
                    "units": _array(STR),
                    "recipe": STR,
                    "campaign": STR,
                    "status": STR,
                    "start": STR,
                    "end": _null("string"),
                    "endReason": _null("string"),
                    "plannedLevers": LEVERS,
                    "actualLevers": LEVERS,
                    "outcome": OUTCOME,
                    "operations": _array(OPERATION_RUN),
                    "lotProduced": _null("string"),
                    "lotsConsumed": _array(_object({"lot": STR, "quantityKg": NUM})),
                }
            ),
        ),
        ObjectType(
            "AnomalyAlertType",
            "Anomaly alert",
            "Event",
            _object(
                {
                    "key": STR,
                    "layer": STR,
                    "state": STR,
                    "score": NUM,
                    "threshold": NUM,
                    "faultClass": _null("string"),
                    "topTags": _array(STR),
                    "openedAt": STR,
                    "clearedAt": _null("string"),
                }
            ),
        ),
        ObjectType(
            "OperatorActionType",
            "Operator setpoint change",
            "Event",
            _object(
                {
                    "operator": STR,
                    "parameter": STR,
                    "old": NUM,
                    "new": NUM,
                    "reason": _null("string"),
                }
            ),
        ),
        ObjectType(
            "RecommendationRecordType",
            "Recommendation (as issued)",
            "Recommendation",
            _object({"modelVersion": STR, "gain": QUANTILES, "levers": LEVER_ADVICE}),
        ),
        ObjectType(
            "RecipeType",
            "Master recipe",
            "Recipe",
            _object(
                {
                    "name": STR,
                    "process": STR,
                    "version": INT,
                    "effectiveFrom": STR,
                    "nominalLevers": LEVERS,
                    "limits": _array(
                        _object(
                            {
                                "parameter": STR,
                                "type": STR,
                                "low": NUM,
                                "high": NUM,
                                "spRelative": _null("boolean"),
                            }
                        )
                    ),
                }
            ),
        ),
        ObjectType(
            "MaterialLotType",
            "Material lot",
            "MaterialLot",
            _object(
                {
                    "lot": STR,
                    "material": STR,
                    "quantityKg": _null("number"),
                    "producedBy": _null("string"),
                    "consumedBy": _array(_object({"batch": STR, "quantityKg": NUM})),
                }
            ),
        ),
        ObjectType(
            "PhaseClassType",
            "Phase class",
            "PhaseClass",
            _object({"name": STR, "bindings": _array(_object({"module": STR, "role": STR}))}),
        ),
    ]
}


# --- objects -----------------------------------------------------------------------------------


@dataclass(slots=True)
class Obj:
    element_id: str
    display_name: str
    type_id: str
    parent_id: str | None = None
    description: str | None = None
    topic: str | None = None  # a data point: its value is the latest message on this topic
    value: Any = None  # everything else: a static value from the graph
    ts: datetime | None = None  # when `value` was recorded; None = the plant's current time
    live: dict[str, str] = field(default_factory=dict)  # value field -> topic, merged at read
    system: dict[str, Any] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)  # outgoing, hierarchy included

    @property
    def components(self) -> list[str]:
        return self.edges.get(HAS_COMPONENT, [])

    @property
    def is_composition(self) -> bool:
        return bool(self.components)

    def to_json(self, metadata: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "elementId": self.element_id,
            "displayName": self.display_name,
            "typeElementId": self.type_id,
            "parentId": self.parent_id,
            "isComposition": self.is_composition,
            "isExtended": False,
        }
        if metadata:
            t = TYPES[self.type_id]
            out["metadata"] = {
                "description": self.description,
                "typeNamespaceUri": t.namespace,
                "sourceTypeId": t.source_type_id,
                "relationships": {k: list(v) for k, v in self.edges.items() if v},
                "system": self.system or None,
            }
        return out


class AddressSpace:
    def __init__(self) -> None:
        self.objects: dict[str, Obj] = {}
        self._by_topic: dict[str, list[str]] = {}

    def add(self, obj: Obj, parent: str | None = None, composed: bool = False) -> Obj:
        if obj.element_id in self.objects:
            raise ValueError(f"duplicate elementId {obj.element_id!r}")
        if obj.type_id not in TYPES:
            raise ValueError(f"{obj.element_id}: unknown type {obj.type_id!r}")
        self.objects[obj.element_id] = obj
        for topic in {obj.topic, *obj.live.values()} - {None}:
            self._by_topic.setdefault(topic, []).append(obj.element_id)
        if parent is not None:
            obj.parent_id = parent
            self.link(parent, HAS_COMPONENT if composed else HAS_CHILDREN, obj.element_id)
        return obj

    def link(self, source: str, rel: str, target: str) -> None:
        """Add an edge and its reverse (i3X stores every relationship both ways)."""
        if source not in self.objects or target not in self.objects:
            return  # an edge to something outside the address space is not an edge
        for a, r, b in ((source, rel, target), (target, REVERSE[rel], source)):
            targets = self.objects[a].edges.setdefault(r, [])
            if b not in targets:
                targets.append(b)

    def get(self, element_id: str) -> Obj | None:
        return self.objects.get(element_id)

    def affected_by(self, topic: str) -> list[str]:
        """Objects whose value changes when a message arrives on `topic`."""
        return self._by_topic.get(topic, [])

    def roots(self) -> list[Obj]:
        return [o for o in self.objects.values() if o.parent_id is None]

    def related(self, element_id: str, rel: str | None = None) -> list[tuple[str, Obj]]:
        obj = self.objects[element_id]
        return [
            (r, self.objects[t])
            for r, targets in obj.edges.items()
            if rel is None or r == rel
            for t in targets
        ]

    def descendants(self, element_id: str, depth: int) -> list[str]:
        """The element plus its components, `depth` levels deep (0 = all), per maxDepth."""
        out, frontier, level = [element_id], [element_id], 1
        while frontier and (depth == 0 or level < depth):
            frontier = [c for e in frontier for c in self.objects[e].components]
            out += frontier
            level += 1
        return out


# --- building from the graph ----------------------------------------------------------------------


@dataclass
class Catalog:
    """Plain rows read from Neo4j; see `i3x.catalog` for the queries. Datetimes are aware."""

    enterprise: dict[str, Any]
    sites: list[dict[str, Any]]  # id, name, location, role
    lines: list[dict[str, Any]]  # site, area {id, name}, line {id, name, process}
    units: list[dict[str, Any]]  # id, line, type, class, process, position, and properties
    modules: list[dict[str, Any]]  # equipment modules and control modules
    tags: list[dict[str, Any]]
    sensors: list[dict[str, Any]]
    bindings: list[dict[str, Any]]
    recipes: list[dict[str, Any]]
    batches: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    lots: list[dict[str, Any]] = field(default_factory=list)


# Lab results the simulator publishes on each equipment class, with their units (the
# engines' LAB_UNITS; tests/i3x/test_space.py keeps the two in step).
LAB_UNITS: dict[str, dict[str, str]] = {
    "bioreactor": {
        "vcd": "1e6 cells/mL",
        "viability": "%",
        "glucose": "g/L",
        "lactate": "g/L",
        "ph_offline": "pH",
        "titer": "g/L",
    },
    "reactor": {"conversion_ipc": "%"},
    "filter_dryer": {
        "assay": "%",
        "free_sa": "%",
        "related": "%",
        "lod": "%",
        "d50": "µm",
        "yield": "%",
    },
    "blender": {"blend_uniformity": "%"},
    "roller_compactor": {"granule_d50": "µm", "bulk_density": "g/mL"},
    "tablet_press": {
        "assay": "%",
        "av": "",
        "dissolution": "%",
        "hardness": "N",
        "friability": "%",
        "free_sa": "%",
        "yield": "%",
    },
}
SCORED_CLASSES = ("bioreactor", "reactor", "filter_dryer", "roller_compactor", "tablet_press")


def _ts(value: datetime | None) -> str | None:
    return format_ts(value) if value is not None else None


def build(cat: Catalog) -> AddressSpace:
    space = AddressSpace()
    enterprise = uns.ENTERPRISE
    space.add(
        Obj(
            enterprise,
            cat.enterprise.get("name", enterprise),
            "EnterpriseType",
            value={"name": cat.enterprise.get("name", enterprise)},
        )
    )
    for site in cat.sites:
        sid = f"{enterprise}/{site['id']}"
        value = {"name": site["name"], **{k: site[k] for k in ("location", "role") if site.get(k)}}
        space.add(
            Obj(sid, site["name"], "SiteType", description=site.get("role"), value=value),
            enterprise,
        )
    line_ids: dict[str, str] = {}  # line id -> elementId
    for ln in cat.lines:
        area_id = f"{enterprise}/{ln['site']}/{ln['area']['id']}"
        if area_id not in space.objects:
            name = ln["area"]["name"]
            space.add(
                Obj(area_id, name, "AreaType", value={"name": name}), f"{enterprise}/{ln['site']}"
            )
        lid = f"{area_id}/{ln['line']['id']}"
        value = {"name": ln["line"]["name"], "process": ln["line"].get("process")}
        space.add(Obj(lid, ln["line"]["name"], "LineType", value=value), area_id)
        line_ids[ln["line"]["id"]] = lid

    phases_of: dict[str, set[str]] = {}
    for b in cat.bindings:
        phases_of.setdefault(b["module"].split("/")[0], set()).add(b["phase"])
    units: dict[str, UnitPath] = {}
    for u in cat.units:
        if u["type"] not in UNIT_TYPES:
            raise ValueError(f"{u['id']}: unknown equipment type {u['type']!r}")
        line_id = line_ids[u["line"]]
        site, area, line = line_id.split("/")[1:4]
        unit = UnitPath(site, area, line, u["id"])
        units[u["id"]] = unit
        props = {
            key: u[prop]
            for key, prop in (
                ("workingVolumeL", "working_volume_l"),
                ("filterAreaM2", "filter_area_m2"),
                ("rollWidthCm", "roll_width_cm"),
                ("stations", "stations"),
            )
            if u.get(prop) is not None
        }
        home = u["process"] == "bioreactor" or u.get("position", 0) == 0
        space.add(
            Obj(
                unit.prefix,
                u["id"],
                f"{u['type']}Type",
                description=f"{u['id']}, {UNIT_TYPES[u['type']].split(' (')[0].lower()} "
                f"({u['process']} process)",
                value={
                    "equipmentType": u["type"],
                    "equipmentClass": u["class"],
                    "process": u["process"],
                    **props,
                },
                live={"batch": uns.state_batch(unit), "operation": uns.state_operation(unit)},
            ),
            line_id,
        )
        _add_data_points(space, unit, u["class"], sorted(phases_of.get(u["id"], ())), home)

    # graph module id (BR-101/EM-PH) -> elementId (…/BR-101/EM-PH, …/BR-101/EM-PH/AIC-102)
    module_ids: dict[str, str] = {}
    for mod in sorted(cat.modules, key=lambda r: r["parent"] is not None):
        unit = units[mod["unit"]]
        if mod["parent"] is None:
            eid, parent = f"{unit.prefix}/{mod['module']}", unit.prefix
            obj = Obj(
                eid,
                f"{mod['unit']} {mod['module']}",
                "EquipmentModuleType",
                description=mod["name"],
                value={"module": mod["module"], "name": mod["name"]},
            )
        else:
            parent = module_ids[mod["parent"]]
            eid = f"{parent}/{mod['module']}"
            obj = Obj(
                eid,
                f"{mod['unit']} {mod['module']}",
                "ControlModuleType",
                value={"module": mod["module"]},
            )
        module_ids[mod["id"]] = eid
        space.add(obj, parent, composed=True)

    for t in cat.tags:
        kind = m.TagKind(t["kind"])
        name = uns.parse(t["topic"]).name
        what = "process value" if kind is m.TagKind.PV else "setpoint"
        space.add(
            Obj(
                t["topic"],
                f"{t['cell']} {name} {kind.value}",
                "ProcessValueType" if kind is m.TagKind.PV else "SetpointType",
                description=f"{name} {what} in {t['unit']}; DCS tag {t['raw_tag']}",
                topic=t["topic"],
                system={"unsTopic": t["topic"], "rawTag": t["raw_tag"], "unit": t["unit"]},
            ),
            module_ids[t["cm"]],
            composed=True,
        )

    for s in cat.sensors:
        unit = units[s["unit"]]
        sid = f"{unit.prefix}/{s['id'].split('/')[-1]}"
        space.add(
            Obj(
                sid,
                s["id"].replace("/", " "),
                "SensorType",
                description=s["model"],
                value={
                    "model": s["model"],
                    "calibrationIntervalDays": s["calibration_interval_days"],
                },
            ),
            unit.prefix,
            composed=True,
        )
        for topic in s["measures"]:
            space.link(sid, MEASURES, topic)

    space.add(
        Obj(ROOT_PHASES, "Phase classes", "FolderType", value=_folder("ISA-88 phase classes", 0))
    )
    phases = sorted({b["phase"] for b in cat.bindings})
    for phase in phases:
        bound = [b for b in cat.bindings if b["phase"] == phase and b["module"] in module_ids]
        pid = f"phase-class/{phase}"
        space.add(
            Obj(
                pid,
                phase,
                "PhaseClassType",
                value={
                    "name": phase,
                    "bindings": [
                        {"module": module_ids[b["module"]], "role": b["role"]} for b in bound
                    ],
                },
            ),
            ROOT_PHASES,
        )
        for b in bound:
            space.link(
                pid, CONTROLS if b["role"] == "control" else MONITORS, module_ids[b["module"]]
            )
    space.objects[ROOT_PHASES].value = _folder("ISA-88 phase classes", len(phases))

    space.add(
        Obj(
            ROOT_RECIPES, "Recipes", "FolderType", value=_folder("Master recipes", len(cat.recipes))
        )
    )
    for r in cat.recipes:
        space.add(
            Obj(
                f"recipe/{r['id']}",
                f"{r['name']} {r['id']}",
                "RecipeType",
                value={
                    "name": r["name"],
                    "process": r.get("process", "bioreactor"),
                    "version": r["version"],
                    "effectiveFrom": r["effective_from"],
                    "nominalLevers": r["nominal"],
                    "limits": r["limits"],
                },
            ),
            ROOT_RECIPES,
        )

    space.add(
        Obj(
            ROOT_BATCHES,
            "Batches",
            "FolderType",
            value=_folder("Every batch run at every site, oldest first", len(cat.batches)),
        )
    )
    for b in sorted(cat.batches, key=lambda r: r["id"]):
        space.add(
            Obj(
                b["id"],
                b["id"],
                "BatchType",
                description=f"{b['recipe']} batch on {b['cell']} ({b['campaign']}), {b['status']}",
                value=_batch_value(b),
                ts=b["end"],
            ),
            ROOT_BATCHES,
        )
        for cell in b.get("cells") or [b["cell"]]:
            if cell in units:
                space.link(b["id"], RAN_ON, units[cell].prefix)
        space.link(b["id"], FOLLOWS_RECIPE, f"recipe/{b['recipe']}")

    _add_events(space, cat)
    _add_lots(space, cat)
    return space


def _add_lots(space: AddressSpace, cat: Catalog) -> None:
    """Material lots and their genealogy (ADR-0020)."""
    space.add(
        Obj(
            ROOT_MATERIALS,
            "Materials",
            "FolderType",
            value=_folder("Material lots: what each batch made and used", len(cat.lots)),
        )
    )
    for lot in sorted(cat.lots, key=lambda r: r["id"]):
        lid = f"lot/{lot['id']}"
        consumed = [c for c in lot.get("consumed_by", []) if c.get("batch")]
        space.add(
            Obj(
                lid,
                f"Lot {lot['id']}",
                "MaterialLotType",
                description=f"{lot['material']} lot {lot['id']}",
                value={
                    "lot": lot["id"],
                    "material": lot["material"],
                    "quantityKg": lot.get("quantity_kg"),
                    "producedBy": lot.get("produced_by"),
                    "consumedBy": [
                        {"batch": c["batch"], "quantityKg": c["kg"]}
                        for c in sorted(consumed, key=lambda c: c["batch"])
                    ],
                },
            ),
            ROOT_MATERIALS,
        )
        if lot.get("produced_by"):
            space.link(lot["produced_by"], PRODUCED, lid)
        for c in consumed:
            space.link(c["batch"], CONSUMED, lid)


def _folder(description: str, count: int) -> dict[str, Any]:
    return {"description": description, "count": count}


def _add_data_points(
    space: AddressSpace, unit: UnitPath, cls: str, phases: list[str], home: bool
) -> None:
    """A unit's state, lab and AI data points. AI points exist where the services publish:
    anomaly scores on scored classes, predictions and advice on a batch's home unit."""
    cell = unit.cell
    points: list[tuple[str, str, str, str]] = [
        (uns.state_batch(unit), f"{cell} batch", "BatchStateType", "Batch running on the unit"),
        (
            uns.state_operation(unit),
            f"{cell} operation",
            "OperationStateType",
            "Current ISA-88 operation (sequential)",
        ),
        *[
            (
                uns.state_phase(unit, p),
                f"{cell} {p}",
                "PhaseStateType",
                f"State of the {p} phase (phases run in parallel)",
            )
            for p in phases
        ],
        *[
            (
                uns.lab(unit, name),
                f"{cell} {name} (lab)",
                "LabResultType",
                f"Lab {name}" + (f" in {u}" if u else ""),
            )
            for name, u in LAB_UNITS.get(cls, {}).items()
        ],
    ]
    if cls in SCORED_CLASSES:
        points.append(
            (
                uns.ai_score(unit),
                f"{cell} anomaly index",
                "AnomalyScoreType",
                "Anomaly index; 1.0 is the alert threshold (advisory)",
            )
        )
    if home:
        what = "final titer (g/L)" if cls == "bioreactor" else "batch yield (%)"
        points += [
            (
                uns.ai_prediction(unit),
                f"{cell} yield prediction",
                "YieldPredictionType",
                f"Predicted {what}, P10/P50/P90 (advisory)",
            ),
            (
                uns.ai_recommendation(unit),
                f"{cell} recommendation",
                "YieldRecommendationType",
                "Open setpoint recommendation; no value when none is open (advisory)",
            ),
        ]
    for topic, name, type_id, description in points:
        space.add(
            Obj(
                topic,
                name,
                type_id,
                description=description,
                topic=topic,
                system={"unsTopic": topic},
            ),
            unit.prefix,
            composed=True,
        )


def _batch_value(b: dict[str, Any]) -> dict[str, Any]:
    outcome = b.get("outcome") or {}
    return {
        "batchId": b["id"],
        "process": b.get("process", "bioreactor"),
        "unit": b["cell"],
        "units": sorted(b.get("cells") or [b["cell"]]),
        "recipe": b["recipe"],
        "campaign": b["campaign"],
        "status": b["status"],
        "start": _ts(b["start"]),
        "end": _ts(b["end"]),
        "endReason": b.get("end_reason"),
        "plannedLevers": b["planned"],
        "actualLevers": b["actual"],
        "outcome": {
            "disposition": outcome.get("disposition"),
            **{
                key: outcome[prop]
                for key, prop in OUTCOME_PROPS.items()
                if outcome.get(prop) is not None
            },
        },
        "operations": _operations(b["operations"]),
        "lotProduced": b.get("produced"),
        "lotsConsumed": [
            {"lot": c["lot"], "quantityKg": c["kg"]}
            for c in sorted(b.get("consumed") or [], key=lambda c: c["lot"])
            if c.get("lot")
        ],
    }


# i3X outcome key -> the graph's Outcome property (graph.core.OUTCOME_LAB and friends).
OUTCOME_PROPS = {
    "titer": "titer",
    "peakVcd": "peak_vcd",
    "viability": "viability",
    "harvestDay": "harvest_day",
    "yieldPct": "yield_pct",
    "assay": "assay",
    "freeSa": "free_sa",
    "related": "related",
    "lod": "lod",
    "d50": "d50",
    "conversionIpc": "conversion_ipc",
    "blendUniformity": "blend_uniformity",
    "granuleD50": "granule_d50",
    "bulkDensity": "bulk_density",
    "dissolution": "dissolution",
    "hardness": "hardness",
    "friability": "friability",
    "av": "av",
}


def _operations(ops: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": op["name"],
            "start": _ts(op["start"]),
            "end": _ts(op["end"]),
            "phases": [
                {
                    "phase": ph["phase"],
                    "state": ph["state"],
                    "start": _ts(ph["start"]),
                    "end": _ts(ph["end"]),
                    "holds": ph["holds"],
                }
                for ph in sorted(op["phases"], key=lambda p: p["phase"])
            ],
        }
        for op in sorted(ops, key=lambda o: o["start"])
    ]


def _add_events(space: AddressSpace, cat: Catalog) -> None:
    for a in cat.alerts:
        space.add(
            Obj(
                a["id"],
                f"{a['id']} {a['key']}",
                "AnomalyAlertType",
                description=f"{a['layer']} alert {a['key']} on batch {a['batch']}",
                value={
                    "key": a["key"],
                    "layer": a["layer"],
                    "state": a["state"],
                    "score": a["score"],
                    "threshold": a["threshold"],
                    "faultClass": a["fault_class"],
                    "topTags": list(a["top_tags"] or []),
                    "openedAt": _ts(a["opened_at"]),
                    "clearedAt": _ts(a["cleared_at"]),
                },
                ts=a["cleared_at"] or a["opened_at"],
            ),
            a["batch"] if a["batch"] in space.objects else ROOT_BATCHES,
        )
        for topic in a["tags"]:
            space.link(a["id"], CONCERNS_TAG, topic)

    for r in cat.recommendations:
        space.add(
            Obj(
                r["id"],
                r["id"],
                "RecommendationRecordType",
                description=f"Setpoint recommendation for batch {r['batch']} (advisory)",
                value={
                    "modelVersion": r["model_version"],
                    "gain": {"p10": r["gain_p10"], "p50": r["gain_p50"], "p90": r["gain_p90"]},
                    "levers": json.loads(r["levers"]),
                },
                ts=r["ts"],
            ),
            r["batch"] if r["batch"] in space.objects else ROOT_BATCHES,
        )

    for e in cat.actions:
        space.add(
            Obj(
                e["id"],
                f"{e['parameter']} {e['old']:g} → {e['new']:g}",
                "OperatorActionType",
                description=(
                    f"Operator {e['operator']} changed {e['parameter']} on batch {e['batch']}"
                ),
                value={
                    "operator": e["operator"],
                    "parameter": e["parameter"],
                    "old": e["old"],
                    "new": e["new"],
                    "reason": e["reason"],
                },
                ts=e["ts"],
            ),
            e["batch"] if e["batch"] in space.objects else ROOT_BATCHES,
        )
        for topic in e["tags"]:
            space.link(e["id"], CONCERNS_TAG, topic)
        if e["recommendation"]:
            space.link(e["id"], ACTED_ON, e["recommendation"])
