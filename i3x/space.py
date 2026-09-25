"""The i3X address space: namespaces, types, objects and relationships (ADR-0016).

Pure: `build(catalog)` turns plain rows (read from Neo4j by `i3x.catalog`) into an
`AddressSpace`. No driver, no broker, no clock.

Shape:

    pharmaco > site > area > line > unit                      HasParent / HasChildren
    unit > equipment modules > control modules > tags         HasComponent / ComponentOf
    unit > sensors, state/*, lab/*, ai/* data points          HasComponent / ComponentOf
    batches > batch > alerts, operator actions, recommendations    HasParent / HasChildren
    recipes > recipe,  phase-classes > phase class            HasParent / HasChildren
    RanOn, FollowsRecipe, ConcernsTag, ActedOn, Measures, Controls, Monitors   graph

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
from common.models import Levers, PhaseClass, format_ts
from common.uns import UnitPath

NS_I3X = "https://cesmii.org/i3x"
NS_PHARMA = "urn:pharmaco:pharma-4"
NAMESPACES = [
    {"uri": NS_I3X, "displayName": "i3X"},
    {"uri": NS_PHARMA, "displayName": "pharma-4 bioreactor POC"},
]

HAS_PARENT, HAS_CHILDREN = "HasParent", "HasChildren"
HAS_COMPONENT, COMPONENT_OF = "HasComponent", "ComponentOf"
RAN_ON, FOLLOWS_RECIPE, CONCERNS_TAG = "RanOn", "FollowsRecipe", "ConcernsTag"
ACTED_ON, MEASURES, CONTROLS, MONITORS = "ActedOn", "Measures", "Controls", "Monitors"

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
LEVERS = _object({k: NUM for k in Levers.model_fields})
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
TYPES: dict[str, ObjectType] = {
    t.element_id: t
    for t in [
        ObjectType("EnterpriseType", "Enterprise", "Enterprise", _NAMED),
        ObjectType("SiteType", "Site", "Site", _NAMED),
        ObjectType("AreaType", "Area", "Area", _NAMED),
        ObjectType("LineType", "Line", "Line", _NAMED),
        ObjectType(
            "BioreactorType",
            "Bioreactor (ISA-88 unit)",
            "Equipment",
            _object(
                {
                    "equipmentType": STR,
                    "workingVolumeL": NUM,
                    "batch": _null("string"),
                    "operation": _null("string"),
                }
            ),
        ),
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
            "Predicted final titer",
            "ai",
            _object({"titer": QUANTILES, "batchDay": NUM, "modelVersion": STR}),
        ),
        ObjectType(
            "YieldRecommendationType",
            "Open setpoint recommendation",
            "ai",
            _object(
                {
                    "id": STR,
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
                    "unit": STR,
                    "recipe": STR,
                    "campaign": STR,
                    "status": STR,
                    "start": STR,
                    "end": _null("string"),
                    "endReason": _null("string"),
                    "plannedLevers": LEVERS,
                    "actualLevers": LEVERS,
                    "outcome": _object(
                        {
                            "titer": _null("number"),
                            "peakVcd": _null("number"),
                            "viability": _null("number"),
                            "harvestDay": _null("number"),
                            "disposition": _null("string"),
                        }
                    ),
                    "operations": _array(OPERATION_RUN),
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

    site: dict[str, Any]
    area: dict[str, Any]
    line: dict[str, Any]
    units: list[dict[str, Any]]
    modules: list[dict[str, Any]]  # equipment modules and control modules
    tags: list[dict[str, Any]]
    sensors: list[dict[str, Any]]
    bindings: list[dict[str, Any]]
    recipes: list[dict[str, Any]]
    batches: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)


# Lab results the simulator publishes, with their units (simulator/engine.py LAB_UNITS).
LAB_UNITS: dict[str, str] = {
    "vcd": "1e6 cells/mL",
    "viability": "%",
    "glucose": "g/L",
    "lactate": "g/L",
    "ph_offline": "pH",
    "titer": "g/L",
}


def _ts(value: datetime | None) -> str | None:
    return format_ts(value) if value is not None else None


def build(cat: Catalog) -> AddressSpace:
    space = AddressSpace()
    site, area, line = cat.site["id"], cat.area["id"], cat.line["id"]
    enterprise = uns.ENTERPRISE
    paths = [
        (enterprise, enterprise, "EnterpriseType", None),
        (f"{enterprise}/{site}", cat.site["name"], "SiteType", enterprise),
        (f"{enterprise}/{site}/{area}", cat.area["name"], "AreaType", f"{enterprise}/{site}"),
        (
            f"{enterprise}/{site}/{area}/{line}",
            cat.line["name"],
            "LineType",
            f"{enterprise}/{site}/{area}",
        ),
    ]
    for eid, name, type_id, parent in paths:
        space.add(Obj(eid, name, type_id, value={"name": name}), parent)
    line_id = paths[-1][0]

    units: dict[str, UnitPath] = {}
    for u in cat.units:
        if u["type"] != "Bioreactor":
            raise ValueError(f"{u['id']}: only bioreactors are modelled, got {u['type']!r}")
        unit = UnitPath(site, area, line, u["id"])
        units[u["id"]] = unit
        space.add(
            Obj(
                unit.prefix,
                u["id"],
                "BioreactorType",
                description=f"{u['id']}, {u['working_volume_l']:g} L fed-batch bioreactor",
                value={"equipmentType": u["type"], "workingVolumeL": u["working_volume_l"]},
                live={"batch": uns.state_batch(unit), "operation": uns.state_operation(unit)},
            ),
            line_id,
        )
        _add_data_points(space, unit)

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
    phases = sorted({b["phase"] for b in cat.bindings} | {p.value for p in PhaseClass})
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
            value=_folder("Every batch run in the suite, oldest first", len(cat.batches)),
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
        if b["cell"] in units:
            space.link(b["id"], RAN_ON, units[b["cell"]].prefix)
        space.link(b["id"], FOLLOWS_RECIPE, f"recipe/{b['recipe']}")

    _add_events(space, cat)
    return space


def _folder(description: str, count: int) -> dict[str, Any]:
    return {"description": description, "count": count}


def _add_data_points(space: AddressSpace, unit: UnitPath) -> None:
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
                uns.state_phase(unit, p.value),
                f"{cell} {p.value}",
                "PhaseStateType",
                f"State of the {p.value} phase (phases run in parallel)",
            )
            for p in PhaseClass
        ],
        *[
            (
                uns.lab(unit, name),
                f"{cell} {name} (lab)",
                "LabResultType",
                f"Daily lab {name} in {u}",
            )
            for name, u in LAB_UNITS.items()
        ],
        (
            uns.ai_score(unit),
            f"{cell} anomaly index",
            "AnomalyScoreType",
            "Anomaly index; 1.0 is the alert threshold (advisory)",
        ),
        (
            uns.ai_prediction(unit),
            f"{cell} titer prediction",
            "YieldPredictionType",
            "Predicted final titer (g/L), P10/P50/P90 (advisory)",
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
        "unit": b["cell"],
        "recipe": b["recipe"],
        "campaign": b["campaign"],
        "status": b["status"],
        "start": _ts(b["start"]),
        "end": _ts(b["end"]),
        "endReason": b.get("end_reason"),
        "plannedLevers": b["planned"],
        "actualLevers": b["actual"],
        "outcome": {
            "titer": outcome.get("titer"),
            "peakVcd": outcome.get("peak_vcd"),
            "viability": outcome.get("viability"),
            "harvestDay": outcome.get("harvest_day"),
            "disposition": outcome.get("disposition"),
        },
        "operations": _operations(b["operations"]),
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
