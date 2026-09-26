"""Wire payload models (ADR-0002, ADR-0013).

Every UNS and `_sim` payload has the same six keys: `v, ts, unit, q, batch, src`.
`v` is typed per topic class; `model_for(topic)` says which model a topic carries.
The one exception is `edge/raw`, which mimics a DCS and carries a flat `RawSample`.

Pydantic is for the wire only. Inside service cores and on the backfill hot path,
data moves as plain dataclasses (see the implementation plan's design rules).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
)

from common import uns
from common.uns import SimCommand, TopicClass, TopicKind

# --- primitives -------------------------------------------------------------------------

BATCH_ID = re.compile(r"^B\d{4}-\d{4}$")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def format_ts(value: datetime) -> str:
    """ISO-8601 UTC with milliseconds and a Z suffix: 2026-09-22T10:15:05.000Z."""
    return _utc(value).strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"


UtcDatetime = Annotated[
    datetime,
    AfterValidator(_utc),
    PlainSerializer(format_ts, return_type=str, when_used="json"),
]


def _batch_id(value: str) -> str:
    if not BATCH_ID.match(value):
        raise ValueError(f"batch id must look like B2026-0142, got {value!r}")
    return value


BatchId = Annotated[str, AfterValidator(_batch_id)]


class Quality(StrEnum):
    GOOD = "GOOD"
    UNCERTAIN = "UNCERTAIN"
    BAD = "BAD"


class Src(StrEnum):
    """Who produced a payload. `operator` is a person acting through the dashboard or CLI.

    ADR-0013 names the data producers; the remaining services appear only as the
    source of their own `_meta/<service>/status` heartbeat.
    """

    SIM = "sim"
    EDGE = "edge"
    ANOMALY = "anomaly"
    YIELD = "yield"
    OPERATOR = "operator"
    HISTORIAN = "historian"
    GRAPH_SYNC = "graph-sync"
    DASHBOARD = "dashboard"
    I3X = "i3x"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Payload[V](_Model):
    v: V
    ts: UtcDatetime
    unit: str | None
    q: Quality = Quality.GOOD
    batch: BatchId | None
    src: Src


# --- domain enums -----------------------------------------------------------------------


class Operation(StrEnum):
    """Sequential ISA-88 operations, per unit (ADR-0011, ADR-0018). IDLE means no batch
    on the unit. Names are unique within a process, so `<batch>/<operation>` is a key."""

    IDLE = "Idle"
    # bioreactor
    SETUP = "Setup"
    INOCULATION = "Inoculation"
    GROWTH = "Growth"
    TEMP_SHIFT = "TempShift"
    PRODUCTION = "Production"
    HARVEST = "Harvest"
    # API: reactor-crystallizer, then filter-dryer (ADR-0019)
    CHARGE = "Charge"
    REACTION = "Reaction"
    CRYSTALLIZATION = "Crystallization"
    TRANSFER = "Transfer"
    FILTRATION = "Filtration"
    WASHING = "Washing"
    DRYING = "Drying"
    DISCHARGE = "Discharge"
    # OSD: blender (Charge, ..., Discharge), roller compactor, tablet press
    BLENDING = "Blending"
    LUBRICATION = "Lubrication"
    COMPACTION = "Compaction"
    COMPRESSION = "Compression"


class PhaseClass(StrEnum):
    """Parallel ISA-88 phases (ADR-0011). Topic form is the lowercase value."""

    TEMP_CTRL = "TEMP_CTRL"
    PH_CTRL = "PH_CTRL"
    DO_CTRL = "DO_CTRL"
    FEED_ADD = "FEED_ADD"
    AGIT_CTRL = "AGIT_CTRL"
    DOSE_ADD = "DOSE_ADD"
    FILTER_CTRL = "FILTER_CTRL"
    VAC_CTRL = "VAC_CTRL"
    BLEND_CTRL = "BLEND_CTRL"
    COMPACT_CTRL = "COMPACT_CTRL"
    MILL_CTRL = "MILL_CTRL"
    TABLET_CTRL = "TABLET_CTRL"
    FEED_CTRL = "FEED_CTRL"


class PhaseState(StrEnum):
    RUNNING = "RUNNING"
    HELD = "HELD"
    COMPLETE = "COMPLETE"


class BatchStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"
    EARLY_HARVEST = "EARLY_HARVEST"


class Campaign(StrEnum):
    MFG = "MFG"
    PC = "PC"  # process characterisation (DoE across the PARs)


class FaultType(StrEnum):
    PH_PROBE_DRIFT = "ph_probe_drift"
    DO_SPARGER_FOULING = "do_sparger_fouling"
    TEMP_CONTROL_LOSS = "temp_control_loss"
    FEED_PUMP_FAILURE = "feed_pump_failure"
    STUCK_SENSOR = "stuck_sensor"
    CONTAMINATION = "contamination"
    # API (ADR-0019)
    JACKET_FOULING = "jacket_fouling"
    DOSING_METER_DRIFT = "dosing_meter_drift"
    AGITATOR_DEGRADATION = "agitator_degradation"
    FILTER_BLINDING = "filter_blinding"
    VACUUM_LEAK = "vacuum_leak"
    # OSD (ADR-0019)
    ROLL_FORCE_DRIFT = "roll_force_drift"
    PUNCH_STICKING = "punch_sticking"
    HOPPER_BRIDGING = "hopper_bridging"
    HVAC_HUMIDITY = "hvac_humidity"


class TagKind(StrEnum):
    PV = "PV"
    SP = "SP"


# --- structured v types -----------------------------------------------------------------


class Levers(_Model):
    """The bioreactor optimizer's decision variables (architecture: setpoint optimization)."""

    shift_day: float = Field(ge=0)
    prod_temp: float
    ph_sp: float
    do_sp: float
    feed_mult: float = Field(gt=0)


class ApiLevers(_Model):
    """Aspirin API levers (ADR-0019)."""

    rxn_temp: float  # °C, reaction hold
    ac2o_ratio: float = Field(gt=0)  # mol acetic anhydride per mol salicylic acid
    rxn_time: float = Field(gt=0)  # h, hold after dosing
    cool_rate: float = Field(gt=0)  # °C/h, crystallisation ramp
    dry_temp: float  # °C, dryer jacket


class OsdLevers(_Model):
    """Aspirin tablet levers (ADR-0019)."""

    lube_time: float = Field(gt=0)  # min, blending with stearic acid
    roll_force: float = Field(gt=0)  # kN/cm, specific roll force
    comp_force: float = Field(gt=0)  # kN, main compression force
    turret_speed: float = Field(gt=0)  # rpm
    feed_frame: float = Field(gt=0)  # rpm


# Each process's levers; the field sets are disjoint, so the union validates unambiguously.
AnyLevers = Levers | ApiLevers | OsdLevers
LEVER_MODELS: dict[str, type[_Model]] = {"bioreactor": Levers, "api": ApiLevers, "osd": OsdLevers}


class BatchStarted(_Model):
    kind: Literal["BATCH_START"] = "BATCH_START"
    batch_id: BatchId
    recipe: str
    campaign: Campaign
    planned_levers: AnyLevers
    process: Literal["bioreactor", "api", "osd"] = "bioreactor"


class OperationChanged(_Model):
    kind: Literal["OPERATION_CHANGE"] = "OPERATION_CHANGE"
    batch_id: BatchId
    previous: Operation
    current: Operation


class PhaseChanged(_Model):
    kind: Literal["PHASE_CHANGE"] = "PHASE_CHANGE"
    batch_id: BatchId
    operation: Operation
    phase: PhaseClass
    state: PhaseState


class BatchEnded(_Model):
    kind: Literal["BATCH_END"] = "BATCH_END"
    batch_id: BatchId
    status: BatchStatus
    reason: str | None = None
    # From the CoA, where the process has one (ADR-0019); None: rejected only if aborted.
    disposition: Literal["ACCEPTED", "REJECTED"] | None = None


BatchEvent = Annotated[
    BatchStarted | OperationChanged | PhaseChanged | BatchEnded, Field(discriminator="kind")
]


class MaterialProduced(_Model):
    """A batch put a lot into stock (ADR-0020). The lot id is the batch id."""

    kind: Literal["MATERIAL_PRODUCED"] = "MATERIAL_PRODUCED"
    batch_id: BatchId
    lot: BatchId
    material: str
    quantity_kg: float = Field(ge=0)


class MaterialConsumed(_Model):
    """A batch drew on a lot (ADR-0020)."""

    kind: Literal["MATERIAL_CONSUMED"] = "MATERIAL_CONSUMED"
    batch_id: BatchId
    lot: BatchId
    material: str
    quantity_kg: float = Field(gt=0)


MaterialEvent = Annotated[MaterialProduced | MaterialConsumed, Field(discriminator="kind")]


class OperatorEvent(_Model):
    action: Literal["SETPOINT_CHANGE"] = "SETPOINT_CHANGE"
    operator: str
    parameter: str  # UNS name under sp/, e.g. "temperature"
    old: float
    new: float
    recommendation_id: str | None = None
    reason: str | None = None


class Alert(_Model):
    id: str
    key: str
    state: Literal["OPEN", "CLEARED"]
    layer: Literal["rules", "stats", "mspc", "iforest"]
    score: float
    threshold: float
    fault_class: FaultType | None = None
    top_tags: list[str] = Field(default_factory=list, max_length=3)
    opened_at: UtcDatetime
    cleared_at: UtcDatetime | None = None


class Quantiles(_Model):
    p10: float
    p50: float
    p90: float


class Prediction(_Model):
    """ai/yield/prediction (ADR-0021): the process's target, e.g. titer or api_yield.
    `batch_day` counts from the batch's alignment reference (inoculation for the
    bioreactor, batch start otherwise)."""

    target: str
    value: Quantiles
    batch_day: float
    model_version: str


class LeverAdvice(_Model):
    current: float
    recommended: float
    frozen: bool


class Recommendation(_Model):
    id: str
    target: str
    levers: dict[str, LeverAdvice]
    predicted_current: Quantiles
    predicted_recommended: Quantiles
    gain: Quantiles
    model_version: str


class TagMeta(_Model):
    raw_tag: str
    topic: str
    unit: str
    kind: TagKind
    deadband: float = Field(ge=0)


class Heartbeat(_Model):
    """`_meta/<service>/status`. The payload's `ts` is the last simulated time the
    service processed; `wall` is wall-clock time, which is what liveness is judged on
    (ADR-0009)."""

    service: str
    state: Literal["online", "offline"]
    wall: UtcDatetime


class FaultLabel(_Model):
    """Ground truth (ADR-0012). Published at onset with end=None, again at end."""

    id: str
    fault: FaultType
    cell: str
    batch_id: BatchId
    onset: UtcDatetime
    end: UtcDatetime | None = None
    params: dict[str, float | str] = Field(default_factory=dict)


class ClockStatus(_Model):
    """`_sim/clock`, retained. Lets a restarted simulator keep time and batch ids
    monotonic (ADR-0009) without reading any database."""

    sim_time: UtcDatetime
    speed: float
    paused: bool
    run_to_day: float | None = None
    run_to_cell: str | None = None
    last_batch_seq: int | None = None


class LotStock(_Model):
    """One lot in a site's stock (ADR-0020). `properties` are its *true* attributes,
    which drive the process that consumes it; they live only under `_sim`."""

    lot: BatchId
    material: str
    quantity_kg: float = Field(ge=0)
    released: UtcDatetime
    properties: dict[str, float] = Field(default_factory=dict)


class Inventory(_Model):
    """`_sim/inventory` and `_sim/opening_stock`, retained: Freiburg's API stock."""

    site: str
    lots: list[LotStock] = Field(default_factory=list)


# --- _sim/cmd ---------------------------------------------------------------------------


class BatchCommand(_Model):
    action: Literal["start", "abort"]
    cell: str  # a train's batch starts on its first unit
    recipe: str | None = None
    campaign: Campaign = Campaign.MFG
    levers: AnyLevers | None = None  # overrides the recipe's nominal levers


class FaultCommand(_Model):
    action: Literal["inject", "clear"]
    cell: str
    fault: FaultType
    params: dict[str, float | str] = Field(default_factory=dict)


class ClockCommand(_Model):
    action: Literal["pause", "resume", "speed", "run_to_day"]
    speed: float | None = Field(default=None, gt=0)
    day: float | None = Field(default=None, ge=0)
    cell: str | None = None  # run_to_day is measured on this unit's batch


class SetpointCommand(_Model):
    cell: str
    parameter: str
    value: float
    operator: str
    recommendation_id: str | None = None
    reason: str | None = None


# --- edge/raw ---------------------------------------------------------------------------


class RawSample(_Model):
    """Flat DCS-style sample on edge/raw; deliberately not the six-key envelope.

    `q` is the DCS status bit. The edge adapter carries it into the UNS `q` field.
    """

    tag: str
    value: float
    t: UtcDatetime
    q: Quality = Quality.GOOD


# --- topic -> model ---------------------------------------------------------------------

ScalarPayload = Payload[float]
BatchStatePayload = Payload[BatchId | None]
OperationPayload = Payload[Operation]
PhaseStatePayload = Payload[PhaseState]
BatchEventPayload = Payload[BatchEvent]
MaterialEventPayload = Payload[MaterialEvent]
OperatorEventPayload = Payload[OperatorEvent]
AlertPayload = Payload[Alert]
PredictionPayload = Payload[Prediction]
RecommendationPayload = Payload[Recommendation]
TagMetaPayload = Payload[TagMeta]
HeartbeatPayload = Payload[Heartbeat]
FaultLabelPayload = Payload[FaultLabel]
ClockStatusPayload = Payload[ClockStatus]
InventoryPayload = Payload[Inventory]

SIM_COMMAND_MODELS: dict[SimCommand, type[BaseModel]] = {
    SimCommand.BATCH: Payload[BatchCommand],
    SimCommand.FAULT: Payload[FaultCommand],
    SimCommand.CLOCK: Payload[ClockCommand],
    SimCommand.SETPOINT: Payload[SetpointCommand],
}


class UnknownTopicError(LookupError):
    pass


def model_for(topic: str) -> type[BaseModel]:
    """The payload model a topic carries. Raises for topics with no defined model."""
    p = uns.parse(topic)
    match p.kind:
        case TopicKind.UNS:
            assert p.cls is not None and p.name is not None
            name = p.name
            if p.cls in (TopicClass.PV, TopicClass.SP, TopicClass.LAB):
                return ScalarPayload
            if p.cls is TopicClass.STATE:
                if name == "batch":
                    return BatchStatePayload
                if name == "operation":
                    return OperationPayload
                if name.startswith("phase/"):
                    return PhaseStatePayload
            if p.cls is TopicClass.EVENTS:
                if name == "batch":
                    return BatchEventPayload
                if name == "operator":
                    return OperatorEventPayload
                if name == "material":
                    return MaterialEventPayload
            if p.cls is TopicClass.AI:
                if name == "anomaly/score":
                    return ScalarPayload
                if name.startswith("anomaly/alert/"):
                    return AlertPayload
                if name == "yield/prediction":
                    return PredictionPayload
                if name == "yield/recommendation":
                    return RecommendationPayload
        case TopicKind.META_TAG:
            return TagMetaPayload
        case TopicKind.META_STATUS:
            return HeartbeatPayload
        case TopicKind.EDGE_RAW:
            return RawSample
        case TopicKind.EDGE_UNMAPPED:
            return RawSample
        case TopicKind.SIM_CMD:
            assert p.command is not None
            return SIM_COMMAND_MODELS[p.command]
        case TopicKind.SIM_CLOCK:
            return ClockStatusPayload
        case TopicKind.SIM_INVENTORY | TopicKind.SIM_OPENING_STOCK:
            return InventoryPayload
        case TopicKind.SIM_FAULTS:
            return FaultLabelPayload
    raise UnknownTopicError(topic)


def encode(model: BaseModel) -> bytes:
    return model.model_dump_json().encode()


def decode(topic: str, data: bytes) -> BaseModel | None:
    """Decode a message for `topic`. An empty payload is a retained-message clear -> None."""
    if not data:
        return None
    return model_for(topic).model_validate_json(data)


def now_utc() -> datetime:
    return datetime.now(UTC)
