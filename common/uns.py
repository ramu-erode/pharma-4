"""UNS topic construction, parsing and delivery policy.

Every topic string in the codebase comes from this module (CLAUDE.md). The tree is
ISA-95 (ADR-0001):

    pharmaco/<site>/<area>/<line>/<cell>/<class>/<name...>

plus three branches with their own shapes:

    pharmaco/_meta/tags/<cell>/<class>/<name>     tag metadata (edge adapter)
    pharmaco/_meta/<service>/status               heartbeat + last-will (every service)
    edge/raw/<device>/<raw tag>, edge/unmapped    outside the UNS (ADR-0005)
    _sim/cmd/<kind>, _sim/clock, _sim/faults/<cell>   outside the UNS (ADR-0012)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

ENTERPRISE = "pharmaco"
META = "_meta"
EDGE = "edge"
SIM = "_sim"

_NAME = re.compile(r"^[a-z0-9_]+$")
_ALERT_KEY = re.compile(r"^[a-z0-9_]+-[a-z0-9_]+$")
_CELL = re.compile(r"^[A-Z]{2}-\d{3}$")
_PATH_SEGMENT = re.compile(r"^[a-z0-9-]+$")
_SERVICE = re.compile(r"^[a-z0-9-]+$")
_DEVICE = re.compile(r"^[A-Z]{2}\d{3}$")
_RAW_TAG = re.compile(r"^[A-Z]{2}\d{3}\.[A-Z]{2,4}-\d{3}\.(PV|SP)$")


class TopicClass(StrEnum):
    PV = "pv"
    SP = "sp"
    LAB = "lab"
    STATE = "state"
    EVENTS = "events"
    AI = "ai"


class SimCommand(StrEnum):
    BATCH = "batch"
    FAULT = "fault"
    CLOCK = "clock"
    SETPOINT = "setpoint"


class TopicKind(StrEnum):
    UNS = "uns"
    META_TAG = "meta_tag"
    META_STATUS = "meta_status"
    EDGE_RAW = "edge_raw"
    EDGE_UNMAPPED = "edge_unmapped"
    SIM_CMD = "sim_cmd"
    SIM_CLOCK = "sim_clock"
    SIM_FAULTS = "sim_faults"


class TopicError(ValueError):
    """A topic or topic part does not follow the UNS rules."""


def _check(pattern: re.Pattern[str], value: str, what: str) -> str:
    if not pattern.match(value):
        raise TopicError(f"invalid {what}: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class UnitPath:
    """The ISA-95 path of one unit (cell), e.g. chennai/upstream/suite-1/BR-101."""

    site: str
    area: str
    line: str
    cell: str

    def __post_init__(self) -> None:
        _check(_PATH_SEGMENT, self.site, "site")
        _check(_PATH_SEGMENT, self.area, "area")
        _check(_PATH_SEGMENT, self.line, "line")
        _check(_CELL, self.cell, "cell")

    @property
    def prefix(self) -> str:
        return f"{ENTERPRISE}/{self.site}/{self.area}/{self.line}/{self.cell}"

    @property
    def device(self) -> str:
        """The DCS device name for this unit: BR-101 -> BR101."""
        return self.cell.replace("-", "")


# --- builders: UNS equipment branch -------------------------------------------------


def _uns(unit: UnitPath, cls: TopicClass, *names: str) -> str:
    for n in names:
        _check(_NAME, n, "name")
    return "/".join((unit.prefix, cls.value, *names))


def pv(unit: UnitPath, name: str) -> str:
    return _uns(unit, TopicClass.PV, name)


def sp(unit: UnitPath, name: str) -> str:
    return _uns(unit, TopicClass.SP, name)


def lab(unit: UnitPath, name: str) -> str:
    return _uns(unit, TopicClass.LAB, name)


def state_batch(unit: UnitPath) -> str:
    return _uns(unit, TopicClass.STATE, "batch")


def state_operation(unit: UnitPath) -> str:
    return _uns(unit, TopicClass.STATE, "operation")


def state_phase(unit: UnitPath, phase: str) -> str:
    """state/phase/<phase>; accepts the phase class name (TEMP_CTRL) or its topic form."""
    return _uns(unit, TopicClass.STATE, "phase", phase.lower())


def events(unit: UnitPath, name: str) -> str:
    return _uns(unit, TopicClass.EVENTS, name)


def ai_score(unit: UnitPath) -> str:
    return _uns(unit, TopicClass.AI, "anomaly", "score")


def ai_alert(unit: UnitPath, key: str) -> str:
    """ai/anomaly/alert/<layer>-<tag|class> (ADR-0013)."""
    _check(_ALERT_KEY, key, "alert key")
    return f"{unit.prefix}/ai/anomaly/alert/{key}"


def ai_prediction(unit: UnitPath) -> str:
    return _uns(unit, TopicClass.AI, "yield", "prediction")


def ai_recommendation(unit: UnitPath) -> str:
    return _uns(unit, TopicClass.AI, "yield", "recommendation")


# --- builders: other branches ---------------------------------------------------------


def meta_tag(cell: str, cls: TopicClass, name: str) -> str:
    _check(_CELL, cell, "cell")
    _check(_NAME, name, "name")
    return f"{ENTERPRISE}/{META}/tags/{cell}/{cls.value}/{name}"


def meta_status(service: str) -> str:
    _check(_SERVICE, service, "service")
    if service == "tags":
        raise TopicError("'tags' is reserved under _meta")
    return f"{ENTERPRISE}/{META}/{service}/status"


def edge_raw(device: str, raw_tag: str) -> str:
    _check(_DEVICE, device, "device")
    _check(_RAW_TAG, raw_tag, "raw tag")
    if not raw_tag.startswith(device + "."):
        raise TopicError(f"raw tag {raw_tag!r} does not belong to device {device!r}")
    return f"{EDGE}/raw/{device}/{raw_tag}"


def edge_unmapped() -> str:
    return f"{EDGE}/unmapped"


def sim_cmd(kind: SimCommand) -> str:
    return f"{SIM}/cmd/{SimCommand(kind).value}"


def sim_clock() -> str:
    return f"{SIM}/clock"


def sim_faults(cell: str) -> str:
    _check(_CELL, cell, "cell")
    return f"{SIM}/faults/{cell}"


# --- subscription patterns ------------------------------------------------------------


def sub_class(cls: TopicClass, unit: UnitPath | None = None) -> str:
    """All topics of one class, for one unit or (default) every unit."""
    base = unit.prefix if unit else f"{ENTERPRISE}/+/+/+/+"
    return f"{base}/{cls.value}/#"


def sub_state_batch() -> str:
    return f"{ENTERPRISE}/+/+/+/+/{TopicClass.STATE.value}/batch"


SUB_UNS_ALL = f"{ENTERPRISE}/#"
SUB_EDGE_RAW = f"{EDGE}/raw/#"
SUB_SIM_CMD = f"{SIM}/cmd/#"
SUB_SIM_FAULTS = f"{SIM}/faults/#"
SUB_META_TAGS = f"{ENTERPRISE}/{META}/tags/#"


# --- parsing --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedTopic:
    kind: TopicKind
    unit: UnitPath | None = None
    cls: TopicClass | None = None
    name: str | None = None  # remainder after the class, e.g. "ph", "phase/temp_ctrl"
    cell: str | None = None
    service: str | None = None
    device: str | None = None
    raw_tag: str | None = None
    command: SimCommand | None = None


def parse(topic: str) -> ParsedTopic:
    """Parse any topic this project uses. Raises TopicError for anything else."""
    parts = topic.split("/")
    try:
        if parts[0] == ENTERPRISE and len(parts) > 1 and parts[1] == META:
            if parts[2] == "tags" and len(parts) == 6:
                cell, cls, name = parts[3], TopicClass(parts[4]), parts[5]
                if meta_tag(cell, cls, name) == topic:
                    return ParsedTopic(TopicKind.META_TAG, cls=cls, name=name, cell=cell)
            elif len(parts) == 4 and parts[3] == "status" and meta_status(parts[2]) == topic:
                return ParsedTopic(TopicKind.META_STATUS, service=parts[2])
        elif parts[0] == ENTERPRISE and len(parts) >= 7:
            unit = UnitPath(*parts[1:5])
            cls = TopicClass(parts[5])
            rest = parts[6:]
            for n in rest[:-1]:
                _check(_NAME, n, "name")
            last = rest[-1]
            is_alert = cls is TopicClass.AI and rest[:2] == ["anomaly", "alert"]
            _check(_ALERT_KEY if is_alert and len(rest) == 3 else _NAME, last, "name")
            return ParsedTopic(TopicKind.UNS, unit=unit, cls=cls, name="/".join(rest))
        elif parts[0] == EDGE:
            if parts[1:] == ["unmapped"]:
                return ParsedTopic(TopicKind.EDGE_UNMAPPED)
            if len(parts) == 4 and parts[1] == "raw" and edge_raw(parts[2], parts[3]) == topic:
                return ParsedTopic(TopicKind.EDGE_RAW, device=parts[2], raw_tag=parts[3])
        elif parts[0] == SIM:
            if parts[1:] == ["clock"]:
                return ParsedTopic(TopicKind.SIM_CLOCK)
            if len(parts) == 3 and parts[1] == "cmd":
                return ParsedTopic(TopicKind.SIM_CMD, command=SimCommand(parts[2]))
            if len(parts) == 3 and parts[1] == "faults" and sim_faults(parts[2]) == topic:
                return ParsedTopic(TopicKind.SIM_FAULTS, cell=parts[2])
    except (IndexError, ValueError) as exc:
        raise TopicError(f"not a pharma-4 topic: {topic!r}") from exc
    raise TopicError(f"not a pharma-4 topic: {topic!r}")


# --- delivery policy (ADR-0002 broker settings, extended by ADR-0012/0013) --------------


@dataclass(frozen=True, slots=True)
class Delivery:
    qos: int
    retain: bool


def delivery(topic: str) -> Delivery:
    """QoS and retain for a topic. The single place this policy lives."""
    p = parse(topic)
    match p.kind:
        case TopicKind.UNS:
            assert p.cls is not None and p.name is not None
            if p.cls in (TopicClass.PV, TopicClass.SP):
                return Delivery(0, True)
            if p.cls in (TopicClass.LAB, TopicClass.STATE):
                return Delivery(1, True)
            if p.cls is TopicClass.EVENTS:
                return Delivery(1, False)
            if p.name.startswith("anomaly/alert/") or p.name == "yield/recommendation":
                return Delivery(1, True)
            return Delivery(0, True)  # ai/anomaly/score, ai/yield/prediction
        case TopicKind.META_TAG | TopicKind.META_STATUS | TopicKind.SIM_CLOCK:
            return Delivery(1, True)
        case TopicKind.EDGE_RAW:
            return Delivery(0, False)
        case TopicKind.EDGE_UNMAPPED:
            return Delivery(1, True)  # retained so a browser always shows the latest
        case TopicKind.SIM_CMD | TopicKind.SIM_FAULTS:
            return Delivery(1, False)
    raise AssertionError(p.kind)
