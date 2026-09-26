"""Edge adapter core: map, enrich, deadband (ADR-0005). Pure; no MQTT.

The adapter maps and enriches, nothing more (CLAUDE.md). It does not detect faults, fix
values or restamp time: `v` is never rounded and `ts` is the source timestamp
(ADR-0009). Backfill calls these functions directly (ADR-0010).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from common import uns
from common.models import Quality, TagKind
from common.plant import Plant, get_plant
from common.uns import TopicClass, UnitPath

TAG_MAP_FILE = Path(__file__).with_name("tag-map.yaml")


@dataclass(frozen=True, slots=True)
class TagEntry:
    raw_tag: str  # full raw tag, e.g. BR101.AIC-102.PV
    unit_path: UnitPath
    cls: TopicClass
    name: str
    unit: str
    deadband: float

    @property
    def topic(self) -> str:
        return (
            uns.pv(self.unit_path, self.name)
            if self.cls is TopicClass.PV
            else uns.sp(self.unit_path, self.name)
        )

    @property
    def kind(self) -> TagKind:
        return TagKind.PV if self.cls is TopicClass.PV else TagKind.SP


@dataclass(frozen=True, slots=True)
class TagMap:
    entries: dict[str, TagEntry]  # by full raw tag

    @classmethod
    def load(cls, plant: Plant | None = None, path: Path = TAG_MAP_FILE) -> TagMap:
        """Every commissioned device's tags, at the unit's path in the plant model."""
        plant = plant or get_plant()
        raw = yaml.safe_load(path.read_text())
        entries: dict[str, TagEntry] = {}
        for device, dev in raw["devices"].items():
            unit = plant.unit(dev["cell"])
            if unit.path.device != device:
                raise ValueError(f"device {device} does not match cell {dev['cell']}")
            if unit.cls != dev["class"]:
                raise ValueError(f"{device}: tag map says {dev['class']}, plant says {unit.cls}")
            for suffix, spec in raw["classes"][dev["class"]].items():
                cls_ = TopicClass(spec["class"])
                if cls_ not in (TopicClass.PV, TopicClass.SP):
                    raise ValueError(f"{suffix}: the edge adapter only owns pv/ and sp/")
                entry = TagEntry(
                    raw_tag=f"{device}.{suffix}",
                    unit_path=unit.path,
                    cls=cls_,
                    name=spec["name"],
                    unit=spec["unit"],
                    deadband=float(spec["deadband"]),
                )
                entry.topic  # validates the name through the topic builder  # noqa: B018
                entries[entry.raw_tag] = entry
        topics = [e.topic for e in entries.values()]
        if len(set(topics)) != len(topics):
            raise ValueError("two raw tags map to the same UNS topic")
        return cls(entries)

    @property
    def cells(self) -> set[str]:
        return {e.unit_path.cell for e in self.entries.values()}


@dataclass(frozen=True, slots=True)
class Mapped:
    """One contextualised value, ready to become a ScalarPayload."""

    topic: str
    value: float
    ts: datetime
    unit: str
    q: Quality
    batch: str | None
    entry: TagEntry


@dataclass(frozen=True, slots=True)
class Unmapped:
    raw_tag: str
    value: float
    ts: datetime
    q: Quality


def map_and_enrich(
    raw_tag: str,
    value: float,
    ts: datetime,
    q: Quality,
    tag_map: TagMap,
    batch_of_cell: dict[str, str | None],
) -> Mapped | Unmapped:
    """Look up the tag and attach unit, batch (the live cache, ADR-0006) and quality."""
    entry = tag_map.entries.get(raw_tag)
    if entry is None:
        return Unmapped(raw_tag, value, ts, q)
    return Mapped(
        topic=entry.topic,
        value=value,
        ts=ts,
        unit=entry.unit,
        q=q,
        batch=batch_of_cell.get(entry.unit_path.cell),
        entry=entry,
    )


@dataclass(slots=True)
class _Last:
    value: float
    ts: datetime
    q: Quality
    batch: str | None


class Deadband:
    """Report-by-exception: publish on change beyond the deadband, on any change of
    quality or batch, or when the floor interval has passed in source time."""

    def __init__(self, floor_s: float, publish_period_s: float | None = None) -> None:
        if publish_period_s is not None and floor_s <= publish_period_s:
            raise ValueError(
                f"deadband floor ({floor_s}s) must exceed the raw publish period "
                f"({publish_period_s}s), or the deadband does nothing (ADR-0009)"
            )
        self.floor = timedelta(seconds=floor_s)
        self._last: dict[str, _Last] = {}

    def offer(self, m: Mapped) -> bool:
        last = self._last.get(m.topic)
        publish = (
            last is None
            or abs(m.value - last.value) > m.entry.deadband
            or m.ts - last.ts >= self.floor
            or m.q is not last.q
            or m.batch != last.batch
        )
        if publish:
            self._last[m.topic] = _Last(m.value, m.ts, m.q, m.batch)
        return publish
