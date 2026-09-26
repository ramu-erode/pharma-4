# Architecture Decision Records

Each ADR records one decision: the context that forced it, what was decided, and what
it costs us. They are short on purpose.

ADRs are immutable once accepted. To change a decision, write a new ADR that
supersedes the old one and update the old one's status line.

When a later ADR corrects only an older ADR's *consequences* or a detail of it, and
leaves its decision standing, the older ADR's status becomes
`Accepted; amended by ADR-NNNN`. The older text stays as written.

Status lines on older ADRs change only when the newer ADR is **accepted**, not while it
is proposed.

## Format

```
# ADR-NNNN: Title

Status: Proposed | Accepted | Accepted; amended by ADR-NNNN | Superseded by ADR-NNNN
Date: YYYY-MM-DD
(optional) Supersedes / Amends / Refines: ADR-NNNN

## Context
## Decision
## Consequences
## Alternatives considered
```

## Index

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-mqtt-unified-namespace.md) | MQTT Unified Namespace as the integration backbone | Accepted; amended by 0009, 0018 |
| [0002](0002-mosquitto-json-over-sparkplug.md) | Mosquitto with JSON payloads, not Sparkplug B | Accepted |
| [0003](0003-python-everywhere.md) | Python for every service, no low-code layer | Accepted |
| [0004](0004-split-timeseries-and-graph.md) | Split storage: TimescaleDB for series, Neo4j for context | Accepted |
| [0005](0005-edge-adapter-separate-from-simulator.md) | A separate edge adapter between raw tags and the UNS | Accepted; amended by 0009 |
| [0006](0006-phase-attribution-via-equipment-binding.md) | Attribute process values to phases via equipment binding | Accepted (refined by 0011) |
| [0007](0007-ai-advisory-only.md) | AI output is advisory and published back into the UNS | Superseded by 0014 |
| [0008](0008-simulator-as-data-source.md) | A simulator, not a real DCS, is the POC data source | Accepted (refined by 0019) |
| [0009](0009-simulated-plant-time.md) | Simulated plant time in `ts`, wall-clock time for liveness | Accepted; amended by 0018 |
| [0010](0010-bulk-loader-and-bootstrap.md) | Backfill bypasses the broker; bootstrap makes the stack self-starting | Accepted |
| [0011](0011-operations-and-parallel-phases.md) | Sequential operations, parallel phases, projected attribution | Accepted; amended by 0018 |
| [0012](0012-sim-branch-control-and-ground-truth.md) | A `_sim` branch for control and ground truth, fenced by ACL | Accepted |
| [0013](0013-payload-typing-and-alert-lifecycle.md) | Typed `v` for structured payloads, and a keyed alert lifecycle | Accepted (refined by 0021) |
| [0014](0014-advisory-ai-paired-gain-gate.md) | AI output is advisory; recommendations are gated on paired gain | Superseded by 0015 |
| [0015](0015-lever-effects-from-the-doe.md) | Lever effects come from the designed experiment, not the in-batch model | Accepted (refined by 0017, 0021) |
| [0016](0016-i3x-read-api.md) | A read-only i3X 1.0 façade is the standard read API | Accepted |
| [0017](0017-llm-assistant-reads-through-i3x.md) | The LLM assistant reads the plant only through i3X | Accepted |
| [0018](0018-multi-site-enterprise.md) | PharmaNextGen, three sites, and batches that move between units | Accepted |
| [0019](0019-api-and-osd-simulation.md) | Simulating aspirin API synthesis and tablet manufacture | Accepted |
| [0020](0020-material-genealogy.md) | Material genealogy across sites | Accepted |
| [0021](0021-ai-profiles-per-equipment-class.md) | Anomaly models per equipment class, yield models per process | Accepted |
