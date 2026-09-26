# ADR-0011: Sequential operations, parallel phases, and a projected attribution table

Status: Accepted; amended by ADR-0018
Date: 2026-09-23
Refines: ADR-0006 (does not change its decision)

## Context

The architecture used "phase" for two different things. `state/phase` was a single
sequential value (Setup → Inoculation → Growth → Temp shift → Production → Harvest),
and the anomaly models were "one per phase" on that basis. ADR-0006 meanwhile
describes parallel ISA-88 phases bound to tags through equipment modules, with
control and monitor roles, and nothing in the graph schema supported it.

ADR-0004 and ADR-0006 also say the graph's intervals are "synced to a flat table" for
bulk joins, but no service owned that sync.

The static plant configuration (hierarchy, sensors, modules, bindings) was said to
arrive "from `_meta` topics", but no service published them.

## Decision

**Two ISA-88 levels.**

- **Operations** are sequential: Setup, Inoculation, Growth, TempShift, Production,
  Harvest. Each unit publishes exactly one retained `state/operation`. The per-stage
  anomaly and yield models key on operations.
- **Phases** run in parallel inside an operation: `TEMP_CTRL`, `PH_CTRL` (which drives
  CO2 and the base pump), `DO_CTRL`, and `FEED_ADD`. Each publishes a retained
  `state/phase/<name>` with a value of `RUNNING`, `HELD` or `COMPLETE`. The simulator
  emits HOLDs.

**The binding is configuration.** `config/plant.yaml` holds the site, area, units,
sensors, equipment modules, control modules, phase classes and aliases. It stands in
for a DeltaV FHX export. graph-sync loads it directly at bootstrap. In the graph:
`(:PhaseClass)-[:BOUND_TO {alias, role}]->(:EquipmentModule|:ControlModule)`, and
`(:ControlModule)-[:HAS_TAG]->(:Tag)`.

**`_meta` is published by whoever knows it.** The edge adapter publishes retained
`_meta/tags/<unit>/<class>/<name>` (raw tag, unit, kind, deadband) from its tag map.
Plant structure is not published over MQTT; it is config. Recipes stay in
`graph/recipes/*.yaml`.

**graph-sync owns the attribution projection.** A pure function `graph/project.py`
resolves ADR-0006's algorithm in Cypher and upserts
`tag_attribution(topic, batch_id, operation, phase, role, phase_state, t_start, t_end)`
in TimescaleDB. graph-sync calls it whenever an operation or phase interval closes or
changes, and backfill calls it in bulk. When no bound phase is found, `phase` and
`role` are null and `operation` carries the fallback, the lowest common ancestor.

**Runtime read path for the AI services.** Dynamic state (batch, operation, phase
state) comes from retained UNS `state/*`. Static context (spec limits, PARs, recipe,
bindings) is read from Neo4j once per batch, on the `events/batch` start event, and
cached. Training reads only TimescaleDB, joined to `tag_attribution` with a plain
`ts BETWEEN t_start AND t_end`.

## Consequences

- ADR-0006 is implemented as written, and the graph can answer "which phase
  controlled this loop" versus "which only watched it".
- graph-sync now also connects to TimescaleDB, but only to write the projection.
- The binding logic exists in exactly one place (`graph/project.py`), not in SQL or
  feature code.
- Neo4j is never on the per-sample path, so graph-sync lag cannot delay scoring. The
  UNS `state/*` values are a live cache and can briefly differ from the graph, just
  as the `batch` field can.
- `state/phase` in architecture.md and CLAUDE.md is renamed to `state/operation` plus
  `state/phase/<name>`.

## Alternatives considered

- **A single sequential phase, superseding ADR-0006.** Much less to build, but it
  loses the parallel-phase story.
- **Bind phase classes straight to tags.** Correct attribution, but it skips the alias
  chain that makes the binding credible on a DeltaV site.
- **A separate projector service, or raw intervals only.** One more container, or the
  ADR-0006 logic duplicated in every consumer.
- **The simulator publishes the whole plant model as `_meta`.** It would put the
  simulator in charge of "truth" about the DCS configuration.
