# ADR-0018: PharmaNextGen, three sites, and batches that move between units

Status: Accepted
Date: 2026-09-26
Amends: ADR-0001 (enterprise name, one site), ADR-0009 (first live batch id), ADR-0011 (operations per unit)

## Context

The POC modelled one enterprise (`pharmaco`), one site (Chennai) and one process: two
bioreactors, each running a whole batch on its own. That shows a UNS, but not the
thing a UNS is for at enterprise scale: several sites, several kinds of process, and
material that flows between them.

We want the demo to look like a real network: a biologics drug-substance site, a
small-molecule API site that makes aspirin (acetylsalicylic acid), and an
oral-solid-dose site that presses tablets from that API. The sites are modelled on
real locations, but the company is fictitious and carries no real brand.

Two assumptions in the code break:

- One `site/area/line` for the whole stack, held in settings.
- One unit per batch. An API batch reacts and crystallises in a reactor, then is
  filtered and dried in a filter-dryer. A tablet batch is blended, roller-compacted
  and compressed on three machines.

## Decision

**Enterprise and sites.** The enterprise is **PharmaNextGen**, and the UNS root is
`pharmanextgen/`. It has three sites:

| Site id | Location | Role | Area / line | Units |
| --- | --- | --- | --- | --- |
| `grange-castle` | Grange Castle, Dublin, Ireland | Biologics drug substance (CHO mAb) | `upstream/suite-1` | BR-101, BR-102 |
| `tuas` | Tuas, Singapore | Small-molecule API (aspirin) | `api/train-1` | RX-201 reactor-crystallizer, FD-202 filter-dryer |
| `freiburg` | Freiburg, Germany | Oral solid dose (aspirin 500 mg tablets) | `osd/line-1` | BL-301 blender, RC-302 roller compactor, TP-303 tablet press |

Grange Castle replaces Chennai. Its bioreactor process is unchanged.

**The plant model is one file per concern, keyed by equipment class.** `config/plant.yaml`
holds the whole hierarchy (enterprise → sites → areas → lines → units). Every unit
names its **equipment class** (`bioreactor`, `reactor`, `filter_dryer`, `blender`,
`roller_compactor`, `tablet_press`). Every line names its **process** (`bioreactor`,
`api`, `osd`). Equipment modules, control modules, sensors and phase bindings are
defined once per equipment class and apply to every unit of that class. The edge tag
map and the simulator's DCS configuration are also keyed by equipment class.
`common/plant.py` is the one reader of `plant.yaml`. The `SITE/AREA/LINE` settings are
gone: a unit's ISA-95 path comes from the plant model.

**A batch runs on a train of units.** A process defines its train. The bioreactor
process runs a whole batch on one unit, so two batches can run side by side. The API
and OSD processes each run one batch through their units in order.

- A unit publishes its own `state/batch` and `state/operation` while the batch is on
  it, and `Idle` otherwise, so ADR-0011 holds per unit.
- Operations are sequential per unit, and operation names are unique within a
  process, so `<batch>/<operation>` stays a natural key in the graph.
- `BATCH_START` is published on the first unit and `BATCH_END` on the last unit.
  Operation and phase changes are published on the unit where they happen.
- graph-sync links `(:Batch)-[:RAN_ON]->(:Equipment)` for every unit an operation ran
  on. Each `Operation` records its unit, and attribution (ADR-0006) only joins a unit's
  tags to that unit's operations.
- The live simulator starts a train batch only when every unit of the train is idle.
  Batches are not pipelined through a train. That would be more realistic, but it
  would not show anything new.

**Batch ids stay global.** Ids are `B<start year>-<global 4-digit sequence>`, as if the
ERP issued them, across every site. Backfill writes each site's history (200
bioreactor, 140 API and 140 tablet batches by default) and numbers all of them in
start-time order. The live sequence continues from the total, so with the defaults
the first live batch is **B2026-0480**, not B2026-0200. The bioreactor history keeps
its seeds, so its data is the same as before, only renumbered.

## Consequences

- Topic patterns keep their shape (`pharmanextgen/+/+/+/+/<class>/#`), so the ACL and
  every subscription only change their root.
- A running stack from before this ADR has `pharmaco/...` data in its volumes. It has
  to be rebuilt (`docker compose down -v`, then `up`). Nothing migrates it.
- Every consumer that assumed a bioreactor (feature lists, dashboard trends, i3X
  object types) now dispatches on the equipment class or process. ADR-0021 covers the AI.
- A batch's `cell` property in the graph is the unit it started on. "Where did it run"
  is answered through `RAN_ON`, which can have several targets.
- The fixed 4-digit sequence allows 10,000 batch ids. At the default history, that is
  about 9,500 live batches of headroom.

## Alternatives considered

- **One UNS root per site.** It would match companies that federate brokers per
  site, but the POC has one broker and ISA-95 puts the enterprise at the root anyway.
- **One unit per batch everywhere**, with the API reaction, isolation and drying all
  in one "reactor". It would be simpler, but anyone from a small-molecule plant would
  see that it is wrong. A filter-dryer is a separate piece of equipment with its own
  tags and faults.
- **A separate batch id per unit procedure.** ISA-88 allows it, but the ERP issues one
  batch number per lot, and genealogy (ADR-0020) is keyed on that number.
