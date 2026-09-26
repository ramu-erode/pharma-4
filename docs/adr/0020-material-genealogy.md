# ADR-0020: Material genealogy across sites

Status: Accepted
Date: 2026-09-26

## Context

With an API site feeding a tablet site (ADR-0018), the most valuable question the graph
can answer becomes "which API lots went into this tablet batch, and what happened to
them at Tuas?" This is forward and backward traceability. A recall, or a dissolution
failure that traces back to coarse API, needs exactly that.

The simulator also needs to know which API lots Freiburg holds, with their true
properties, when a tablet batch starts, because API particle size and free SA drive
the tablet process. The simulator reads no database. Backfilled lots are known only to
the bootstrap that wrote them.

## Decision

**Lots are batch ids.** As in most ERPs, the lot number of what a batch produces is
its batch number. An API lot is `B2026-0415`, and so is the tablet lot of a tablet
batch.

**Movements are events.** A new event stream `events/material` carries typed payloads
(ADR-0013). The simulator publishes it, standing in for the MES:

- `MATERIAL_PRODUCED {batch_id, lot, material, quantity_kg}`, on the unit that
  discharges it: FD-202 for API, TP-303 for tablets.
- `MATERIAL_CONSUMED {batch_id, lot, material, quantity_kg}`, on the unit that charges
  it: BL-301, one event per API lot used.

**In the graph** (graph-sync):
`(:Batch)-[:PRODUCED]->(:MaterialLot {id})-[:OF]->(:Material {id})` and
`(:Batch)-[:CONSUMED {quantity_kg}]->(:MaterialLot)`. A tablet batch reaches the API
batch that made its lot in two hops, whatever site each ran at. i3X serves lots and
materials as objects, with `Produced` and `Consumed` relationships and their reverses.

**Allocation is first-in, first-out.** A tablet batch consumes the oldest released API
lots until it has 200 kg, so an API lot of about 580 kg feeds about three tablet
batches, and a tablet batch sometimes draws on two lots. A lot is released when its
batch completes with a passing CoA. Rejected lots are never consumed.

**Inventory is simulation state, handed off over `_sim`.**

- The simulator keeps Freiburg's API inventory, with each lot's true properties, and
  publishes it retained on `_sim/inventory`.
- Bootstrap, after backfill, publishes the stock left at the end of history once,
  retained, on `_sim/opening_stock`.
- A simulator with no inventory of its own adopts the opening stock, whenever it
  arrives.

Both topics are under `_sim/#`, so the AI services and i3X cannot see them (ADR-0012).
Everything they need about a lot is in the CoA the API batch published on `lab/*`.

## Consequences

- Bootstrap becomes an MQTT client, with its own broker user that may write only
  `_sim/opening_stock`, and it depends on the broker.
- A lot made live at Tuas is available to Freiburg as soon as it completes. The weeks
  of QC release and shipping are skipped, so a demo can show both ends in minutes.
  Backfilled history keeps a 21-day release-and-ship gap.
- Excipient lots (MCC, starch, stearic acid) are not tracked. They would add nodes,
  but no new relationship kind.

## Alternatives considered

- **Genealogy only in the graph, from the planner.** Backfill knows the allocation,
  but a live batch would have no way to express it. An event is how an MES reports it.
- **Let the simulator query the historian for lots.** It would work, but it would give
  the plant stand-in a read path into the IT stores, which the UNS exists to avoid.
- **Lots as UNS "cells".** Lots are not equipment. The ISA-95 path is for where
  things happen, not what moves.
