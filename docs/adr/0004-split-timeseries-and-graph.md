# ADR-0004: Split storage — TimescaleDB for series, Neo4j for context

Status: Accepted
Date: 2026-09-23

## Context

The POC has two very different data shapes. One is a firehose of numeric samples:
every tag, every few seconds, for 14 days, across 200 historical batches. The other
is a small, densely connected web of context: which batch ran on which vessel, under
which recipe, with which sensors, hitting which spec limits, producing which titer.

The questions differ too. "Plot pH for batch 142" is a range scan. "Which batches
sharing this probe ended below 3 g/L, and what alerts preceded them?" is a traversal.

## Decision

Two stores, with a firm boundary:

- **TimescaleDB** holds every tag value. One hypertable
  `tag_values(ts, topic, batch_id, value, quality)` plus a one-minute continuous
  aggregate for charts and feature extraction.
- **Neo4j** holds context and relationships only: site, area, equipment, sensor, tag,
  spec limit, recipe, batch, phase, event, outcome.

**Raw tag values never go into the graph.** Relationships never live in the historian.
The graph-sync service subscribes to `_meta`, `state`, `events`, `lab` and
`ai/*/alert` — deliberately not `pv` — which keeps the graph at roughly 50 nodes per
batch.

Where a bulk query needs both (building training features, say), the graph's batch and
phase intervals are synced into a flat table in TimescaleDB and joined there on time,
rather than querying Neo4j per sample.

## Consequences

- Each store does what it is good at, and neither is asked to do the other's job.
- Two databases to run, back up and reason about. Worth it here; the demo would be
  weaker with either alone.
- Any question spanning both needs an explicit join. The interval-table pattern above
  is the sanctioned way to do it; ad-hoc per-value graph lookups are not.
- The graph staying small means it can be shown live on a laptop, which is a large
  part of its demo value.

## Alternatives considered

- **Everything in PostgreSQL.** Fewer moving parts, but recursive genealogy queries
  become CTEs that nobody wants to read on a screen, and the graph visual — the thing
  that sells the idea — is lost.
- **Everything in Neo4j.** Time-series volume at this rate would bloat the store and
  make traversals slow.
- **InfluxDB instead of TimescaleDB.** A reasonable swap, and flagged as an open
  question if a client's stack already uses it. SQL and the ability to join against
  batch intervals in the same engine decided it for now.
