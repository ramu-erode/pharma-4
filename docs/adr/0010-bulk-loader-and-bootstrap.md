# ADR-0010: Historical backfill bypasses the broker; a bootstrap service makes the stack self-starting

Status: Accepted
Date: 2026-09-23

## Context

Training needs 200 historical batches: about 52M tag rows at the 60 s publish period
(ADR-0009), plus events, lab results and graph context. Two problems follow.

Pushing that history through MQTT, at a few thousand messages a second, takes hours.
It also overwrites the retained "current" values with historical ones.

CLAUDE.md says a change that needs manual setup is not done. But a fresh
`docker compose up` has no history and no models, so the AI services have nothing to
run.

## Decision

**Backfill runs the pipeline in-process, without the broker.** `simulator.backfill`
calls the same pure functions the live services use:

1. the simulator produces raw samples,
2. the edge adapter's `map_and_enrich()` and deadband turn them into UNS messages,
3. the historian's writer bulk-loads them into TimescaleDB with `COPY`,
4. graph-sync's handlers `MERGE` events, lab and state into Neo4j, then run the
   attribution projection (ADR-0011).

This is the **only sanctioned bypass** of ADR-0001's "all data flows through the
broker". It is a bulk loader, not a producer. It writes nothing retained, and no
live service reads from it.

**A one-shot `bootstrap` compose service makes first start automatic.** It is
idempotent and runs, in order:

- applies the SQL migrations and `graph/schema.cypher`,
- loads `config/plant.yaml` and `graph/recipes/*.yaml` into the graph,
- runs backfill if `tag_values` is empty,
- trains the models if the `models` volume has no artefacts for the current data hash.

The AI services declare `depends_on: bootstrap: service_completed_successfully`. A
fixed seed makes it reproducible. `python -m simulator.backfill` and
`python -m ai.train_all` remain for deliberate re-runs.

## Consequences

- A clean checkout reaches demo-ready with `docker compose up`, in a few minutes.
- Mapping, deadband and sync logic exist once. For that to work they must stay pure
  functions that can be called without a broker, which is a design constraint on
  those services.
- Backfilled rows never passed through Mosquitto, so the broker's ACL did not police
  them. The loader is trusted code.

## Alternatives considered

- **Replay the history over MQTT.** Every byte takes the real path, but it takes
  hours and pollutes retained state.
- **A separate SQL/Cypher generator.** Fastest to write, but the mapping and
  attribution logic would exist twice and drift.
- **Commit pre-built dumps and models.** Binary blobs in git that go stale whenever
  the simulator changes.
- **Documented manual steps.** Goes against the project's own "no manual setup" rule.
