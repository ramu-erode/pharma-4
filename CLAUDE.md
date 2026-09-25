# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

**pharma-4** is a proof of concept for Pharma 4.0 data architecture. It simulates a
biopharmaceutical fed-batch bioreactor (CHO cell culture producing a monoclonal
antibody), publishes its data into a **Unified Namespace** over MQTT, models the
context in a **Neo4j knowledge graph**, and runs two AI use cases on top:

1. **Anomaly detection** — catch developing equipment and process faults before they
   breach spec.
2. **Yield optimization** — predict final titer mid-batch and recommend setpoints
   inside the validated design space.

It is a demo and learning vehicle, not a GMP system. Nothing here is validated, and
the AI output is advisory only.

## Architecture in one paragraph

The simulator emits raw DCS-style tags (`BR101.AIC-102.PV`) to `edge/raw/#`. The
**edge adapter** maps them into the ISA-95 UNS topic tree, adds unit, batch id and
quality, applies a deadband, and republishes. Everything else subscribes to the
broker: the **historian** writes to TimescaleDB, **graph-sync** maintains Neo4j,
and the **anomaly** and **yield** services score the stream and publish their
results back into the UNS under `ai/*`. A Streamlit **dashboard** reads all three.
The **i3x** service is the one standard read API: i3X 1.0 over the broker, historian
and graph, read-only (ADR-0016). The LLM **assistant** reads the plant only through it
(ADR-0017).

Full design: `docs/architecture.md` and the ADRs in `docs/adr/`.

## Repository layout

```
config/      plant.yaml: hierarchy, modules, phase classes, bindings (FHX stand-in)
common/      Shared UNS topic builder, pydantic payload models, MQTT helpers
simulator/   True process state, sensor model, control, faults, clock, ctl, backfill
edge/        Edge adapter: tag map, context enrichment, deadband
historian/   TimescaleDB writer and SQL migrations
graph/       Neo4j sync, attribution projection, schema.cypher, recipes
ai/          features.py, anomaly/ and yield_/ services, plus training entry points
bootstrap/   One-shot, idempotent first-start: migrations, backfill, training
i3x/         i3X 1.0 read API: address space, values, subscriptions, FastAPI app
assistant/   LLM Q&A through i3X only: i3X client, tools, tool loop, CLI
dashboard/   Streamlit app
docs/adr/    Architecture Decision Records
tests/       pytest suites mirroring the package layout
```

## Conventions

**Language and tooling**

- Python 3.12 everywhere. No second language in the services.
- `ruff` for lint and format. `pytest` for tests. `pydantic` for payloads.
- Type hints on all public functions. Prefer explicit over clever.

**UNS topics**

- Never hand-build a topic string. Use the builder in `common/uns.py`.
- Topic tree: `pharmaco/<site>/<area>/<line>/<cell>/<class>/<name>`.
- Only the owning service publishes to a branch. The edge adapter owns `pv/*`,
  `sp/*` and `_meta/tags`; the simulator owns `lab/*`, `state/*`, `events/*`,
  `edge/raw` and `_sim/clock|faults`; the AI services own `ai/*`; the dashboard may
  write only `_sim/cmd/#`. Every service owns its own `_meta/<service>/status`.
  Respect the Mosquitto ACL — if a publish needs a new branch, update the ACL in the
  same change.
- Every payload carries `v`, `ts` (UTC ISO-8601), `unit`, `q`, `batch`, `src`. `ts`
  is simulated plant time, so never compare it with `now()`; use heartbeats for
  liveness (ADR-0009). `v` is typed per topic class (ADR-0013).
- `_sim/#` exists only because the plant is simulated. The AI services and the i3x
  service must never read it: it carries ground-truth fault labels (ADR-0012). Whatever
  i3X serves, an LLM may see, so i3x queries never touch `FaultInjection` or
  `fault_labels` either (`tests/i3x/test_fence.py`).

**Data placement**

- Time series go to TimescaleDB. Context and relationships go to Neo4j. Do not put
  raw tag values in the graph, and do not put relationships in the historian.
- The graph is the source of truth for batch and phase attribution. The `batch` field
  stamped on a payload is a convenience cache and may be wrong for late data.

**Naming**

- Equipment ids are `BR-101` style in the UNS and graph; raw DCS tags keep their
  native `BR101.AIC-102.PV` form and only appear in `edge/raw` and the tag map.
- Batch ids are `B<start year>-<global 4-digit seq>`, e.g. `B2026-0142`. The sequence
  never resets; the first live batch is `B2026-0200`.

## Working practices

- **Read the relevant ADR before changing a design decision.** If a change
  contradicts an accepted ADR, write a new ADR that supersedes it rather than
  quietly diverging.
- **One concern per service.** Resist adding logic to the edge adapter; it maps and
  enriches, nothing more.
- **Tests alongside behaviour.** Every fault type in the simulator needs a test that
  asserts the anomaly layer catches it. Every tag-map change needs a mapping test.
- Keep the whole stack runnable with `docker compose up`. If a change needs manual
  setup steps, it is not done. The one-shot `bootstrap` service handles migrations,
  backfill and training (ADR-0010).
- Do not commit secrets. MQTT and database credentials come from environment
  variables, with defaults only in `.env.example`.

## Common commands

```bash
docker compose up -d              # bring up everything; the dashboard is on http://localhost:8501
docker compose logs -f simulator  # follow one service
python -m simulator.backfill      # re-generate the 200 historical batches (bootstrap does this on first up)
python -m ai.train_all            # re-train anomaly and yield models (bootstrap does this on first up)
python -m ai.anomaly.evaluate     # anomaly model vs ground-truth labels (writes models/anomaly_eval.json)
python -m ai.yield_.evaluate      # optimizer vs the simulator's true titer (writes models/yield_eval.json)
python -m graph.replay [--all]    # rebuild the graph from the historian
python -m simulator.ctl --help    # demo control: start batch, inject fault, speed, run to day N
pytest                            # run the test suite (fault harness is marked slow)
pytest -m compose                 # smoke-test against the running stack
python -m assistant "question"    # ask the plant via i3X (needs ANTHROPIC_API_KEY in .env)
curl localhost:8600/v1/info       # i3X read API; other endpoints need X-API-Key: $I3X_API_KEY
ruff check . && ruff format .     # lint and format
```

## Domain notes worth knowing

- A **fed-batch** run lasts about 14 days: growth phase, a temperature shift around
  day 5, then production, then harvest. Titer (g/L) is the yield metric.
- The **temperature shift day**, **pH band**, **DO stability** and **feed timing** are
  the real levers on titer. They are the optimizer's decision variables.
- **Operations are sequential; phases run in parallel.** `state/operation` is one value
  (Growth, TempShift, Production…). Within it, phases such as TEMP_CTRL, PH_CTRL,
  DO_CTRL and FEED_ADD run at once, each in `state/phase/<name>`. A process value
  belongs to a phase through *equipment binding*, not through time alone. See
  ADR-0006 and ADR-0011.
- In biologics, "the process is the product" — you cannot test quality in at the end.
  That is why process data and context matter so much here.

## Out of scope for this repo

GMP validation, audit trails, e-signatures, user management, real DCS or PI
connectivity, MQTT clustering and TLS. Each has a named upgrade path in
`docs/architecture.md`; do not start implementing them without discussion.
