# Implementation plan

How to build what [architecture.md](architecture.md) and ADR-0001 to ADR-0014 describe. The plan has one foundation step and then five increments. Each increment ends demoable, and each numbered task is roughly one reviewable change.

Last updated 2026-09-23.

## Before starting

1. ~~Accept ADR-0009 to ADR-0014~~ — done 2026-09-23.
2. ~~Initialise the git repository~~ — done: `main`, remote `origin`.
3. Everything below is Python 3.12, `ruff` and `pytest`, per ADR-0003.

## Design rules that apply to every task

These follow from the ADRs, and the plan leans on them throughout.

- **Pure core, thin shell.** Every service is a pure core (plain functions and classes, no I/O) plus an MQTT/DB shell. Backfill (ADR-0010) and the fault harness both call the cores directly. If a core needs a broker to be tested, the design is wrong.
- **Pydantic at the wire only.** `common/models.py` validates messages entering and leaving MQTT. Inside the cores, and on the backfill hot path, data moves as frozen dataclasses or `NamedTuple`s. Validating 50M messages with pydantic would dominate backfill time.
- **One source of topic truth.** Every topic string comes from `common/uns.py`, and QoS/retain come from one policy function there. Tests check this; review enforces it.
- **Seeds everywhere.** Every random draw comes from a `numpy.random.Generator` derived from `(global_seed, batch_seq, purpose)`. Backfill, the harness and the demo are reproducible.
- **Config through environment.** `common/settings.py` (pydantic-settings) is the only thing that reads env vars. Defaults live in `.env.example`, secrets never in git.

## Step 0 — Foundation (size: S)

| # | Task | Files |
| --- | --- | --- |
| 0.1 | Package and tooling. Runtime pins go in `requirements.txt` (the "one dependency file"); `requirements-dev.txt` adds `-r requirements.txt` plus ruff, pytest, pytest-timeout, hypothesis. `pyproject.toml` holds package metadata, ruff config and pytest markers (`slow`, `compose`); `post-create.sh` already installs all three. | `pyproject.toml`, `requirements*.txt`, `.gitignore` |
| 0.2 | `common/settings.py`: service name, MQTT host and credentials, DB DSNs, seed, speed, publish period, deadband floor, `BACKFILL_BATCHES` (default 200; set 30 for fast dev loops). | `common/settings.py`, `.env.example` |
| 0.3 | `common/uns.py`: `UnitPath`, a `TopicClass` enum, builders (`pv`, `sp`, `lab`, `state`, `state_phase`, `events`, `ai_score`, `ai_alert(key)`, `ai_prediction`, `ai_recommendation`, `meta_tag`, `meta_status(service)`, `edge_raw`, `edge_unmapped`, `sim_cmd(kind)`, `sim_clock`, `sim_faults(unit)`), `parse(topic) -> ParsedTopic`, subscription patterns, and `delivery(topic) -> (qos, retain)` implementing the broker-settings table. | `common/uns.py` |
| 0.4 | `common/models.py`: a generic `Payload[V]` with the six keys, plus the `Quality` and `Src` enums. Concrete `v` types: `float`, `OperationName`, `PhaseState`, `BatchEvent`, `OperatorEvent`, `Alert`, `Prediction`, `Recommendation`, `TagMeta`, `Heartbeat`, `FaultLabel`, and the `SimCommand` union. `model_for(topic)` maps each topic class to its model. **Heartbeat convention:** `ts` = the last simulated time the service processed, and `v.wall` = wall-clock time. That gives liveness and lag in a single message. | `common/models.py` |
| 0.5 | `common/mqtt.py`: `connect(service)` with credentials from settings, a last-will of `offline` on `_meta/<service>/status`, reconnect with backoff, a 30 s wall-clock heartbeat thread, `publish(client, topic, payload)` using `uns.delivery()`, and `subscribe(client, pattern, handler)`, which parses with `model_for`. Validation failures are logged and counted, never raised into the loop. | `common/mqtt.py` |
| 0.6 | Container scaffolding: one `Dockerfile` (python:3.12-slim, `pip install -r requirements.txt`, copy the source); `docker-compose.yml` with mosquitto, timescaledb and neo4j, healthchecks on each (`pg_isready`, the Neo4j HTTP port, `mosquitto_sub -C 1 -t '$SYS/broker/uptime'`) and named volumes; and `mosquitto/` holding `mosquitto.conf`, a static `acl` from the architecture ACL table, and an entrypoint that builds `passwd` from env vars. | `Dockerfile`, `docker-compose.yml`, `mosquitto/*` |

**Tests:** round-trip every builder through `parse`; `delivery()` against the broker table; a sample payload for every topic class validated through `model_for`; an **ACL test** that parses `mosquitto/acl` and asserts each service can write exactly its own branches and that `anomaly` and `yield` have no `_sim/#` access.

**Done when:** `pytest` and `ruff` pass, and `docker compose up` starts the three infrastructure containers healthy.

## Increment 1 — Simulator, edge adapter, live UNS (size: L)

**Goal:** raw tags in, contextualised UNS out, and the simulator under control from a CLI.

### 1a. Simulator core (pure)

| # | Task | Files |
| --- | --- | --- |
| 1.1 | **Performance spike first.** Write the Monod/logistic ODE with plain-float Euler steps and time 14 days at a 5 s step. Budget: **≤ 2 s per batch** on one core. Failing that, raise the integration step to 15 s (the dynamics are minutes to hours) and record the change as an amendment to ADR-0009. Everything later assumes this budget. | `simulator/process.py` |
| 1.2 | `process.py`: a `TrueState` dataclass (VCD, viability, glucose, lactate, titer accumulator, true pH, DO, temperature, volume, dissolved CO2) and `step(state, actuators, dt) -> TrueState`, pure. Titer grows with the integral of viable cells, modulated by temperature, pH band, DO stability and feed. | `simulator/process.py` |
| 1.3 | `sensors.py`: a `Sensor` per tag with noise σ (never zero, so stuck detection works — ADR-0009/Q8), an offset function and a stuck flag; `measure(true, t, rng) -> float`. Lab sampling includes `ph_offline` from the **true** pH. | `simulator/sensors.py` |
| 1.4 | `control.py`: PI loops on **measured** values — temperature; pH split-range (CO2 below SP, base above); DO cascade (agitation, then O2 flow); the feed bolus schedule. Phase HOLD freezes the phase's loop outputs. | `simulator/control.py` |
| 1.5 | `recipes.py` + `graph/recipes/v1.yaml`–`v3.yaml`: nominal levers, PARs, and spec limits per tag. The simulator and graph-sync read the same files. A `Levers` dataclass. | `simulator/recipes.py`, `graph/recipes/*.yaml` |
| 1.6 | `faults.py`: a `Fault` protocol (`onset`, `apply(true, sensors, actuators, t)`, `label()`), with six implementations matching the architecture fault table. pH drift acts on the **sensor offset only**; the loop does the rest. | `simulator/faults.py` |
| 1.7 | `batches.py`: the batch state machine — Setup → … → Harvest; the phase instances per operation (`TEMP_CTRL`, `PH_CTRL`, `DO_CTRL`, `FEED_ADD`); random short HOLDs; outcome rules (contamination → ABORTED; viability < 60% → EARLY_HARVEST); batch-id allocation `B<year>-<seq>`. | `simulator/batches.py` |
| 1.8 | `engine.py`: `run_batch(spec, seed) -> Iterator[SimMessage]`, which yields raw samples on the publish period, lab, state changes, events and fault labels, with simulated timestamps. It is **the** shared core for live runs, backfill and tests. | `simulator/engine.py` |
| 1.9 | `truth.py`: `simulate_remaining(snapshot, levers, seed) -> float`, a noise-free forward run from a state snapshot, for optimizer evaluation. | `simulator/truth.py` |
| 1.10 | `dcs_tags.yaml`: the 15 raw tags per reactor from the architecture table, including `TI-199.PV`. | `simulator/dcs_tags.yaml` |

### 1b. Simulator shell

| # | Task | Files |
| --- | --- | --- |
| 1.11 | `clock.py`: `SimClock` — the monotonic start rule (read retained `state/*`, take the max with wall-clock now); speed, pause and run-to-day-N; pacing that sleeps between steps and never lets `ts` go backwards. Publishes `_sim/clock`. | `simulator/clock.py` |
| 1.12 | `run.py`: subscribes to `_sim/cmd/#` (batch, fault, clock, setpoint); drives one `engine` per unit; publishes raw to `edge/raw`, and lab/state/events to the UNS, labels to `_sim/faults`. A setpoint command becomes an actuator override plus an `events/operator` message. | `simulator/run.py` |
| 1.13 | `ctl.py`: a CLI publishing the same `SimCommand` models (`start-batch`, `inject`, `clear`, `speed`, `pause`, `run-to-day`, `setpoint`). | `simulator/ctl.py` |

### 1c. Edge adapter

| # | Task | Files |
| --- | --- | --- |
| 1.14 | `edge/tag-map.yaml`, authored separately from `dcs_tags.yaml`. | `edge/tag-map.yaml` |
| 1.15 | Core: `TagMap.load()`; `map_and_enrich(raw, tagmap, batch_of_unit) -> Mapped | Unmapped`; `Deadband.offer(mapped) -> bool` (change beyond the band, or the floor elapsed in `ts`); never rounds `v`; a startup check that the floor exceeds the publish period. | `edge/core.py` |
| 1.16 | Shell: subscribes to `edge/raw/#` and `…/state/batch`; publishes `pv`/`sp`, `edge/unmapped` and retained `_meta/tags/*`; heartbeat. | `edge/adapter.py` |
| 1.17 | Compose services `simulator` and `edge-adapter`, each with its own MQTT user. | `docker-compose.yml` |

**Tests:** ODE sanity (VCD grows then declines; titer is monotonic; mass balance within tolerance). The pH-drift test: the measured PV stays within ±0.02 of SP while the true pH falls and CO2 flow rises. Control-loop settling. Clock monotonicity across a simulated restart. Engine determinism (same seed, same message stream). **Tag-map coverage** (every sim tag is mapped except `expected_unmapped`). Deadband behaviour, including a stuck value appearing only on floor publishes with a bit-identical `v`. Enrichment uses the cached batch.

**Done when:** `docker compose up` shows raw → UNS in MQTT Explorer; `TI-199.PV` lands in `edge/unmapped`; `simulator.ctl start-batch BR-101` then `speed 3600` runs a batch to harvest; heartbeats and last-will work (stop a container and its status turns `offline`).

## Increment 2 — Historian, backfill, bootstrap (size: M)

**Goal:** `docker compose up` goes from a clean checkout to 200 historical batches with no manual steps.

| # | Task | Files |
| --- | --- | --- |
| 2.1 | Migrations plus a ~40-line runner with a `schema_migrations` table. `001` `tag_values` hypertable (7-day chunks, index on `(topic, ts)`); `002` `tag_values_1m` continuous aggregate; `003` `uns_events` (jsonb); `004` `tag_attribution` (GiST or btree on `(topic, t_start, t_end)`); `005` `fault_labels`; `006` a **compression policy** on `tag_values`, segmented by topic, for chunks older than 7 days. | `historian/migrations/*.sql`, `historian/migrate.py` |
| 2.2 | Writer core: `route(topic, payload) -> Row`, sending scalars to `tag_values`, structured messages to `uns_events` and `_sim/faults` to `fault_labels`; `copy_rows(conn, table, rows)` via psycopg 3 `COPY`. | `historian/core.py` |
| 2.3 | Historian shell: subscribes to `pharmaco/#` and `_sim/faults/#`; buffers rows and flushes every 1 s or every 5k rows. | `historian/writer.py` |
| 2.4 | **Campaign planner** (pure): assigns 200 batches to two reactors with about 2 days' turnaround, ending the day before first boot (so starting around mid-2022); ~150 manufacturing batches across recipe v1 → v3 in date order with lever jitter; ~50 PC batches from a Latin hypercube over the PARs in two campaigns; faults on about 15% (roughly even per type, contamination rare). Output: a list of `BatchSpec`. | `simulator/campaign.py` |
| 2.5 | `backfill.py`: plan → `multiprocessing.Pool` over batches → `engine.run_batch` → `edge.core` map + deadband → `historian.core.route` → COPY in chunks. Graph writes are added in increment 3. Honours `BACKFILL_BATCHES`. | `simulator/backfill.py` |
| 2.6 | `bootstrap/run.py`: an idempotent sequence (migrate → backfill if `tag_values` is empty → *graph and training steps added in increments 3–5*), with a clear log per step. Compose service `bootstrap` (`restart: "no"`), and `historian` depending on it. | `bootstrap/run.py`, `docker-compose.yml` |

**As built:** idempotency uses a `bootstrap_state` table (migration 007) rather than "`tag_values` is empty". It records the plan's end instant, so an interrupted backfill is redone with the same batches, and backfill owns every row before that instant. The 1-minute aggregate materialises only the last 30 days; older history is served in real-time mode from compressed raw data (materialising all of it measured 2.5 GB).

**Budgets to measure and record in this file:** backfill of 200 batches in under 10 minutes on an 8-core laptop; TimescaleDB at most 3 GB after compression. If either is missed, first reduce `tag_values` width (a `topic_id smallint` through a `topics` table) before touching the publish period.

**Tests:** route selection per topic class; the campaign plan (counts, date order, LHS coverage of every PAR, about 15% faults, no reactor overlap); backfill on 3 batches into a throwaway schema (skipped when no DB is available); bootstrap run twice is a no-op the second time.

**Done when:** a clean `docker compose up` ends with 200 batches in TimescaleDB, and a notebook plot shows titer spread by recipe version and campaign.

## Increment 3 — Knowledge graph and attribution (size: M)

**Goal:** context and the parallel-phase binding, queryable live.

| # | Task | Files |
| --- | --- | --- |
| 3.1 | `config/plant.yaml`: site, area and line; BR-101 and BR-102; equipment modules (thermal, pH, aeration, feed); control modules and their tags; sensors (model, calibration due); phase classes with aliases resolved per unit and `role`. | `config/plant.yaml` |
| 3.2 | `graph/schema.cypher`: uniqueness constraints per natural key; indexes on `Batch.id`, `Tag.topic`, `Event.id`. | `graph/schema.cypher` |
| 3.3 | Loaders: `plant.yaml` and recipes → `MERGE` statements, run by bootstrap. | `graph/load.py` |
| 3.4 | Sync core: `handle(topic, payload) -> list[Stmt]` for `_meta/tags`, `state/operation`, `state/phase/*`, `events/batch`, `events/operator`, `lab/*` at harvest (Outcome), `ai/anomaly/alert/*`, `ai/yield/recommendation` and `_sim/faults`. Everything is idempotent `MERGE`, batched with `UNWIND` for backfill. | `graph/core.py` |
| 3.5 | `common/levers.py`: `actual_levers(planned, operator_events) -> Levers`. **As built:** derived from `BATCH_START` plus `events/operator` rather than SP history: exact, and both graph-sync and increment 5 training have those messages. | `common/levers.py` |
| 3.6 | `graph/project.py`: a Cypher implementation of ADR-0006 (running phase instances ∩ bound phase classes → role, with the operation as fallback) → upsert `tag_attribution` rows for an interval. Called when an interval closes, and in bulk by backfill. | `graph/project.py` |
| 3.7 | Sync shell, plus backfill and bootstrap wiring (load config → backfill writes graph → bulk projection). | `graph/sync.py`, `simulator/backfill.py`, `bootstrap/run.py` |

**As built:** backfill does not feed graph-sync directly. Bootstrap replays the graph from the historian (`graph/replay.py`, migration 008 adds `uns_events.seq` for arrival order); 200 batches replay and project in about 20 s. Clean `docker compose up` to demo-ready including the graph: 108 s.

**Tests:** projection against a hand-built graph covering overlapping phases, a HOLD interval, a tag with no bound phase (fallback), and a tag with both `control` and `monitor` roles; replay idempotency (the same messages twice give the same node counts); `extract_levers` recovers the levers the engine was given, within tolerance; node count per batch about 60.

**Done when:** the four architecture Cypher queries return sensible results on the backfilled graph, and `tag_attribution` shows `PH_CTRL/control` and `FEED_ADD/monitor` rows for `pv/ph` during feeding.

## Increment 4 — Anomaly detection (size: L)

**Goal:** the pH-drift and DO-fouling demo beats, with measured detection performance.

| # | Task | Files |
| --- | --- | --- |
| 4.1 | `ai/features.py`: `to_grid(rows, start, end) -> DataFrame` (1-minute LOCF) and `windows(grid, context) -> DataFrame` (30-min windows every 5 min; the feature list from the architecture doc). One function, called from training and from the live buffer. | `ai/features.py` |
| 4.2 | `ai/data.py`: training extraction from TimescaleDB — clean batches only, raw rows joined to `tag_attribution` for operation, phase state and role. Never touches `fault_labels`. | `ai/data.py` |
| 4.3 | Layer 1 `rules.py`: spec limits, rate-of-change, stuck value (3 bit-identical published values). | `ai/anomaly/rules.py` |
| 4.4 | Layer 2 `stats.py`: EWMA and CUSUM per tag per operation, including CO2 rate and `pv/ph − lab/ph_offline`. | `ai/anomaly/stats.py` |
| 4.5 | Layer 3 `mspc.py` + `iforest.py`: an `AgeAlignedScaler` (mean and std per operation per 1-hour age bin, from clean batches); PCA with T², SPE, 99th-percentile limits and SPE contributions; Isolation Forest on the same scaled features. | `ai/anomaly/mspc.py`, `ai/anomaly/iforest.py` |
| 4.6 | `alerts.py`: an `AlertManager` keyed by `<layer>-<tag|class>`; opens after 2 windows over threshold and clears after 4 under; emits OPEN, CLEARED and then the empty retained clear. Fault-class suggestion by a simple lookup from the contributing-tag pattern. | `ai/anomaly/alerts.py` |
| 4.7 | `detector.py` (pure): `Detector.step(window, context) -> list[AlertChange]`, applying suppression (TempShift ±2 h; tags whose controlling phase is HELD). | `ai/anomaly/detector.py` |
| 4.8 | Training plus a model manifest (data hash, library versions, seed) saved with joblib to the `models` volume; `ai/train_all.py`; bootstrap trains when the manifest hash differs. | `ai/anomaly/train.py`, `ai/train_all.py` |
| 4.9 | Service shell: subscribes to `pharmaco/#` (it has no ACL access to `_sim`), a rolling buffer per unit, the per-batch context from Neo4j on `events/batch` start, scoring every 5 simulated minutes; publishes the score and alerts. Compose service `anomaly`. | `ai/anomaly/service.py` |
| 4.10 | `notebooks/model_eval.ipynb`: detection rate per fault, lead time against the true spec breach, false alerts per clean batch. | `notebooks/model_eval.ipynb` |

**Tests:**

- **Feature parity:** `windows()` on a replayed live buffer equals the training features for the same batch.
- **Alert lifecycle:** state-machine tests.
- **Fault harness** (`tests/harness.py`, `@pytest.mark.slow`): simulate ~30 clean batches with `engine` and `edge.core` in-process, using `multiprocessing`, cached per session. Fit, then run one seeded batch per fault type through `Detector` and assert:
  - an alert of the expected layer and class opens before the simulator's true spec-breach time;
  - a clean batch produces fewer than 1 alert on average over 5 seeds.
- Budget under 60 s. If the budget is missed, cache the fitted fixture under `.pytest_cache`, keyed by a hash of `simulator/` and `ai/`.

**Done when:** the live demo works — pH drift with the PV flat, and the `stats-co2_flow` alert opening before the true pH leaves the band, visible as a graph Event; DO fouling with PCA contributions naming agitation and DO. The notebook meets the targets, or records honestly where it doesn't.

## Increment 5 — Yield, recommendations, dashboard (size: L)

**Goal:** the full 10-minute demo script.

| # | Task | Files |
| --- | --- | --- |
| 5.1 | `yield_/dataset.py`: one row per batch per day (days 3–12), non-ABORTED only. Trajectory summaries via `ai/features`, levers via `common/levers.extract_levers`, alert counts from `uns_events` (never labels), planned harvest day. Grouped train/test split **by batch**. | `ai/yield_/dataset.py` |
| 5.2 | `train.py`: a PLS baseline; 20 bootstrap LightGBM members; P10/P50/P90 quantile models; SHAP summary saved to the `models` volume. RMSE and MAPE per batch day. | `ai/yield_/train.py` |
| 5.3 | `optimize.py` (pure): `open_levers(day, actual)` → the Optuna TPE search space from the PARs; `paired_gain(ensemble, x_current, x_candidate)`; the gate (P10 > 0 and median ≥ 0.1 g/L). A fixed trial budget, e.g. 200, fits within the 6-hour publish cadence. | `ai/yield_/optimize.py` |
| 5.4 | Service: publishes the prediction every 6 simulated hours and publishes or clears the retained recommendation; PARs come from the graph once per batch. Compose service `yield`. | `ai/yield_/service.py` |
| 5.5 | Ground-truth check in the notebook: for a sample of mid-batch snapshots, the gap between `truth.simulate_remaining` under the recommended levers and under the true optimum (grid search). | `notebooks/model_eval.ipynb` |
| 5.6 | Dashboard: the five pages from the architecture doc; a background MQTT subscriber via `st.cache_resource`; live panels via `st.fragment(run_every=…)`; the **Demo sidebar** (batch, fault, speed, run-to-day, DCS console citing a recommendation id). It holds the only `_sim/cmd` write credential. | `dashboard/app.py`, `dashboard/pages/*` |

**Tests:** the dataset has no leakage (no `fault_labels` column reachable; the split is by batch); `open_levers` freezes the shift day once `day ≥ shift_day`; the gate stays closed on a synthetic ensemble whose gains straddle zero and opens on one clearly positive; the recommendation always lies within the PARs (hypothesis property test); the operator action produces an `events/operator` message and an `ACTED_ON` relationship. Compose smoke test (`-m compose`): start a batch, and assert rows in TimescaleDB, nodes in Neo4j, and a prediction on the broker.

**Done when:** a presenter can run the demo script from the architecture doc end to end from the dashboard alone, and the notebook reports RMSE by day plus the optimizer-versus-truth gap.

## Critical path and parallelism

```
0 ─► 1a (engine, after the 1.1 spike) ─► 1c edge core ─► 2 backfill ─► 3 graph/projection ─► 4 anomaly ─► 5 yield
             └─► 1b shell / ctl ─────────┘                               └─► 5.6 dashboard (can start in 2)
```

- **1.1 (the engine speed spike) is the gating risk.** Backfill time, harness time and demo smoothness all depend on it.
- The dashboard's Live and UNS-browser pages can start as soon as increment 1 publishes. Its Graph page can follow increment 3.
- Increment 4 only needs `tag_attribution` from increment 3 for suppression and training joins. For early work, a stub projection by operation time window unblocks it.

## Risks to watch

| Risk | Early signal | Response |
| --- | --- | --- |
| Engine too slow in Python | Spike 1.1 above 2 s per batch | 15 s integration step (amend ADR-0009); `multiprocessing` everywhere |
| TimescaleDB too large on a laptop | More than 3 GB after backfill | Compression (already in 2.1), then `topic_id` |
| Maturity alignment too noisy with ~120 clean batches per age bin | Clean-batch false alerts above 1 | Widen age bins to 2 h; smooth the bin statistics |
| Recommendation gate never opens | Notebook: gate-open rate about 0 | Check lever effect sizes in `process.py` *against the truth function*, not by loosening the gate |
| Fault harness above 60 s | CI timing | Fixture cache keyed by code hash |
| Streamlit and MQTT threading | UI freezes | Subscriber thread plus a thread-safe snapshot dict; no MQTT calls inside render |

## Tracking

Record measured budgets (engine time per batch, backfill time, DB size, harness time) and the evaluation headline numbers below as they come in, so later changes can be compared.

| Metric | Budget | Measured |
| --- | --- | --- |
| Engine, 1 batch at 5 s step | ≤ 2 s | 1.25 s process + control only; 2.07 s including all 313k raw samples (2026-09-23) |
| Backfill, 200 batches, 8 cores | < 10 min | 86 s on 12 cores (76 s simulate, 9 s compress); clean `docker compose up` to bootstrap complete: 91 s (2026-09-23) |
| TimescaleDB size after compression | ≤ 3 GB | 141 MB database, 115 MB `tag_values` for 7.24M rows (2026-09-23) |
| Fault harness | < 60 s | — |
| False alerts per clean batch | < 1 | — |
