# ADR-0021: Anomaly models per equipment class, yield models per process

Status: Accepted
Date: 2026-09-26
Refines: ADR-0015 (lever effects from the DoE), ADR-0013 (alerts)

## Context

The anomaly and yield services were written for the bioreactor: fixed feature lists,
`Growth`/`Production` as the scored operations, batch age since inoculation as the
alignment axis, titer as the target. ADR-0018 adds five equipment classes and two
processes whose batches move between units.

## Decision

**Anomaly detection runs per equipment class.** An `AnomalyProfile` holds everything
that was a module constant:

- the signals and window features;
- the feature-to-tag map;
- the scored operations and the alignment axis;
- the settling rule;
- the rules layer's limit features and rate-of-change rule;
- the fault-class heuristic.

The bioreactor profile is the old constants, unchanged, so its models and harness
results are unchanged. There is one model per class that has something to score:
`bioreactor` (Growth, Production), `reactor` (Reaction, Crystallization),
`filter_dryer` (Filtration, Drying), `roller_compactor` (Compaction) and
`tablet_press` (Compression).

- The new classes align on **hours since the operation started**. Their operations
  are driven by recipe steps, not by a culture's age. A blend is too short for
  30-minute windows, so the blender is not scored.
- Each unit is detected on its own. When a batch arrives on a unit (its first
  operation change out of `Idle`), the service starts that unit's window series.
  Alerts are published on that unit.
- Models are stored as `anomaly-<class>`, and the evaluation report as
  `anomaly-<class>_eval.json`.

**Yield prediction and advice run per process.** A `YieldProfile` holds:

- the target (bioreactor titer in g/L, API yield in %, tablet yield in %);
- the levers and their exposure function;
- which levers are open at which stage, and the minimum change and gain;
- when predictions run;
- the feature function.

The bioreactor profile is the old behaviour. API predictions run hourly from Reaction
to the end of Drying, and tablet predictions every 30 minutes through Compaction and
Compression. Each process has its own DoE response surface (ADR-0015), fitted to its
own clean characterisation runs, and its own gate:

| Process | Min gain | Frozen once |
| --- | --- | --- |
| bioreactor | 0.1 g/L | shift day: at the shift |
| API | 0.5 % yield | Ac2O ratio: dosing starts; reaction temperature and hold: Reaction ends; cooling rate: Crystallization ends |
| OSD | 0.5 % yield | lubrication time: Lubrication ends; roll force: Compaction ends |

The tablet surface is fitted to ln(100 − yield) and transformed back. Tablet losses
(rejects for capping, weight and chipping) compound multiplicatively, so a quadratic in
raw yield misfits the characterisation corners and its bootstrap band swamps the gain.
Measured on fresh tab-v1 batches: with raw yield the gate never opened, although the
true gain was 1.0–1.3%. On the loss scale it opened on all four and predicted 1.2× the
true gain.

Models are `yield-<process>`.

**Payloads become target-neutral.** `ai/yield/prediction` carries
`{target, value: P10/P50/P90, batch_day, model_version}`, and a recommendation names
its `target`. `batch_day` is days since the batch's alignment reference (inoculation
for the bioreactor, batch start otherwise). The unit field names the target's unit.

## Consequences

- One anomaly service and one yield service still serve the whole enterprise. Each
  loads the models it has and ignores units whose class it has no model for.
- A new equipment class needs a profile, simulator faults and harness cases, but no
  new service.
- Bootstrap trains and evaluates every profile whose history is large enough. With a
  small dev backfill (`BACKFILL_*_BATCHES`), a process without enough clean or DoE
  batches is skipped with a warning rather than failing the stack.

## Alternatives considered

- **One model per process across its units.** A batch is on one unit at a time, and
  the units share no signals, so a joint model would be several independent models
  sharing a file.
- **Separate services per site.** That would mirror a federated deployment, but it
  would duplicate the lifecycle code for no demo value. Nothing stops splitting them
  later, because profiles are data.
