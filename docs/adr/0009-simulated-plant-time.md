# ADR-0009: Simulated plant time in `ts`, wall-clock time for liveness

Status: Accepted; amended by ADR-0018
Date: 2026-09-23
Amends: ADR-0001 (consequences), ADR-0005 (deadband floor)

## Context

The simulator runs faster than real time (up to 3600×: a 14-day batch in about six
minutes). Until now the design never said which clock `ts` carries, nor which clock the
deadband floor, heartbeats and staleness checks run on. It needs to be one answer,
because everything downstream builds on it:

- ADR-0001 says the `ts` field detects stale retained values. That only works if
  `ts` is close to wall-clock time.
- Trends, features and the batch calendar only make sense in process time.
- At a 5 s sample period and 3600×, raw traffic is about 19k msg/s, which is more
  than a Python edge adapter should be asked to carry for a demo.
- Backfilled history and live batches share one calendar, and `ts` must never go
  backwards per topic, because the deadband, retained state and hypertable ordering
  all assume it doesn't.

## Decision

**`ts` is simulated plant time.** The source stamps it, and the edge adapter passes it
through unchanged. Anything that measures process time counts in `ts`: the deadband
floor, feature windows, batch days and alert hysteresis.

**Liveness runs on wall-clock time.** Every service publishes a retained
`pharmaco/_meta/<service>/status` heartbeat every 30 s of wall-clock time, and
registers an MQTT last-will that marks it `offline`. Staleness is judged from the
heartbeat, never from `ts`.

**The publish period is decoupled from the integration step.** The simulator
integrates on a 5 s step and publishes raw samples on a configurable period in
simulated time, 60 s by default (roughly 1.8k msg/s across both reactors at 3600×).
The deadband floor ("publish at least every N") defaults to 10 simulated minutes and
must be longer than the publish period. The adapter refuses to start if it isn't.

**The clock is monotonic across restarts.** On start, simulator time is
`max(wall-clock now, latest ts in retained state/* on the broker)`. Speed is set at
runtime (pause, 1×, 60×, 600×, 3600×) through `_sim/cmd/clock` (ADR-0012).

**Calendar and ids.** Batch ids are `B<start year>-<global 4-digit sequence>`, issued
by the simulator as the MES stand-in and never reused. Backfill creates B…-0000 to
B…-0199, ending the day before first boot, so the first live batch is B2026-0200.

## Consequences

- The ADR-0001 consequence "the `ts` field … mitigates [staleness]" no longer holds.
  The per-service heartbeat and last-will replace it.
- Live `ts` runs ahead of wall-clock time once a fast demo has been going for a while.
  Nothing may compare `ts` with `now()`.
- The architecture's "every 5 s" becomes "configurable, 60 s by default in demo
  mode". Feature windows (30 min) still hold 30 grid points.
- A stuck-sensor rule has to account for the deadband: see architecture, anomaly
  layer 1.

## Alternatives considered

- **Wall-clock `ts`, with process time in a separate field.** Staleness via `ts`
  keeps working, but every trend, feature and join would have to key on the second
  field.
- **Both clocks in every payload.** Precise, but it changes the six-key contract every
  service shares.
- **Keep 5 s sampling at 3600× and optimise the hot path.** Spends effort on
  throughput, which the POC isn't about.
