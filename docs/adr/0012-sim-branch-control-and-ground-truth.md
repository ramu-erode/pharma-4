# ADR-0012: A `_sim` branch for simulator control and ground-truth labels, fenced by the ACL

Status: Accepted
Date: 2026-09-23

## Context

Two kinds of message exist only because the plant is simulated.

**Control.** The demo has to start a batch on cue, inject a fault on cue, pause at
"day 4", and let an operator change a setpoint. At 3600× nothing waits for the
presenter.

**Ground truth.** Evaluation needs the onset, type and unit of every injected fault.
If labels travel as ordinary `events/*`, the anomaly service could read the answer,
and the yield model's "anomaly events so far" feature could quietly count real faults
instead of alerts.

## Decision

A `_sim/` branch, outside `pharmaco/`, next to `edge/`:

| Topic | Publisher | Purpose |
| --- | --- | --- |
| `_sim/cmd/batch` | dashboard, `simulator.ctl` | start/abort a batch on a unit |
| `_sim/cmd/fault` | dashboard, `simulator.ctl` | inject/clear a fault |
| `_sim/cmd/clock` | dashboard, `simulator.ctl` | pause, resume, speed, run-to-day-N |
| `_sim/cmd/setpoint` | dashboard "DCS console" | manual operator setpoint change |
| `_sim/clock` | simulator | current sim time and speed (retained) |
| `_sim/faults/<unit>` | simulator | ground-truth fault onset/end labels |

**Leakage is prevented by the broker, not by convention.** The Mosquitto ACL grants the
`anomaly` and `yield` users no access to `_sim/#`. The historian stores labels in
`fault_labels`. graph-sync creates `:FaultInjection` nodes, which are never `:Event`.
Evaluation joins labels to alerts after the fact.

**Operator actions are human actions through a DCS stand-in.** A setpoint change via
`_sim/cmd/setpoint` may cite a recommendation id. The simulator applies it and emits
`events/operator` (operator, tag, old → new, recommendation id). graph-sync turns that
into an Event linked to the Recommendation. The recommendation itself has no Apply
button.

## Consequences

- One mechanism serves the dashboard sidebar and the CLI, and nothing calls the
  simulator over HTTP.
- In a real plant `_sim/` doesn't exist: control goes to the DCS operator console and
  there are no labels. Removing the branch touches nothing downstream.
- The graph can trace advice → operator action → outcome, which is the ADR-0007 story
  told end to end.
- The dashboard gets its first write permission: `_sim/cmd/#` only.

## Alternatives considered

- **An HTTP control API on the simulator.** A direct service-to-service call, which
  the architecture otherwise avoids.
- **Label only the backfill.** Live demo injections couldn't be scored for lead time.
- **`events/fault` with a `ground_truth` flag.** The easiest to build and the easiest
  to leak.
- **An Apply button on the recommendation.** Reads as closed loop to a GMP audience.
