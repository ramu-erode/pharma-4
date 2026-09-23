# ADR-0013: Typed `v` for structured payloads, and a keyed alert lifecycle

Status: Accepted
Date: 2026-09-23

## Context

ADR-0002 says every payload carries `v, ts, unit, q, batch, src`, one value per topic.
That fits `pv/ph`. It does not obviously fit alerts, recommendations, batch events or
operator actions, which carry structured content.

`ai/anomaly/alert` was also one retained topic per unit. Two concurrent alerts would
overwrite each other, nothing ever cleared an alert, and a fault staying above
threshold for hours would emit an alert every window. That floods the graph with
Events and makes "false alerts per batch" meaningless.

## Decision

**All six keys, on every payload, no exceptions.** `v` is typed per topic class in
`common/models.py`:

| Topic class | `v` type | `unit` |
| --- | --- | --- |
| `pv`, `sp`, `lab`, `ai/anomaly/score` | float | engineering unit |
| `state/*` | string enum | null |
| `events/*`, `ai/*/alert/*`, `ai/yield/*` | nested pydantic model | null (or the model's unit, e.g. `g/L`) |

`src` is an enum: `sim`, `edge`, `anomaly`, `yield`, `operator`. A test pushes a sample
of every topic class through `common/uns.py` and its model.

**Alerts are keyed and stateful.** Each alert has an `id` and a `state` of
`OPEN` → `CLEARED`, and is published retained to `ai/anomaly/alert/<key>`, where
`<key>` = `<layer>-<tag or fault class>` (e.g. `stats-co2_flow`). Hysteresis: it opens
after 2 consecutive scoring windows over threshold and clears after 4 under. On clear,
the service publishes `CLEARED`, then an empty retained message to remove it from the
broker. graph-sync `MERGE`s one Event per `id` and updates it on clear. One lifecycle
counts as one alert in evaluation.

## Consequences

- Consumers read `ts` and `batch` the same way on every topic.
- Several alerts can be open on a unit at once, and a restarted dashboard sees exactly
  the open set.
- `common/uns.py` and the ACL gain one extra topic level under `ai/anomaly/alert/`.
- Hysteresis adds up to two windows of detection delay. Evaluation measures lead time
  from alert open, so the cost is visible.

## Alternatives considered

- **Envelope + body for structured messages.** Cleaner typing, but it breaks the
  one-sentence rule.
- **Flatten structured messages into scalar topics.** Loses atomicity: a consumer can
  see half an alert.
- **One retained topic holding a JSON list of open alerts.** Consumers would have to
  diff the list.
- **Stateless alert events.** Duplicates in the graph, and no "currently open" after a
  restart.
