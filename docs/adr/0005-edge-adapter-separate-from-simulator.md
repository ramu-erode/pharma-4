# ADR-0005: A separate edge adapter between raw tags and the UNS

Status: Accepted; amended by ADR-0009
Date: 2026-09-23

## Context

The simulator could publish straight into clean UNS topics. That would be one fewer
service and one fewer hop.

But it would hide the step that matters most when this pattern meets a real plant. A
DCS or historian exposes tags as `BR101.AIC-102.PV` with no unit, no batch, no
hierarchy. Turning those into `pharmaco/chennai/upstream/suite-1/BR-101/pv/ph` with
unit, batch id and quality attached is the actual work of adopting a UNS, and it is
where the effort goes on a real project.

If the simulator emitted finished topics, the POC would be demonstrating a conclusion
without showing the step that produces it.

## Decision

The simulator emits **raw DCS-style tags** to `edge/raw/<device>/<tag>` with a flat
payload. A separate **edge adapter** subscribes there and:

1. looks the tag up in `edge/tag-map.yaml`,
2. enriches it with unit, current batch id (from `state/batch`) and quality,
3. applies a per-tag deadband — publish on change beyond the deadband, or at least
   once every 60 s,
4. publishes retained to the UNS topic.

Unmapped tags go to `edge/unmapped` rather than being dropped, so a new DCS tag
surfaces during commissioning. The adapter publishes a heartbeat to
`_meta/edge/status`.

Low-rate context data — `lab/*`, `state/*`, `events/*` — bypasses the adapter. The
simulator publishes it directly, standing in for LIMS and MES, which in a real plant
emit data that already carries context.

## Consequences

- The `edge/raw` branch makes the before-and-after visible in MQTT Explorer, which is
  a demo beat in its own right.
- Swapping in a real source changes only the adapter's input: an OPC UA client or a PI
  Web API poller replaces the `edge/raw/#` subscription, and the map, enrichment,
  deadband and output are untouched.
- The deadband means the UNS carries report-by-exception traffic like a real plant,
  not a fixed 5-second tick.
- One more service and one more hop of latency. Irrelevant at this scale.
- The batch id stamped by the adapter is a **cache, not the truth**. It can be wrong
  for late or replayed data. See ADR-0006.

## Alternatives considered

- **Simulator publishes UNS topics directly.** Simpler, but removes the step the POC
  exists to demonstrate.
- **Mapping inside each consumer.** Every consumer would then need the tag map, and
  they would drift.
