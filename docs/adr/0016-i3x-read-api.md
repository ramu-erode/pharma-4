# ADR-0016: A read-only i3X 1.0 façade is the standard read API

Status: Accepted
Date: 2026-09-24

## Context

The stack has no standard way to read it. The dashboard queries the broker,
TimescaleDB and Neo4j directly, so every new consumer has to learn three schemas and
the rules that sit between them: which store is the source of truth for what
(ADR-0004), that `ts` is plant time (ADR-0009), and that ground truth must never
reach an AI consumer (ADR-0012).

CESMII released i3X 1.0 in June 2026: an open HTTP API for contextualised
manufacturing data. It standardises five things, and we already hold all of them:
object types, objects, relationships, current values and history. CESMII ships a
conformance suite, a reference server and an MCP server that lets an LLM client
browse any conformant server.

1.0 is new and adoption is early, so building on it is a bet on the standard. A
façade is a cheap bet: it adds no new store and moves no data.

## Decision

**A new `i3x` service serves i3X 1.0 over the three stores, read-only.** It is a
façade, not a store. Each store answers the questions it already owns:

| i3X concept | Source |
| --- | --- |
| Object types, objects, relationships | Neo4j (the plant model and batch context), cached for a few seconds |
| Current value | A last-value cache fed by the broker (`pharmaco/#`) |
| History | TimescaleDB: `tag_values` for numbers, `uns_events` for structured payloads |
| Subscriptions (sync) | The same broker feed, queued per subscription |

**Scope: the required core only.** It covers `info`, namespaces, object and
relationship types, objects, related objects, current value, history, and sync
subscriptions. The optional parts return 501 and `/info` declares them false.
`PUT /objects/value|history` is out because writes would open a second control path
around ADR-0012 and ADR-0015. SSE streaming is out because it adds moving parts
without adding capability. The expected verdict from CESMII's suite is
**1.0 Compatible**.

**Address space.** ElementIds are human-readable and stable:

- Plant hierarchy (`HasParent`/`HasChildren`): enterprise → site → area → line →
  unit. The elementIds are UNS path prefixes, e.g. `pharmaco/chennai/upstream/suite-1/BR-101`.
- The unit's physical model (`HasComponent`/`ComponentOf`): unit → equipment
  modules → control modules → tags, plus the unit's sensors, state, lab and AI data
  points. **A data point's elementId is its UNS topic**, so reading
  `…/BR-101/pv/ph` through i3X and subscribing to it over MQTT name the same thing.
  Reading a unit with `maxDepth: 0` returns its whole live state in one call.
- Batch context (`HasChildren`), under three root folders:
  - `batches` → each batch → its alerts, operator actions and recommendations
  - `recipes`
  - `phase-classes`

  A batch's value carries its recipe, its planned and actual levers, its outcome and
  its operation and phase timeline. Operations and phases are not separate objects.
- Graph relationships from Neo4j, each with its registered reverse: `RanOn`,
  `FollowsRecipe`, `ConcernsTag`, `ActedOn`, `Measures`, `Controls` and `Monitors`.
  `Controls`/`Monitors` carry the phase binding role of ADR-0011, so "which phase
  controls pH and which only watches it" can be answered through i3X alone.

**Values.** Timestamps are simulated plant time, RFC 3339 UTC (ADR-0009). The
current value of a static object is stamped with the plant-time high-water mark.
Quality maps `GOOD`→`Good` and `UNCERTAIN`→`Uncertain`. `BAD` becomes `Bad` with a
null value, as the spec requires. A data point that has not reported yet is
`GoodNoData`, and so is a unit with no batch running.

**The ground-truth fence carries over.** The façade gets its own broker user, which
may read `pharmaco/#` but not `_sim/#`. Its graph queries match explicit labels and
never touch `FaultInjection`, and its SQL never reads `fault_labels`. Tests enforce
all three. Anything an i3X client sees, an AI may see.

**Auth.** Every endpoint except `GET /info` needs the API key from `I3X_API_KEY`,
sent as `X-API-Key` or as a bearer token. Plain HTTP on localhost, as for the rest of
the POC. TLS is on the out-of-scope list.

## Consequences

- There is one documented way to read the plant. A new consumer, human or LLM,
  learns i3X, not our schemas.
- CESMII's MCP server (Node.js) can point at the façade from a desktop client. It
  runs outside the stack, so ADR-0003 is untouched. The conformance suite also needs
  Node, and runs only as a dev-time check.
- The dashboard keeps its direct store queries. Moving it onto i3X would be churn
  without a new capability.
- History has no aggregation in i3X 1.0, and a 14-day batch has ~20k points per tag.
  The façade caps a history read per element and returns 206 when it truncates.
  Clients such as the assistant downsample on their side.
- Batch, alert and recipe objects come from a graph snapshot refreshed every few
  seconds, so they can lag the broker slightly. Only data points change on
  subscriptions.
- If i3X stalls as a standard, the façade is one service to delete; nothing depends
  on it but the assistant (ADR-0017).

## Alternatives considered

- **A bespoke REST API.** It would be just as thin and would fit us better, but every
  consumer would learn a one-off schema. That is exactly the problem.
- **OPC UA.** Heavier, binary, and aimed at control-level access rather than
  contextualised enterprise reads.
- **Write support (`PUT`).** It would let the conformance suite reach "Full 1.0
  Compliance". Rejected: a second write path to the plant needs a validation story
  we don't have, and ADR-0012 routes every operator action through the DCS stand-in.
- **Operations, phase instances and holds as i3X objects.** That would mean about
  5,000 more objects, most of them never read. Folding them into the batch's value
  answers the same questions in one read.
