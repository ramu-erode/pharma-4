# ADR-0001: MQTT Unified Namespace as the integration backbone

Status: Accepted; amended by ADR-0009
Date: 2026-09-23

## Context

A pharma plant's data is spread across a DCS, a historian, LIMS, MES and ERP. The
usual way to join them is point-to-point integration, which grows as N² and leaves
every new consumer waiting on an integration project. The POC has to demonstrate an
alternative that a client could actually adopt.

We also need several consumers — history, graph, anomaly detection, yield
prediction, dashboard — reading the same live data without coordinating with each
other or with the producer.

## Decision

All data flows through a single MQTT broker organised as a **Unified Namespace**.
Producers publish once; consumers subscribe independently. No service calls another
service to fetch process data.

The topic tree follows the **ISA-95 equipment hierarchy**:

```
pharmaco/<site>/<area>/<line>/<cell>/<class>/<name>
```

so a topic identifies what a value is without a lookup table. Current state is
**retained** on the broker, so a new or restarted subscriber sees every current value
immediately rather than waiting for the next update.

The AI services publish their output back into the same namespace under an `ai/*`
branch, which makes alerts and predictions ordinary UNS data that anything else can
consume.

## Consequences

- Adding a consumer costs nothing on the producer side. This is the point we want to
  demonstrate.
- The broker becomes a single point of failure. Acceptable for a POC; production
  needs clustering (see `docs/architecture.md`).
- Topic design becomes a governed artifact. A sloppy tree is worse than no tree, so
  `common/uns.py` owns topic construction and the ACL enforces ownership.
- Retained messages mean stale values can look live after a producer dies. The
  `ts` field in every payload and the edge heartbeat mitigate this.

## Alternatives considered

- **Direct database writes from each producer.** Simpler for one consumer, but every
  new consumer means a new pipeline, which is the problem we are trying to show a way
  out of.
- **Kafka.** Better for replay and high throughput, heavier to run, and far less
  common on the OT side of a pharma plant. MQTT is what the field devices and OT
  tooling speak.
- **Reading PI directly from every service.** Couples every consumer to PI's API and
  its availability, and does not generalise to LIMS or MES data.
