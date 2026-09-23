# ADR-0002: Mosquitto with JSON payloads, not Sparkplug B

Status: Accepted
Date: 2026-09-23

## Context

Two choices sit underneath ADR-0001: which broker, and what goes on the wire.

Sparkplug B is the industrial MQTT convention. It defines birth and death
certificates, a compact protobuf payload and a prescribed topic namespace, and it is
what many UNS deployments standardise on. It also adds tooling, a schema step and a
topic structure we would have to work around to keep the ISA-95 tree readable.

The POC has to run on one laptop and be readable by someone opening MQTT Explorer for
the first time.

## Decision

Use **Eclipse Mosquitto 2.x** with **JSON payloads**, one value per topic:

```json
{ "v": 7.03, "ts": "2026-09-22T10:15:05.000Z", "unit": "pH",
  "q": "GOOD", "batch": "B2026-0142", "src": "sim" }
```

`q` uses OPC UA-style quality (`GOOD`, `UNCERTAIN`, `BAD`). Payload shapes live in
`common/models.py` as pydantic models, which is our schema enforcement in place of
Sparkplug's.

QoS and retain are set per topic class: QoS 0 retained for high-rate `pv/*` and
`sp/*`, QoS 1 retained for `lab/*` and `state/*`, QoS 1 non-retained for `events/*`,
QoS 1 retained for open `ai/*` alerts and recommendations.

Mosquitto runs with username/password and an ACL file so each service can publish
only to the branch it owns.

## Consequences

- Anyone can read the wire format in MQTT Explorer without a decoder. For a demo
  whose purpose is explanation, this matters more than efficiency.
- Payloads are larger than protobuf and there is no birth/death certificate, so
  producer liveness needs its own mechanism (the edge heartbeat).
- We lose the interoperability Sparkplug buys with commercial OT tooling. If a client
  standardises on Sparkplug, the edge adapter is the single place that changes.
- One value per topic keeps the tree browsable and lets consumers subscribe narrowly,
  at the cost of more messages than a grouped payload.

## Alternatives considered

- **HiveMQ CE or EMQX with Sparkplug B.** Closer to a real plant and better at scale.
  Rejected for the POC as setup cost with no demonstrative benefit; named as the
  production upgrade path.
- **JSON with many values per message.** Fewer messages, but consumers then filter in
  code and the topic stops identifying the value.
