# ADR-0003: Python for every service, no low-code layer

Status: Accepted
Date: 2026-09-23

## Context

The edge adapter — mapping raw DCS tags into the UNS and enriching them — is the job
Node-RED is usually given in UNS projects. It is the tool most OT engineers recognise,
and its flow canvas demonstrates the edge layer well.

The counterweight: Node-RED means a second language and runtime, a `flows.json` that
is awkward to review in a pull request, logic hidden inside Function nodes, and no
natural place for unit tests. The rest of the stack is Python because the AI work
requires it.

An earlier draft of the architecture included a Node-RED container. It was removed
before implementation started.

## Decision

Every service is Python 3.12. The edge adapter is a small Python service
(`edge/adapter.py` plus `edge/tag-map.yaml`), not a flow.

`ruff` for lint and format, `pytest` for tests, `pydantic` for payload models. Shared
concerns — topic construction, payload models, MQTT connection handling — live in
`common/` and are imported by every service rather than reimplemented.

## Consequences

- The tag map and its edge cases are unit-testable, and the adapter is reviewable as
  ordinary code.
- One runtime, one dependency file, one test command.
- We give up the visual flow that makes the edge layer obvious in a demo. The
  architecture diagram carries that explanation instead.
- If a client already runs Node-RED, this decision does not travel. The adapter's
  contract — subscribe to `edge/raw/#`, publish contextualised values to the UNS — is
  deliberately small enough that a Node-RED flow could replace it without touching
  anything downstream.

## Alternatives considered

- **Node-RED for the edge adapter.** Recognisable to OT audiences and quick to
  prototype. Rejected on maintainability and testability for a repo meant to be read.
- **Node-RED as well as Python, for demo value.** Rejected: two implementations of the
  same contract is worse than either alone.
