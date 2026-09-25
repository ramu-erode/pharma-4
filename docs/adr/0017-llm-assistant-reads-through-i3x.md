# ADR-0017: The LLM assistant reads the plant only through i3X

Status: Accepted
Date: 2026-09-24
Refines: ADR-0015 (advisory AI)

## Context

Parked increment 6 was LLM Q&A over the graph and history: "why was batch 142
low?" answered from context, alerts and trends. The obvious build gives the model a
Cypher tool and a SQL tool. That is also the riskiest build. The model would have to
learn our schemas. It could reach `FaultInjection` nodes or `fault_labels`, the
ground truth that ADR-0012 keeps away from every AI. And the answers would
demonstrate nothing about interoperability.

ADR-0016 puts a standard, fenced read API in front of the stores. It is a much
better foundation for an LLM than raw query tools.

## Decision

**The assistant's only view of the plant is the i3X API.** Its tools mirror i3X
calls one to one:

- list types
- find objects
- get objects
- get related objects
- read current values
- read history, downsampled on the client side with min/max/mean per bucket

It has no database driver and no broker client. A test fails if the `assistant`
package imports one. Whatever the façade won't serve, the assistant can't see, so
the ground-truth fence of ADR-0012 holds for the LLM by construction.

**Where it runs.** `assistant/` is a library with a CLI (`python -m assistant`) and
an **Ask** page in the dashboard. It is not a separate service: it holds no state
between questions and serves no other consumer. It calls the Claude API directly
with a hand-written tool loop. The loop is bounded by a fixed number of tool rounds,
and tool results are capped in size with a note telling the model to narrow its
query.

**Model.** `claude-opus-5` by default, with adaptive thinking and server-side
refusal fallbacks (`fallbacks: "default"`), configurable through `ASSISTANT_MODEL`.
The key comes from `ANTHROPIC_API_KEY` in `.env`. It is never in `.env.example`.
Without a key the Ask page explains how to set one, and the rest of the stack runs
unchanged.

**Advisory, and says so.** The system prompt makes the assistant:

- cite the elementIds and plant-time windows it read
- separate what the data shows from what it infers
- never present an answer as a disposition or an instruction to change the process

Answers are not published to the UNS. ADR-0015 publishes AI output that other
services act on, such as alerts and recommendations. A chat answer is read by one
person, so it is ephemeral.

**External LLM clients** use CESMII's i3X MCP server against the same façade. They
see exactly what the built-in assistant sees.

## Consequences

- The demo shows the architecture's point: an LLM that had never seen this plant
  browses it through an open standard and answers from context, not from our table
  names.
- Answers depend on how good the address space is. A question the i3X model can't
  express, such as a cross-batch aggregate across 200 batches, costs many tool calls
  or can't be answered. The fix is a better address space, not a side door to the
  stores.
- There is no evaluation set yet. Answer quality is checked by hand against known
  batches. A scripted fake client tests the loop's mechanics, not the model.
- Running it costs API tokens per question. The system prompt and tools are
  cache-stable, so repeat questions read the prefix from the prompt cache.

## Alternatives considered

- **Cypher and SQL tools.** More capable for aggregates. Rejected: they bypass the
  fence and teach the model our schemas.
- **i3X plus a few preset queries.** A middle ground, but it makes the i3X story
  weaker. Revisit if the address space proves too thin.
- **CESMII's MCP server inside the stack.** Least code, but it would put Node.js in
  a service (ADR-0003). Desktop users can still run it outside the stack.
- **An `assistant` compose service with its own HTTP API.** A second API to design
  for a single caller, the dashboard.
