# ADR-0006: Attribute process values to phases via equipment binding

Status: Accepted
Date: 2026-09-23

## Context

A process value has to be attributable to the batch and phase it belongs to, or none
of the analysis means anything. This is harder than it looks.

A DCS tag never carries a batch id. `BR101.AIC-102.PV` is pH on a vessel; it reads the
same during a batch, during CIP and while idle. A batch is a *time window on a piece
of equipment*, so a value belongs to a batch because of where and when it was
measured.

Phases make it harder still. In ISA-88, phases can run **in parallel** on one unit:
temperature control, pH control and feed addition all active at once. At any instant
several phases are running, so "the current phase" is not a single value, and time
alone cannot say which phase a given tag belongs to.

## Decision

**Equipment ownership decides phase attribution, not time alone.**

A phase does not own a tag directly. It runs on equipment modules, and those contain
the control modules and their tags. Attribution for tag X on unit U at time t:

1. Find phase instances running on U at t (from the batch journal's start/end events).
2. Keep only those whose phase class is **bound** to X, directly or through X's
   equipment module.
3. If exactly one remains, assign it. If several remain, keep them all with a
   **role** — `control` (the phase drives the loop or writes the setpoint) or
   `monitor` (it only reads the value). If none remain, fall back to the lowest common
   ancestor: the operation step or operation.

Attribution is therefore **many-to-many with a role**, not a single foreign key.
Forcing one phase per value is where this normally goes wrong.

The binding is configuration, not inference. In a DeltaV site it comes from the
**alias** mechanism: phase logic is written against aliases (`#FEED_PUMP#`) that each
unit module resolves to real modules, and both sides are in the FHX export. The chain
is phase class → alias → module → parameter → tag.

Neo4j is the **source of truth** for attribution, because it works for late data,
backfills and corrected timestamps. The `batch` field stamped on a payload by the edge
adapter is a convenience cache for the live path only.

## Consequences

- Correct attribution when phases overlap, which is the normal case on a bioreactor.
- Analysis can distinguish "the phase controlling this loop" from "a phase that merely
  watched it" — the difference between blaming feed addition and blaming pH control.
- The binding table must be maintained and must come from configuration, not a
  spreadsheet. On a real site it is re-derived when the DCS configuration changes.
- Per-value graph queries are too slow for bulk work, so phase intervals are synced to
  a flat table and joined on time (ADR-0004).
- Held phases need their state carried alongside the attribution, so an excursion
  during HOLD is not blamed on active processing.

## Alternatives considered

- **Stamp one `phase` field at the edge.** Fast and simple, but wrong whenever phases
  overlap, and wrong for any replayed or late data.
- **Resolve purely by time window.** Returns every active phase with no way to choose,
  which is the same as no attribution.
- **Ignore phases, attribute only to batch.** Loses the resolution the anomaly models
  need, since Growth and Production behave differently enough to warrant separate
  models.
