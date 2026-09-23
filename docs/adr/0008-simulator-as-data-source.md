# ADR-0008: A simulator, not a real DCS, is the POC data source

Status: Accepted
Date: 2026-09-23

## Context

The POC needs data with real structure: plausible process dynamics, faults to detect,
and a yield outcome that actually depends on how the batch was run. Real plant data is
not available, and would not come with labelled faults or a known ground-truth yield
function even if it were.

Without labels there is no way to say whether the anomaly detection works. Without a
known yield function there is no way to check whether the optimizer finds the optimum
or just a plausible-looking answer.

## Decision

A **Python simulator** models a 14-day CHO fed-batch run on two bioreactors
(BR-101, BR-102). Cell growth follows a Monod/logistic model with glucose uptake and
lactate production. Final titer depends on the integral of viable cells, the
temperature-shift day, the pH band held, feed timing and DO stability, plus noise —
so the optimizer's levers are real and learnable.

It publishes the tags a PI site would have: temperature, pH, DO, agitation, sparge,
cumulative base and feed, pressure, weight, plus daily lab results (VCD, viability,
glucose, lactate, titer).

**Faults are injected and labelled**: pH probe drift, DO sparger fouling, temperature
control loss, feed pump failure, stuck sensor, and rare contamination. About 15% of
the 200 backfilled historical batches carry one.

Because the ground-truth yield function is known, the POC can check whether the
optimizer finds the true optimum — something no real dataset would allow.

## Consequences

- Detection rate, detection lead time and false-alert rate are all measurable against
  known labels. Evaluation is real even though the data is not.
- Demos are reproducible and a fault can be injected live on cue.
- Model performance figures are **not** evidence the approach works on real data. Any
  client-facing material must say so.
- The simulator becomes a maintained artifact: new fault types mean new dynamics, and
  every fault needs a test asserting the anomaly layer catches it.
- Swapping in real data is scoped by ADR-0005: the input side of the edge adapter
  changes, nothing downstream does.

## Alternatives considered

- **Public bioprocess datasets.** Real but small, unlabelled for faults, and with no
  way to inject a fault during a live demo.
- **Replaying a recorded trace.** No fault labels, no counterfactual, and the
  optimizer could not be evaluated at all.
- **Connecting to a real DCS or PI instance.** Not available, and would put the POC
  inside a client's validated environment, which is the wrong place to experiment.
