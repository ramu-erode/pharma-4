# ADR-0014: AI output is advisory; recommendations are gated on paired gain

Status: Superseded by ADR-0015
Date: 2026-09-23
Supersedes: ADR-0007

## Context

ADR-0007 made the AI advisory and published its output back into the UNS. Both still
hold, and are restated below so this ADR stands alone. Two parts of it do not survive
contact with the design.

**The gate compares the wrong quantity.** ADR-0007 withholds a recommendation when the
expected gain is smaller than the model's P10–P90 spread. That spread is the
uncertainty of the *absolute* final titer, likely ±0.4 g/L or more mid-batch, while a
realistic gain from one lever change is smaller. The gate would almost never open.
Most of the error is also shared between the "current" and "recommended" predictions
and cancels in their difference.

**The surrogate can't see the levers.** The mid-batch model's features summarise what
has already happened, so varying a future setpoint changes none of its inputs, and the
optimizer would search a flat landscape.

## Decision

Carried over from ADR-0007 unchanged:

- **Advisory only.** The AI services never write to control. A person decides.
- **Bounded by PARs.** Recommendations stay inside the recipe's proven acceptable
  ranges, read from the graph once per batch (ADR-0011).
- **Output goes back into the UNS.** `ai/anomaly/score`, `ai/anomaly/alert/<key>`
  (ADR-0013), `ai/yield/prediction` and `ai/yield/recommendation`. Alerts and
  recommendations become graph nodes.
- **Explainable, layered anomaly detection:** rules, EWMA/CUSUM, and PCA (T²/SPE)
  alongside Isolation Forest. Every alert names its layer, score and top contributing
  tags.

New:

- **Levers are explicit model inputs.** Each training row (batch, day d) = trajectory
  summaries up to d + the batch's *actual whole-batch* lever values. At inference,
  levers whose window has passed are frozen at their actual values, and the optimizer
  (Optuna TPE) searches only the open ones.
- **Paired-gain gate.** A bootstrap ensemble (20 LightGBM members) computes, per
  member, gain = f(recommended) − f(current). A recommendation is published only if
  the ensemble's P10 gain is above 0 **and** its median gain is at least 0.1 g/L.
  Quantile models still produce the displayed P10/P50/P90 titer band.
- **Recommendations are retained and replaceable.** A new one overwrites the old, and
  an empty retained message clears it when the gate closes.
- **graph-sync also subscribes to `ai/yield/recommendation`**, creating
  `:Recommendation` nodes that operator actions link to (ADR-0012).
- **Optimizer quality is scored against ground truth.** `simulator.truth` gives the
  true titer for any lever set. Offline evaluation reports how close the recommended
  levers get to the true optimum.

## Consequences

- The recommendation beat in the demo can actually fire, for a reason that holds up
  statistically.
- Training needs lever variation across the history. This is supplied by recipe
  versions and a characterisation DoE block (architecture.md).
- Mid-batch changes to an open lever are approximated by the whole-batch value the
  model was trained on. This is a known limitation, recorded in the evaluation.
- Twenty models to train instead of one. Cheap at this data size.

## Alternatives considered

- **Keep ADR-0007's gate.** Honest, but in practice silent.
- **Tune the simulator until gains beat the absolute spread.** Bakes an unrealistic
  process into the demo.
- **Separate lever-only model for optimization.** Ignores how this particular batch is
  going.
- **Optimize by running a fitted mechanistic model forward.** Close to fitting the
  simulator to itself.
