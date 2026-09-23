# ADR-0015: Lever effects come from the designed experiment, not the in-batch model

Status: Accepted
Date: 2026-09-23
Supersedes: ADR-0014

## Context

ADR-0014 used the mid-batch titer model (LightGBM, fitted to every historical batch,
with the levers as inputs) as the optimizer's surrogate. It gated recommendations on
the ensemble's paired gain.

The ground-truth check that ADR-0014 asked for showed this does not work. At day 4 on
fresh batches:

- predicted gains ran about **2.8× the true gains**;
- the gate opened on 8 of 8 batches, including all 4 on the near-optimal current
  recipe, where the true gain was about zero.

The advice pointed the right way, but the gate could not tell a batch with room to
improve from one without. The bias was shared by every ensemble member, so neither
paired gains nor splitting the ensemble into search and judge halves removed it. Two
things cause it:

- **Confounding.** Across history, lever settings move together with recipe version
  and time.
- **Unseen inputs.** The optimizer changed levers while holding an observed trajectory
  fixed, which produces inputs the model never saw.

The history already holds the right evidence: the process-characterisation campaign.
Its lever settings were randomised across the PARs by design (a Latin hypercube), which
is exactly how QbD separates cause from correlation.

## Decision

Carried over from ADR-0014 unchanged:

- **Advisory only.** The AI services never write to control. A person decides, through
  the DCS console, citing the recommendation (ADR-0012).
- **Bounded by PARs** read from the graph once per batch (ADR-0011).
- **Output goes back into the UNS:** `ai/anomaly/score`, `ai/anomaly/alert/<key>`
  (ADR-0013), `ai/yield/prediction`, `ai/yield/recommendation`. Alerts and
  recommendations become graph nodes.
- **Explainable, layered anomaly detection:** rules, EWMA, PCA T²/SPE and Isolation
  Forest.
- **Levers are explicit inputs of the mid-batch titer model**, which still produces the
  displayed P10/P50/P90 band. Its hold-out error narrows from 0.49 g/L at day 3 to
  0.27 at day 12.
- **The paired-gain gate:** P10 > 0 and median ≥ 0.1 g/L, and some lever must
  actually move.
- **Optimizer quality is scored against the simulator's ground truth**
  (`python -m ai.yield_.evaluate`).

New:

- **Lever effects come from a response surface fitted to the clean DoE runs.** It is a
  quadratic in the five levers, coded to [-1, 1] over the PARs, fitted with Huber loss
  and a light ridge penalty (`ai/yield_/rsm.py`).
  - **Clean DoE runs** are process-characterisation batches that ended COMPLETE with
    no rules-layer alert. A process deviation excludes a DoE run, as it would in a QbD
    study. No fault labels are read.
  - **At least 25 runs** are required; the quadratic has 21 terms.
- **Uncertainty comes from 200 bootstrap refits** over those runs. Half drive the
  search and the other half judge the winner, and the judges' paired gains feed the
  gate.
- **Local moves only.** The search stays within a trust region of 25% of each PAR's
  width around the current setpoints. A global quadratic misfits the true surface far
  from the data: its global optimum, followed, *lost* 0.5 g/L against the current
  recipe.
- **Mid-batch changes count only for what is left.** A change made on day d is scored
  as a proportionally smaller whole-batch change, by the remaining fraction of that
  lever's window (`rsm.exposure`).

## Consequences

- Measured on the same check (4 batches each on recipes v1 and v3, day 4):
  - the gate opened 5 of 8 times, and **all 5 were truly worth ≥ 0.1 g/L**;
  - on the near-optimal recipe it stayed closed where the true gain was about zero;
  - predicted gains now run about **1.4×** the true ones, against 2.8× before.
- Recommendations rest on 46 runs of a designed experiment, which is the evidence a
  process owner would accept for moving a validated setpoint. More DoE runs make the
  surface better; more routine batches do not.
- Two models are involved: one for the in-batch band, one for lever effects. The
  dashboard shows both. The recommendation payload's `predicted_recommended` is the
  band model's current prediction plus the surface's judged gain.
- The trust region means one recommendation moves a lever by at most a quarter of its
  PAR. Reaching a distant optimum takes several successive recommendations, each
  checked again.
- A plant without a characterisation campaign in its history gets no recommendations.
  That is the honest outcome.

## Alternatives considered

- **Keep ADR-0014 and state the limitation.** Rejected: a gate that opens on
  near-optimal batches is not decision support.
- **Discount predicted gains by a factor estimated on held-out batches.** It treats the
  symptom; the confounding stays.
- **Fit the surface to all batches, DoE and manufacturing.** Measured: worse. The
  manufacturing batches bring back the confounding, and the optimum moved to the PAR
  edges.
- **A global surface optimum instead of local moves.** Measured: wrong far from the
  data.
