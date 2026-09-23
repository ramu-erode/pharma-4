# ADR-0007: AI output is advisory and published back into the UNS

Status: Superseded by ADR-0014
Date: 2026-09-23

## Context

The POC's two AI use cases both produce something a plant could act on: an anomaly
alert and a setpoint recommendation. Closing the loop — letting the optimizer write
setpoints to the controller — would be a striking demo.

It would also be the wrong thing to build. In a GMP plant a change to a validated
setpoint goes through review, and a system that writes to control is squarely in
scope for computerised system validation. Beyond compliance, a model trained on 200
simulated batches has no business moving a real bioreactor.

There is also a design question: where do the AI results go? A private database
would make them invisible to everything else.

## Decision

**Advisory only.** The AI services never write to control. Recommendations name the
current and suggested values and the expected gain; a person decides.

Recommendations are bounded by the recipe's **proven acceptable ranges**, read from
the graph, so the optimizer cannot suggest leaving the validated design space even in
principle. No recommendation is emitted when the expected gain is smaller than the
model's own P10–P90 spread.

AI output is **published back into the UNS** under the equipment's `ai/*` branch:
`ai/anomaly/score`, `ai/anomaly/alert`, `ai/yield/prediction`,
`ai/yield/recommendation`. Alerts are picked up by graph-sync and become `Event`
nodes, closing the loop into the context model.

Every alert carries what drove it: the score, which detection layer fired, and the top
contributing tags. Anomaly detection runs in three layers — rules, univariate
statistics (EWMA/CUSUM), and multivariate (PCA with T²/SPE alongside Isolation
Forest) — partly so there is always an explainable account of why something fired.

## Consequences

- The POC stays outside the GxP decision path, which keeps the validation
  conversation simple: this is decision support.
- AI results become ordinary UNS data. The dashboard, the graph and any future
  consumer read them the same way they read a sensor.
- Alerts becoming graph events is what makes "which alerts preceded low-titer
  batches?" answerable, which is the payoff query of the whole demo.
- Explainability is a design constraint on model choice, not an afterthought. PCA with
  T² and SPE is kept partly because it is the multivariate SPC method biopharma
  already uses and regulators recognise.
- We forgo the closed-loop demo.

## Alternatives considered

- **Closed-loop setpoint writes.** Rejected on validation grounds and because the
  models do not deserve that trust.
- **AI results to their own database or API.** Rejected: it would break the
  publish-once principle of ADR-0001 and make the results invisible to other
  consumers.
- **A single best-performing model.** Rejected in favour of layered detection, because
  an unexplainable alert is not actionable in this domain.
