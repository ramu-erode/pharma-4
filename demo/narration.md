# pharma-4 demo: narration script

Read over `pharma-4-demo.mp4` (4:29). Times are where each scene starts. The on-screen
captions carry the same points, so the narration can be looser than this.

| Time | Scene | Narration |
| --- | --- | --- |
| 0:00 | Title | This is pharma-4, a proof of concept for Pharma 4.0 data architecture. It simulates a fed-batch bioreactor growing CHO cells that make a monoclonal antibody. It then shows what becomes possible once that data lives in a unified namespace and a knowledge graph. |
| 0:09 | Agenda | We'll start a batch, inject a fault the sensor itself hides, watch the AI catch it, look at the yield prediction, and then explore the context behind it all. |
| 0:18 | Live page, idle | Two 2000-litre bioreactors, BR-101 and BR-102, both idle. The sidebar is our demo control. It stands in for the MES and the operator's DCS console, and it talks to the simulator over MQTT. |
| 0:35 | Start batch | I start a batch on BR-101 with the current recipe. Every sensor, setpoint, phase and event now flows through the broker. |
| 0:43 | Run to day 4 | The clock runs at 3600 times real time, so a second is a simulated hour. Everything in the stack runs on plant time, which is why fourteen days can pass in a few minutes. |
| 0:52 | Day 4, Growth | We're on day 4, in the Growth operation. Operations run in sequence, but the control phases for temperature, pH, DO and feed run in parallel inside each one. |
| 1:02 | Trends | These trends have been through the edge adapter. It maps raw DCS tags like BR101.AIC-102.PV onto ISA-95 topics and adds the unit, batch and quality to each value. |
| 1:12 | Inject pH drift | Now the fault. The pH probe starts to drift upward. The control loop holds the *measured* pH at its setpoint, so the *true* pH quietly falls. On screen, everything looks fine. |
| 1:27 | Ground truth | The simulator records the fault as ground truth, but the broker's access rules keep the AI services away from it. They have to find the fault from process data alone. |
| 1:36 | Evidence | A few simulated hours later, pH still sits on its setpoint. |
| 1:44 | Gas flows | But look at the gas flows. The loop keeps adding CO₂ to hold that reading while base use stays flat. No single-tag alarm on pH would ever fire here. |
| 2:02 | Alerts | The anomaly service caught it. Both the multivariate PCA layer and the statistical EWMA layer opened alerts, and every one suggests pH probe drift. Each alert lists the tags behind it: CO₂ flow and base use. |
| 2:12 | Anomaly index | The index is normalised so 1.0 is the threshold. An alert opens after two windows over it and clears after four under, which keeps the noise down. |
| 2:22 | Evaluation | The models are scored offline against the hidden fault labels, for detection rate, lead time before the true spec breach, and false alerts per clean batch. |
| 2:40 | Yield | From day 3, a gradient-boosted ensemble predicts final titer every six simulated hours, with a calibrated band that narrows as the batch runs. |
| 2:49 | Recommendation | The optimizer looks for setpoint changes: shift day, temperature, pH, DO, feed. It only speaks up when the designed experiment says the gain is real, at least 0.1 grams per litre and unlikely to be zero. For this batch, nothing clears that bar, so it stays quiet. |
| 3:04 | Drivers | SHAP shows what drives the remaining gain. And the AI is advisory: a person changes any setpoint, and the graph links that action to the advice it followed. |
| 3:17 | Graph | Context lives in Neo4j: the equipment hierarchy, recipes and spec limits, every batch's operations, phases, alerts and outcome. |
| 3:26 | Phase bindings | This query shows which phase controls each tag on BR-101 and which only monitors it. That binding is how a value gets attributed to the right phase. |
| 3:36 | Genealogy | And here is our batch's genealogy, including the alerts we just watched open. |
| 3:49 | UNS browser | Finally, the unified namespace itself: one ISA-95 topic tree, where every payload carries value, plant timestamp, unit, quality, batch and source. |
| 4:02 | Ownership | Each branch has exactly one owner, enforced by the broker. |
| 4:11 | i3X and Ask | The same plant is also served read-only through the i3X standard API. An LLM assistant can answer questions through it and never sees the ground truth. |
| 4:21 | Close | All of it comes up with one command, docker compose up. It's a learning vehicle, not a validated GMP system, and the AI is advisory only. |
