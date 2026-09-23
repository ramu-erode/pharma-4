# pharma-4

A proof of concept for Pharma 4.0 data architecture. A simulated CHO fed-batch
bioreactor publishes into an MQTT **Unified Namespace**. A **Neo4j knowledge graph**
holds the context. Two AI services, **anomaly detection** and **yield
optimization**, publish their results back into the namespace. The AI output is
advisory only. Nothing here is validated.

## Run it

```bash
docker compose up -d
```

The first start takes about three minutes. A one-shot `bootstrap` service:

- applies the database schemas;
- generates 200 historical batches (2022–2026);
- builds the graph;
- trains the models and writes their evaluation reports.

Later starts skip whatever is already done.

Open the dashboard at <http://localhost:8501>. From the **Demo** sidebar you can
start a batch, change the speed, run it to a given day, inject a fault, and change a
setpoint the way an operator would.

The same controls work from a terminal:

```bash
python -m simulator.ctl start-batch BR-101
python -m simulator.ctl run-to-day BR-101 2.3
python -m simulator.ctl inject BR-101 ph_probe_drift
python -m simulator.ctl run-to-day BR-101 4
```

To watch the namespace itself, point MQTT Explorer at `localhost:1883`. Use the
read-only user `explorer`; its password is in `.env.example`.

## What is where

| | |
| --- | --- |
| Design | [docs/architecture.md](docs/architecture.md), decisions in [docs/adr](docs/adr/README.md) |
| Build plan and measured numbers | [docs/implementation-plan.md](docs/implementation-plan.md) |
| Conventions for contributors | [CLAUDE.md](CLAUDE.md) |

## Tests

```bash
pip install -r requirements-dev.txt && pip install -e .
pytest              # unit tests and the in-process fault harness (~2 min)
pytest -m compose   # against the running stack
```
