# pharma-4

A proof of concept for Pharma 4.0 data architecture. A simulated CHO fed-batch
bioreactor publishes into an MQTT **Unified Namespace**. A **Neo4j knowledge graph**
holds the context. Two AI services, **anomaly detection** and **yield
optimization**, publish their results back into the namespace. An **i3X 1.0** API
serves the whole plant through CESMII's open standard, and an **LLM assistant**
answers questions by reading only that API. The AI output is advisory only. Nothing
here is validated.

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

## Read it through i3X

The `i3x` service serves the plant as i3X 1.0 at <http://localhost:8600/v1>
([ADR-0016](docs/adr/0016-i3x-read-api.md)). It is read-only. `GET /info` is open; the
rest needs the key from `.env.example` (`I3X_API_KEY`), sent as `X-API-Key` or as a
bearer token.

```bash
curl localhost:8600/v1/info
curl -H "X-API-Key: dev-i3x-key" -X POST localhost:8600/v1/objects/value \
  -H "Content-Type: application/json" \
  -d '{"elementIds": ["pharmaco/chennai/upstream/suite-1/BR-101"], "maxDepth": 0}'
```

A data point's elementId is its UNS topic. Timestamps are simulated plant time.

**Conformance.** CESMII's suite (needs Node.js 18+) rates the façade *1.0 Compatible*:

```bash
git clone --branch 1.0 https://github.com/cesmii/i3X.git && cd i3X/conformance-tests
node bin/i3x-test.js run http://localhost:8600/v1 --token dev-i3x-key
```

Use `--token`, not `--header`: the suite leaves `--header` credentials on its
"no auth" probe, then warns that auth isn't required.

**From an LLM desktop client.** Add CESMII's MCP server to your client's MCP config:

```json
{"mcpServers": {"pharma-4": {"command": "npx", "args": ["-y", "i3x-mcp@latest"],
  "env": {"I3X_BASE_URL": "http://localhost:8600/v1",
          "I3X_AUTH_SCHEME": "bearer", "I3X_TOKEN": "dev-i3x-key"}}}}
```

## Ask the plant

The dashboard's **Ask** page and the CLI put questions to Claude, which reads the
plant only through i3X ([ADR-0017](docs/adr/0017-llm-assistant-reads-through-i3x.md)).
Put your key in `.env`, never in `.env.example`:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
docker compose up -d dashboard          # the Ask page picks up the key
python -m assistant "Why is the predicted titer of the batch on BR-101 falling?"
```

Each answer lists the i3X calls it made. Without a key, the rest of the stack runs as
usual and the Ask page explains what is missing.

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
