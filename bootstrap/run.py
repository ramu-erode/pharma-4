"""One-shot, idempotent first start (ADR-0010). Takes a clean checkout to demo-ready:

1. apply TimescaleDB migrations
2. apply the Neo4j schema and load configuration (enterprise, sites, units, bindings,
   tags, recipes)
3. backfill the historical batches of every site, unless already complete, and hand the
   Freiburg API stock left at the end to the live simulator on `_sim/opening_stock`
   (ADR-0020)
4. replay into the graph any batch the historian has and the graph lacks, and project
   its attribution (so a Neo4j reset heals itself)
5. train the AI models (anomaly per equipment class, then yield per process) if their
   training data changed (ADR-0021)
6. write the models' evaluation reports (the dashboard shows them) when missing or stale

    python -m bootstrap.run

Running it again is a no-op, apart from re-publishing the retained opening stock.
Services that need its results declare
`depends_on: bootstrap: condition: service_completed_successfully`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ai import store, train_all
from ai.anomaly import evaluate as anomaly_evaluate
from ai.profiles import PROFILES as ANOMALY_PROFILES
from ai.yield_ import evaluate as yield_evaluate
from ai.yield_.profiles import PROFILES as YIELD_PROFILES
from ai.yield_.train import train_yield_all
from common import models as m
from common import uns
from common.mqtt import connect as mqtt_connect
from common.settings import Settings, get_settings
from graph import db as graph_db
from graph import load, replay
from historian.migrate import connect, migrate
from simulator import backfill

log = logging.getLogger("bootstrap")
SERVICE = "bootstrap"


def reports_current(models_dir: str) -> bool:
    """Each evaluation report names the model version it evaluated; a report written for
    another version (or cut short) is stale."""
    names = [store.anomaly_name(c) for c in ANOMALY_PROFILES]
    names += [store.yield_name(p) for p in YIELD_PROFILES]
    for name in names:
        manifest = store.manifest(models_dir, name)
        if manifest is None:
            continue  # not trained (too little history): nothing to report
        path = Path(models_dir) / f"{name}_eval.json"
        if not path.exists() or json.loads(path.read_text()).get("model") != manifest.get(
            "version"
        ):
            return False
    return True


def publish_opening_stock(settings: Settings, stock: dict | None) -> None:
    """Hand the stock at the end of history to the live simulator (ADR-0020). A simulator
    with no inventory of its own adopts it, whenever it arrives."""
    if stock is None:
        return
    payload = m.InventoryPayload(
        v=m.Inventory.model_validate(stock), ts=m.now_utc(), unit="kg", batch=None, src=m.Src.SIM
    )
    try:
        client = mqtt_connect(SERVICE, m.Src.SIM, settings, presence=False)
    except Exception:
        log.warning("broker unavailable: the opening stock was not handed over")
        return
    try:
        client.publish(uns.sim_opening_stock(), payload).wait_for_publish(timeout=10)
        log.info("opening stock: %d lots at Freiburg", len(payload.v.lots))
    finally:
        client.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = get_settings()
    t0 = time.perf_counter()
    with connect(settings.postgres_dsn, wait_s=120) as conn, graph_db.connect(settings) as driver:
        log.info("step 1/6: TimescaleDB migrations")
        migrate(conn)
        log.info("step 2/6: graph schema and configuration")
        graph_db.run_each(driver, load.schema_statements())
        graph_db.run_all(driver, load.config_statements())
        log.info("step 3/6: backfill (%d batches)", settings.backfill_total)
        backfill.run(conn, settings)
        publish_opening_stock(settings, backfill.get_state(conn, "opening_stock"))
        missing = replay.missing_batches(conn, driver)
        log.info("step 4/6: graph replay (%d batches missing)", len(missing))
        if missing:
            replay.replay(conn, driver, missing)
        log.info("step 5/6: model training")
        retrained = train_all.train_anomaly_all(conn, settings)
        retrained = train_yield_all(conn, settings) or retrained
        if retrained or not reports_current(settings.models_dir):
            log.info("step 6/6: evaluation reports")
            anomaly_evaluate.evaluate_all(settings)
            yield_evaluate.evaluate_all(settings, per_recipe=4)
        else:
            log.info("step 6/6: evaluation reports are current")
    log.info("bootstrap complete in %.0f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
