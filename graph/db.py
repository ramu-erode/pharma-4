"""Neo4j connection and statement execution, shared by graph-sync, replay and bootstrap."""

from __future__ import annotations

import time
from collections.abc import Iterable

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from common.settings import Settings
from graph.core import Stmt


def connect(settings: Settings, wait_s: float = 120.0) -> Driver:
    """Open a driver, retrying while Neo4j starts."""
    # Single-node lookups on unique keys trip Neo4j's informational 'cartesian product'
    # notice; only warnings are worth logging.
    driver = GraphDatabase.driver(
        settings.bolt_uri,
        auth=settings.neo4j_credentials,
        notifications_min_severity="WARNING",
    )
    deadline = time.monotonic() + wait_s
    while True:
        try:
            driver.verify_connectivity()
            return driver
        except ServiceUnavailable:
            if time.monotonic() > deadline:
                raise
            time.sleep(2.0)


def run_all(driver: Driver, stmts: Iterable[Stmt]) -> None:
    """Run statements in one write transaction, in order."""
    stmts = list(stmts)
    if not stmts:
        return

    def work(tx) -> None:
        for query, params in stmts:
            tx.run(query, params).consume()

    with driver.session() as session:
        session.execute_write(work)


def run_each(driver: Driver, stmts: Iterable[Stmt]) -> None:
    """Run statements in auto-commit mode (schema DDL cannot share a transaction)."""
    with driver.session() as session:
        for query, params in stmts:
            session.run(query, params).consume()
