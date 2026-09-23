"""Worker process pools for bulk jobs (backfill, training, evaluation).

Always `spawn`, never `fork`: a forked child inherits the parent's threads' state. After
LightGBM has run in the parent, its OpenMP (libgomp) thread pool makes a forked child
that touches OpenMP hang forever; the Neo4j driver and MQTT clients keep threads too.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def pool(
    initializer: Callable[..., Any] | None = None, initargs: tuple = (), workers: int | None = None
) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=workers or max(1, (os.cpu_count() or 2) - 1),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    )
