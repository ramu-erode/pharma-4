"""Evaluate the anomaly model against the ground-truth labels in history (plan task 4.10).

    python -m ai.anomaly.evaluate

Faulty batches never enter training, so detection figures are out of sample. The
false-alert rate here is measured on the model's own training batches (in sample); the
out-of-sample figure comes from the fault harness in tests/ai. Writes the report to
`<MODELS_DIR>/anomaly_eval.json` for the dashboard.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ai import store
from ai.anomaly import train
from ai.context import controlled_by_config
from ai.offline import from_history, to_input
from common.settings import Settings, get_settings
from historian.migrate import connect

log = logging.getLogger("evaluate")

_settings: Settings | None = None


def _init(settings: Settings) -> None:
    global _settings
    _settings = settings


def _score(batch_id: str) -> dict:
    assert _settings is not None
    model = store.load(_settings.models_dir, "anomaly")
    with connect(_settings.postgres_dsn) as conn:
        c = from_history(conn, batch_id)
    result = train.run(model, to_input(batch_id, c, controlled_by_config()))
    opens = [ch for ch in result.changes if ch.state == "OPEN"]
    out = {"batch_id": batch_id, "alerts": len(opens), "faults": []}
    for label in c.labels:
        onset = (label.onset - c.series.origin).total_seconds() / 60
        after = [o for o in opens if o.end >= onset]
        out["faults"].append(
            {
                "fault": label.fault.value,
                "detected": bool(after),
                "delay_h": (after[0].end - onset) / 60 if after else None,
                "first_key": after[0].key if after else None,
                "suggested": after[0].evidence.fault_class.value
                if after and after[0].evidence.fault_class
                else None,
                "alerts_before_onset": len(opens) - len(after),
            }
        )
    return out


def evaluate(settings: Settings) -> dict:
    manifest = store.manifest(settings.models_dir, "anomaly")
    if manifest is None:
        raise RuntimeError("no anomaly model; run python -m ai.train_all")
    with connect(settings.postgres_dsn) as conn:
        faulty = [r[0] for r in conn.execute("SELECT DISTINCT batch_id FROM fault_labels")]
    clean = manifest["trained_on"]
    with ProcessPoolExecutor(initializer=_init, initargs=(settings,)) as pool:
        fault_results = list(pool.map(_score, faulty))
        clean_results = list(pool.map(_score, clean))

    per_type: dict[str, list[dict]] = defaultdict(list)
    for r in fault_results:
        for f in r["faults"]:
            per_type[f["fault"]].append(f)
    report = {
        "model": manifest["version"],
        "per_fault": {
            kind: {
                "batches": len(fs),
                "detected": sum(f["detected"] for f in fs),
                "median_delay_h": float(np.median([f["delay_h"] for f in fs if f["detected"]]))
                if any(f["detected"] for f in fs)
                else None,
                "correct_class": sum(f["suggested"] == kind for f in fs),
            }
            for kind, fs in sorted(per_type.items())
        },
        "false_alerts_per_clean_batch_in_sample": float(
            np.mean([r["alerts"] for r in clean_results])
        ),
        "batches": fault_results,
    }
    Path(settings.models_dir, "anomaly_eval.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    report = evaluate(get_settings())
    print(f"model {report['model']}")
    print(
        f"{'fault':22s} {'batches':>7s} {'detected':>8s} {'median delay h':>15s} {'class ok':>8s}"
    )
    for kind, r in report["per_fault"].items():
        delay = "-" if r["median_delay_h"] is None else f"{r['median_delay_h']:.1f}"
        print(
            f"{kind:22s} {r['batches']:7d} {r['detected']:8d} {delay:>15s} {r['correct_class']:8d}"
        )
    print(f"false alerts per clean batch (in sample): "
          f"{report['false_alerts_per_clean_batch_in_sample']:.2f}")  # fmt: skip


if __name__ == "__main__":
    main()
