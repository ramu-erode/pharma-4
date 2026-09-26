"""Evaluate each anomaly model against the ground-truth labels in history (plan task
4.10, ADR-0021).

    python -m ai.anomaly.evaluate

One report per equipment class, over the faults labelled on units of that class. Faulty
batches never enter training, so detection figures are out of sample. The false-alert
rate here is measured on the model's own training batches (in sample); the out-of-sample
figure comes from the fault harness in tests/ai. Writes
`<MODELS_DIR>/anomaly-<class>_eval.json` for the dashboard.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from ai import store
from ai.anomaly import train
from ai.offline import from_history, to_input
from ai.profiles import PROFILES
from common import pools
from common.plant import get_plant
from common.settings import Settings, get_settings
from historian.migrate import connect

log = logging.getLogger("evaluate")

_settings: Settings | None = None


def _init(settings: Settings) -> None:
    global _settings
    _settings = settings


def unit_of(cls: str) -> str | None:
    """The unit whose slice of a batch a class's model scores: None for the bioreactor
    (the batch's own unit), otherwise the train's one unit of that class."""
    return None if cls == "bioreactor" else get_plant().cells(cls=cls)[0]


def _score(args: tuple[str, str, str | None]) -> dict:
    assert _settings is not None
    cls, batch_id, cell = args
    model = store.load(_settings.models_dir, store.anomaly_name(cls))
    with connect(_settings.postgres_dsn) as conn:
        c = from_history(conn, batch_id, cell)
    result = train.run(model, to_input(batch_id, c, profile=model.profile))
    opens = [ch for ch in result.changes if ch.state == "OPEN"]
    out = {"batch_id": batch_id, "cell": c.cell, "alerts": len(opens), "faults": []}
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


def evaluate(settings: Settings, cls: str = "bioreactor") -> dict | None:
    """Evaluate one class's model; None if it has no model."""
    name = store.anomaly_name(cls)
    manifest = store.manifest(settings.models_dir, name)
    if manifest is None:
        return None
    cells = get_plant().cells(cls=cls)
    with connect(settings.postgres_dsn) as conn:
        # Only finished batches the historian holds from start to end (not a live batch).
        faulty = [
            (cls, r[0], r[1])
            for r in conn.execute(
                "SELECT DISTINCT f.batch_id, f.cell FROM fault_labels f "
                "JOIN uns_events e ON e.batch_id = f.batch_id "
                "AND e.payload -> 'v' ->> 'kind' = 'BATCH_END' WHERE f.cell = ANY(%s)",
                (cells,),
            )
        ]
    clean = [(cls, b, unit_of(cls)) for b in manifest["trained_on"]]
    with pools.pool(_init, (settings,)) as pool:
        fault_results = list(pool.map(_score, faulty))
        clean_results = list(pool.map(_score, clean))

    per_type: dict[str, list[dict]] = defaultdict(list)
    for r in fault_results:
        for f in r["faults"]:
            per_type[f["fault"]].append(f)
    report = {
        "model": manifest["version"],
        "equipment_class": cls,
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
    Path(settings.models_dir, f"{name}_eval.json").write_text(json.dumps(report, indent=2))
    return report


def evaluate_all(settings: Settings) -> dict[str, dict]:
    return {cls: report for cls in PROFILES if (report := evaluate(settings, cls)) is not None}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for report in evaluate_all(get_settings()).values():
        _print(report)


def _print(report: dict) -> None:
    print(f"\n{report['equipment_class']}: model {report['model']}")
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
