"""The optimizer against ground truth (ADR-0014, plan task 5.5).

    python -m ai.yield_.evaluate [--batches 4] [--day 4]

For fresh simulated batches of each recipe, stop at `day`, ask for a recommendation,
and let the simulator say what following it *truly* gains (`simulator.truth`). Reports
predicted versus true gains and how the gate decided. Writes
`<MODELS_DIR>/yield_eval.json` for the dashboard.

This check is only possible because the plant is simulated (ADR-0008). It exists to
catch a surrogate that is confidently wrong, which a hold-out RMSE cannot show.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ai import store
from ai.offline import Collector
from ai.yield_ import optimize
from ai.yield_.features import YieldInput, features_at
from common import pools
from common.settings import Settings, get_settings
from common.uns import UnitPath
from edge.core import Deadband, Mapped, TagMap, map_and_enrich
from simulator import batches as b
from simulator import campaign, recipes, truth
from simulator.engine import BatchRun, RawOut
from simulator.wire import to_wire


def case(args: tuple[Settings, b.BatchSpec, float]) -> dict:
    settings, spec, day = args
    model = store.load(settings.models_dir, "yield")
    tag_map = TagMap.load(settings.site, settings.area, settings.line)
    unit = UnitPath(settings.site, settings.area, settings.line, spec.cell)
    run = BatchRun(spec, publish_period_s=settings.publish_period_s)
    col = Collector(spec.start)
    deadband = Deadband(settings.deadband_floor_s, settings.publish_period_s)
    cache: dict[str, str | None] = {spec.cell: None}
    for msg in run.advance(b.t_of_day(day)):
        if isinstance(msg, RawOut):
            mp = map_and_enrich(msg.tag, msg.value, msg.t, msg.q, tag_map, cache)
            if isinstance(mp, Mapped) and deadband.offer(mp):
                col.value(mp.topic, mp.ts, mp.value, mp.q.value)
            continue
        wire = to_wire(unit, spec.batch_id, msg)
        if wire:
            if wire[0].endswith("/state/batch"):
                cache[spec.cell] = wire[1].v
            col.message(*wire)
    c = col.collected()
    x = features_at(YieldInput(c.series, c.ctx, c.lab, c.levers, []), day)
    advice = optimize.recommend(
        model, x, c.levers, day, run.operation.value, recipes.get(spec.recipe_id).par, seed=1
    )
    now = truth.simulate_remaining(run, c.levers)
    rec = truth.simulate_remaining(run, c.levers.model_copy(update=advice.recommended))
    return {
        "batch": spec.batch_id,
        "recipe": spec.recipe_id,
        "gate_open": advice.passes_gate,
        "predicted_gain_p10": float(np.quantile(advice.gain, 0.1)),
        "predicted_gain_p50": float(np.median(advice.gain)),
        "true_gain": rec - now,
        "recommended": advice.recommended,
    }


def evaluate(settings: Settings, per_recipe: int, day: float) -> dict:
    specs = [
        s
        for r in ("v1", "v3")
        for s in campaign.manufacturing(r, per_recipe, datetime(2026, 10, 1, tzinfo=UTC), 99)
    ]
    with pools.pool() as pool:
        cases = list(pool.map(case, [(settings, s, day) for s in specs]))
    opened = [c for c in cases if c["gate_open"]]
    report = {
        "day": day,
        "cases": cases,
        "gate_opened": len(opened),
        "gate_opened_and_truly_worth_it": sum(c["true_gain"] >= optimize.MIN_GAIN for c in opened),
        "median_predicted_over_true": float(
            np.median(
                [c["predicted_gain_p50"] / c["true_gain"] for c in cases if c["true_gain"] > 0.02]
            )
        ),
    }
    Path(settings.models_dir, "yield_eval.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m ai.yield_.evaluate")
    p.add_argument("--batches", type=int, default=4, help="per recipe")
    p.add_argument("--day", type=float, default=4.0)
    args = p.parse_args()
    report = evaluate(get_settings(), args.batches, args.day)
    for c in report["cases"]:
        print(
            f"{c['recipe']} {c['batch']}: predicted gain P50 {c['predicted_gain_p50']:+.2f} "
            f"(P10 {c['predicted_gain_p10']:+.2f}), true {c['true_gain']:+.2f}, "
            f"gate {'OPEN' if c['gate_open'] else 'closed'}"
        )
    print(
        f"gate opened {report['gate_opened']}/{len(report['cases'])}, of which truly worth it "
        f"(true gain >= {optimize.MIN_GAIN} g/L): {report['gate_opened_and_truly_worth_it']}; "
        f"predicted/true gain, median: {report['median_predicted_over_true']:.1f}x"
    )


if __name__ == "__main__":
    main()
