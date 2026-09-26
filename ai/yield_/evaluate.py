"""The optimizer against ground truth (ADR-0015, plan task 5.5), for every process
(ADR-0021).

    python -m ai.yield_.evaluate [--batches 4]

For fresh simulated batches of each process's oldest and current recipe, stop mid-batch
(the bioreactor on day 4, the API half an hour into Reaction, tablets half an hour into
Compaction), ask for a recommendation, and let the simulator say what following it
*truly* gains (`simulator.truth`). Reports predicted versus true gains and how the gate
decided. Writes `<MODELS_DIR>/yield-<process>_eval.json` for the dashboard.

This check is only possible because the plant is simulated (ADR-0008). It exists to
catch a surrogate that is confidently wrong, which a hold-out RMSE cannot show.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ai import store
from ai.offline import Collector
from ai.yield_ import optimize
from ai.yield_ import profiles as yp
from ai.yield_.features import YieldInput, features_at
from common import pools, uns
from common.plant import get_plant
from common.settings import Settings, get_settings
from edge.core import Deadband, Mapped, TagMap, map_and_enrich
from simulator import batches as b
from simulator import campaign, recipes, truth
from simulator import train_campaign as tc
from simulator.engine import BatchRun
from simulator.messages import RawOut
from simulator.processes import Run, Spec, make_run
from simulator.train import TrainRun
from simulator.wire import to_wire

Sink = Callable[[str, object], None]
START = datetime(2026, 10, 1, tzinfo=UTC)
# Where a train batch stops for advice: this many hours into this operation.
STOP = {"api": ("Reaction", 0.5), "osd": ("Compaction", 0.5)}


def _feed(run: Run, until_h: float, tag_map: TagMap, sink: Sink) -> None:
    """Advance `run` and pass what the UNS would carry to `sink(topic, payload)`."""
    unit = get_plant().path(run.spec.cell)
    cells = run.cells if isinstance(run, TrainRun) else (run.spec.cell,)
    cache: dict[str, str | None] = dict.fromkeys(cells)
    deadband = Deadband(get_settings().deadband_floor_s, get_settings().publish_period_s)
    for msg in run.advance(until_h):
        if isinstance(msg, RawOut):
            mp = map_and_enrich(msg.tag, msg.value, msg.t, msg.q, tag_map, cache)
            if isinstance(mp, Mapped) and deadband.offer(mp) and mp.batch is not None:
                sink(mp.topic, mp)
            continue
        wire = to_wire(unit, run.spec.batch_id, msg)
        if wire:
            if wire[0].endswith("/state/batch"):
                cache[uns.parse(wire[0]).unit.cell] = wire[1].v
            sink(*wire)


def bio_case(args: tuple[Settings, b.BatchSpec, float]) -> dict:
    settings, spec, day = args
    model = store.load(settings.models_dir, store.yield_name("bioreactor"))
    run = BatchRun(spec, publish_period_s=settings.publish_period_s)
    col = Collector(spec.start)

    def sink(topic: str, payload: object) -> None:
        if isinstance(payload, Mapped):
            col.value(topic, payload.ts, payload.value, payload.q.value)
        else:
            col.message(topic, payload)

    _feed(run, b.t_of_day(day), TagMap.load(), sink)
    c = col.collected()
    x = features_at(YieldInput(c.series, c.ctx, c.lab, c.levers, []), day)
    advice = optimize.recommend(
        model, x, c.levers, day, run.operation.value, recipes.get(spec.recipe_id).par, seed=1
    )
    now = truth.simulate_remaining(run, c.levers)
    rec = truth.simulate_remaining(run, c.levers.model_copy(update=advice.recommended))
    return _row(spec, advice, rec - now)


def train_case(args: tuple[Settings, str, Spec]) -> dict:
    settings, process, spec = args
    model = store.load(settings.models_dir, store.yield_name(process))
    run = make_run(spec, publish_period_s=settings.publish_period_s)
    col = yp.TrainCollector(spec.start)

    def sink(topic: str, payload: object) -> None:
        if isinstance(payload, Mapped):
            col.value(topic, payload.ts, payload.value)
        else:
            col.message(topic, payload)

    op, after = STOP[process]
    tag_map = TagMap.load()
    while not run.done and not (run.operation.value == op and run.op_age() >= after):
        _feed(run, run.t_h + 0.1, tag_map, sink)
    day = run.t_h / 24.0
    inp = col.inp
    x = yp.TRAIN_FEATURES[process](inp, day)
    at = yp.at_for(process, inp, day)
    advice = optimize.recommend(
        model, x, inp.levers, day, at.operation, recipes.get(spec.recipe_id).par, seed=1, at=at
    )
    now = truth.train_remaining(run, inp.levers)
    rec = truth.train_remaining(run, inp.levers.model_copy(update=advice.recommended))
    return _row(spec, advice, rec - now)


def _row(spec: Spec, advice: optimize.Advice, true_gain: float) -> dict:
    return {
        "batch": spec.batch_id,
        "recipe": spec.recipe_id,
        "gate_open": advice.passes_gate,
        "predicted_gain_p10": float(np.quantile(advice.gain, 0.1)),
        "predicted_gain_p50": float(np.median(advice.gain)),
        "true_gain": true_gain,
        "recommended": advice.recommended,
    }


def evaluate(
    settings: Settings, per_recipe: int = 4, day: float = 4.0, process: str = "bioreactor"
) -> dict | None:
    """Evaluate one process's optimizer; None if it has no yield model."""
    name = store.yield_name(process)
    manifest = store.manifest(settings.models_dir, name)
    if manifest is None:
        return None
    profile = yp.PROFILES[process]
    versions = recipes.of_process(process)
    chosen = [versions[0].id, versions[-1].id]  # the oldest and the current recipe
    if process == "bioreactor":
        specs = [s for r in chosen for s in campaign.manufacturing(r, per_recipe, START, 99)]
        jobs, fn = [(settings, s, day) for s in specs], bio_case
    else:
        specs = [s for r in chosen for s in tc.manufacturing(process, r, per_recipe, START, 99)]
        jobs, fn = [(settings, process, s) for s in specs], train_case
    with pools.pool() as pool:
        cases = list(pool.map(fn, jobs))
    opened = [c for c in cases if c["gate_open"]]
    significant = [c for c in cases if c["true_gain"] > profile.min_gain / 5]
    report = {
        "model": manifest.get("version"),
        "process": process,
        "unit": profile.unit,
        "min_gain": profile.min_gain,
        "stop": f"day {day:g}"
        if process == "bioreactor"
        else f"{STOP[process][1]:g} h into {STOP[process][0]}",
        "day": day,
        "cases": cases,
        "gate_opened": len(opened),
        "gate_opened_and_truly_worth_it": sum(c["true_gain"] >= profile.min_gain for c in opened),
        "median_predicted_over_true": float(
            np.median([c["predicted_gain_p50"] / c["true_gain"] for c in significant])
        )
        if significant
        else None,
    }
    Path(settings.models_dir, f"{name}_eval.json").write_text(json.dumps(report, indent=2))
    return report


def evaluate_all(settings: Settings, per_recipe: int = 4) -> dict[str, dict]:
    return {
        p: r for p in yp.PROFILES if (r := evaluate(settings, per_recipe, process=p)) is not None
    }


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m ai.yield_.evaluate")
    p.add_argument("--batches", type=int, default=4, help="per recipe")
    args = p.parse_args()
    for report in evaluate_all(get_settings(), args.batches).values():
        unit = report["unit"]
        print(f"\n{report['process']} (stopped at {report['stop']})")
        for c in report["cases"]:
            print(
                f"{c['recipe']} {c['batch']}: predicted gain P50 {c['predicted_gain_p50']:+.2f} "
                f"(P10 {c['predicted_gain_p10']:+.2f}), true {c['true_gain']:+.2f} {unit}, "
                f"gate {'OPEN' if c['gate_open'] else 'closed'}"
            )
        ratio = report["median_predicted_over_true"]
        print(
            f"gate opened {report['gate_opened']}/{len(report['cases'])}, of which truly worth "
            f"it (true gain >= {report['min_gain']:g} {unit}): "
            f"{report['gate_opened_and_truly_worth_it']}; predicted/true gain, median: "
            + (f"{ratio:.1f}x" if ratio is not None else "n/a")
        )


if __name__ == "__main__":
    main()
