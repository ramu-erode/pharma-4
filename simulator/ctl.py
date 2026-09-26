"""Demo control from a terminal: the same `_sim/cmd/*` messages the dashboard sends.

    python -m simulator.ctl start-batch BR-101 [--recipe v3] [--campaign PC]
    python -m simulator.ctl start-batch RX-201            (Tuas API train)
    python -m simulator.ctl start-batch BL-301            (Freiburg tablet line)
    python -m simulator.ctl inject BR-101 ph_probe_drift [--param rate=0.01]
    python -m simulator.ctl inject RX-201 jacket_fouling
    python -m simulator.ctl clear BR-101 ph_probe_drift
    python -m simulator.ctl speed 600 | pause | resume
    python -m simulator.ctl run-to-day BR-101 4
    python -m simulator.ctl setpoint BR-101 prod_temp 33.5 [--rec R-1] [--reason ...]
    python -m simulator.ctl setpoint TP-303 comp_force 14
    python -m simulator.ctl abort BR-101

A person issues these, so they carry `src = operator` and connect as the dashboard's
MQTT user, the only one allowed to write `_sim/cmd` (ADR-0012).
"""

from __future__ import annotations

import argparse
import getpass
import sys

from pydantic import BaseModel

from common import models as m
from common import uns
from common.mqtt import connect
from common.settings import get_settings
from common.uns import SimCommand

USER = "dashboard"


def _params(pairs: list[str]) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def build(args: argparse.Namespace) -> tuple[SimCommand, BaseModel]:
    match args.cmd:
        case "start-batch":
            return SimCommand.BATCH, m.BatchCommand(
                action="start", cell=args.cell, recipe=args.recipe, campaign=args.campaign
            )
        case "abort":
            return SimCommand.BATCH, m.BatchCommand(action="abort", cell=args.cell)
        case "inject" | "clear":
            return SimCommand.FAULT, m.FaultCommand(
                action=args.cmd, cell=args.cell, fault=args.fault, params=_params(args.param)
            )
        case "speed":
            return SimCommand.CLOCK, m.ClockCommand(action="speed", speed=args.speed)
        case "pause" | "resume":
            return SimCommand.CLOCK, m.ClockCommand(action=args.cmd)
        case "run-to-day":
            return SimCommand.CLOCK, m.ClockCommand(
                action="run_to_day", cell=args.cell, day=args.day
            )
        case "setpoint":
            return SimCommand.SETPOINT, m.SetpointCommand(
                cell=args.cell,
                parameter=args.parameter,
                value=args.value,
                operator=args.operator,
                recommendation_id=args.rec,
                reason=args.reason,
            )
    raise SystemExit(f"unknown command {args.cmd}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m simulator.ctl", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start-batch")
    s.add_argument("cell")
    s.add_argument("--recipe")
    s.add_argument("--campaign", default="MFG", choices=[c.value for c in m.Campaign])
    sub.add_parser("abort").add_argument("cell")
    for name in ("inject", "clear"):
        f = sub.add_parser(name)
        f.add_argument("cell")
        f.add_argument("fault", choices=[x.value for x in m.FaultType])
        f.add_argument("--param", action="append", default=[], help="key=value, repeatable")
    sub.add_parser("speed").add_argument("speed", type=float)
    sub.add_parser("pause")
    sub.add_parser("resume")
    r = sub.add_parser("run-to-day")
    r.add_argument("cell")
    r.add_argument("day", type=float)
    sp = sub.add_parser("setpoint")
    sp.add_argument("cell")
    sp.add_argument(
        "parameter", choices=[k for lm in m.LEVER_MODELS.values() for k in lm.model_fields]
    )
    sp.add_argument("value", type=float)
    sp.add_argument("--rec", help="recommendation id this change follows")
    sp.add_argument("--reason")
    sp.add_argument("--operator", default=getpass.getuser())
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    kind, command = build(args)
    client = connect(USER, m.Src.OPERATOR, get_settings(), presence=False)
    try:
        payload = m.SIM_COMMAND_MODELS[kind](
            v=command, ts=m.now_utc(), unit=None, batch=None, src=m.Src.OPERATOR
        )
        client.publish(uns.sim_cmd(kind), payload).wait_for_publish(timeout=5)
        print(f"sent {kind.value}: {command.model_dump_json()}")
    finally:
        client.close()


if __name__ == "__main__":
    main(sys.argv[1:])
