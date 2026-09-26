"""Dashboard pages and the Demo sidebar (architecture: dashboard). No side effects on
import: `dashboard/app.py` wires these into navigation, and tests render them one by one.

Live, Alerts, Yield, Graph and UNS browser pages for every site (ADR-0018), plus a Demo
sidebar that drives the simulator through `_sim/cmd` and a DCS console for operator
setpoint changes. There is no Apply button on a recommendation: a person enters the
change (ADR-0012, ADR-0015). The Ask page is the exception to "reads the stores
directly": its assistant reads only through the i3X API (ADR-0017).
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import pandas as pd
import psycopg
import streamlit as st

from ai import store
from ai.profiles import PROFILES as ANOMALY_PROFILES
from ai.yield_.profiles import PROFILES as YIELD_PROFILES
from assistant import agent
from common import models as m
from common import uns
from common.plant import get_plant
from common.settings import Settings, get_settings
from common.uns import SimCommand, TopicClass, UnitPath
from dashboard.live import LiveState
from graph import db as graph_db
from simulator import recipes
from simulator.processes import FAULTS

REFRESH = "3s"
PLANT = get_plant()
SITE_NAMES = {s.id: f"{s.name} ({s.props.get('location', '')})" for s in PLANT.sites}
LINE_OF_SITE = {ln.site.id: ln for ln in PLANT.lines}
PROCESS_NAMES = {
    "bioreactor": "mAb drug substance (Grange Castle)",
    "api": "Aspirin API (Tuas)",
    "osd": "Aspirin 500 mg tablets (Freiburg)",
}
# Each process's batch starts on, and publishes its yield advice on, this unit.
HOME = {ln.process: ln.units[0] for ln in PLANT.lines}
TREND_TAGS: dict[str, dict[str, list[str]]] = {
    "bioreactor": {
        "Temperature (°C)": ["pv/temperature", "sp/temperature"],
        "pH": ["pv/ph", "sp/ph"],
        "DO (% air sat.)": ["pv/do", "sp/do"],
        "Agitation (rpm)": ["pv/agitation"],
        "Gas flows (L/min)": ["pv/air_flow", "pv/o2_flow", "pv/co2_flow"],
        "Totals": ["pv/feed_total", "pv/base_total"],
    },
    "reactor": {
        "Temperature (°C)": ["pv/temperature", "sp/temperature", "pv/jacket_temperature"],
        "Conversion (%, Raman)": ["pv/conversion"],
        "Crystal chord length (µm, FBRM)": ["pv/chord_length"],
        "Anhydride dosing (kg)": ["pv/dose_total"],
        "Agitator power (kW)": ["pv/agitator_power"],
        "Level (%)": ["pv/level"],
    },
    "filter_dryer": {
        "Filtrate flow (L/min)": ["pv/filtrate_flow"],
        "Filtrate total (L)": ["pv/filtrate_total"],
        "Vacuum (mbar)": ["pv/vacuum", "sp/vacuum"],
        "Temperatures (°C)": ["pv/jacket_temperature", "pv/cake_temperature"],
        "Cake moisture (%, NIR)": ["pv/moisture"],
    },
    "blender": {
        "Blend RSD (%, NIR)": ["pv/blend_rsd"],
        "Revolutions": ["pv/revolutions"],
        "Room humidity (% RH)": ["pv/room_rh"],
    },
    "roller_compactor": {
        "Roll force (kN/cm)": ["pv/roll_force", "sp/roll_force"],
        "Ribbon density (g/cm³, NIR)": ["pv/ribbon_density"],
        "Gap (mm)": ["pv/roll_gap"],
        "Feed screw (rpm)": ["pv/screw_speed"],
        "Room humidity (% RH)": ["pv/room_rh"],
    },
    "tablet_press": {
        "Compression force (kN)": ["pv/comp_force", "sp/comp_force"],
        "Ejection force (N)": ["pv/ejection_force"],
        "Tablet weight (mg)": ["pv/tablet_weight"],
        "Weight RSD (%)": ["pv/weight_rsd"],
        "Hardness (N)": ["pv/hardness"],
        "Room humidity (% RH)": ["pv/room_rh"],
    },
}


# --- shared resources ----------------------------------------------------------------------


@st.cache_resource
def settings() -> Settings:
    return get_settings()


@st.cache_resource
def live() -> LiveState:
    return LiveState(settings())


@st.cache_resource
def pg() -> psycopg.Connection:
    return psycopg.connect(settings().postgres_dsn, autocommit=True)


@st.cache_resource
def neo4j():
    return graph_db.connect(settings(), wait_s=30)


def unit(cell: str) -> UnitPath:
    return PLANT.path(cell)


def sql(query: str, params: tuple = ()) -> list[tuple]:
    try:
        return pg().execute(query, params).fetchall()
    except psycopg.Error as exc:
        st.warning(f"TimescaleDB: {exc}")
        return []


def model_file(name: str) -> dict | None:
    path = Path(settings().models_dir) / name
    return json.loads(path.read_text()) if path.exists() else None


def current_batch(cell: str) -> str | None:
    p = live().get(uns.state_batch(unit(cell)))
    return p.v if p is not None else None


def train_batch(cell: str) -> tuple[str | None, str | None]:
    """The batch running anywhere on `cell`'s train, and the unit it is on now."""
    for c in PLANT.train_of(cell):
        batch = current_batch(c)
        if batch:
            return batch, c
    return None, None


def last_batch(cell: str) -> str | None:
    """The running batch, or else the most recent one on this unit."""
    running = current_batch(cell)
    if running:
        return running
    rows = sql(
        "SELECT batch_id FROM uns_events WHERE topic = %s AND batch_id IS NOT NULL "
        "ORDER BY ts DESC LIMIT 1",
        (uns.events(unit(cell), "batch"),),
    )
    return rows[0][0] if rows else None


def site_picker(key: str) -> str:
    return st.selectbox("Site", list(SITE_NAMES), format_func=SITE_NAMES.get, key=key)


# --- Demo sidebar --------------------------------------------------------------------------


def sidebar() -> None:
    lv = live()
    st.sidebar.header("Demo")
    clock = lv.clock()
    if clock:
        state = "paused" if clock.paused else f"{clock.speed:g}×"
        st.sidebar.caption(f"Simulated time {clock.sim_time:%Y-%m-%d %H:%M} UTC · {state}")
    site = st.sidebar.selectbox("Site", list(SITE_NAMES), format_func=SITE_NAMES.get)
    line = LINE_OF_SITE[site]
    process = line.process
    if line.train:
        cell = line.units[0]
        st.sidebar.caption(f"{PROCESS_NAMES[process]}: {' → '.join(line.units)}")
        running, where = train_batch(cell)
        st.sidebar.caption(f"{running} on {where}" if running else "Line idle")
    else:
        cell = st.sidebar.selectbox("Unit", line.units)
        running = current_batch(cell)
        st.sidebar.caption(f"{cell}: {running or 'idle'}")

    c1, c2 = st.sidebar.columns(2)
    versions = [r.id for r in reversed(recipes.of_process(process))]
    recipe = c1.selectbox("Recipe", ["current", *versions])
    if c2.button("Start batch", disabled=running is not None, width="stretch"):
        lv.send(
            SimCommand.BATCH,
            m.BatchCommand(
                action="start", cell=cell, recipe=None if recipe == "current" else recipe
            ),
        )
    speed = st.sidebar.select_slider("Speed", [60, 600, 3600], value=3600)
    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("Speed", width="stretch"):
        lv.send(SimCommand.CLOCK, m.ClockCommand(action="speed", speed=speed))
    if c2.button("Pause", width="stretch"):
        lv.send(SimCommand.CLOCK, m.ClockCommand(action="pause"))
    if c3.button("Resume", width="stretch"):
        lv.send(SimCommand.CLOCK, m.ClockCommand(action="resume"))
    c1, c2 = st.sidebar.columns(2)
    if line.train:
        hours = c1.number_input("Hour", 0.0, 48.0, 3.0, 0.5)
        day = hours / 24.0
    else:
        day = c1.number_input("Day", 0.0, 14.0, 4.0, 0.25)
    if c2.button("Run to", disabled=running is None, width="stretch"):
        lv.send(SimCommand.CLOCK, m.ClockCommand(action="run_to_day", cell=cell, day=day))

    st.sidebar.subheader("Faults")
    fault = st.sidebar.selectbox("Fault", [f.value for f in FAULTS[process]])
    target = st.sidebar.selectbox("On unit", line.units) if line.train else cell
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Inject", disabled=running is None, width="stretch"):
        lv.send(SimCommand.FAULT, m.FaultCommand(action="inject", cell=target, fault=fault))
    if c2.button("Clear", disabled=running is None, width="stretch"):
        lv.send(SimCommand.FAULT, m.FaultCommand(action="clear", cell=target, fault=fault))

    st.sidebar.subheader("DCS console")
    st.sidebar.caption("A person changes a setpoint. Cite the recommendation it follows, if any.")
    rec = lv.get(uns.ai_recommendation(unit(cell)))
    parameter = st.sidebar.selectbox("Parameter", list(m.LEVER_MODELS[process].model_fields))
    advice = rec.v.levers.get(parameter) if rec else None
    value = st.sidebar.number_input(
        "New value",
        value=float(round(advice.recommended, 3)) if advice is not None else 0.0,
        format="%.3f",
    )
    rec_id = st.sidebar.text_input("Recommendation id", value=rec.v.id if rec else "")
    reason = st.sidebar.text_input("Reason", value="")
    if st.sidebar.button("Change setpoint", disabled=running is None, width="stretch"):
        lv.send(
            SimCommand.SETPOINT,
            m.SetpointCommand(
                cell=cell,
                parameter=parameter,
                value=value,
                operator="demo",
                recommendation_id=rec_id or None,
                reason=reason or None,
            ),
        )
        st.sidebar.success("Sent to the DCS stand-in")

    if process in ("api", "osd"):
        stock = lv.get(uns.sim_inventory())
        if stock is not None and clock is not None:
            lots = [lot for lot in stock.v.lots if lot.released <= clock.sim_time]
            st.sidebar.caption(
                f"Freiburg API stock: {sum(lot.quantity_kg for lot in lots):,.0f} kg released "
                f"in {len(lots)} lot(s) (simulation state, `_sim/inventory`)"
            )


# --- pages -----------------------------------------------------------------------------------


def trend(batch_id: str, cell: str, suffixes: list[str]) -> pd.DataFrame:
    topics = [f"{unit(cell).prefix}/{s}" for s in suffixes]
    rows = sql(
        "SELECT ts, topic, value FROM tag_values WHERE batch_id = %s AND topic = ANY(%s) "
        "ORDER BY ts",
        (batch_id, topics),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "topic", "value"])
    df["tag"] = df["topic"].str.replace(f"{unit(cell).prefix}/", "", regex=False)
    return df.pivot_table(index="ts", columns="tag", values="value").sort_index().ffill()


def page_live() -> None:
    st.title("Live")
    lv = live()
    site = site_picker("live_site")
    line = LINE_OF_SITE[site]
    st.caption(f"{PROCESS_NAMES[line.process]} · {line.area.name} · {line.line.name}")

    @st.fragment(run_every=REFRESH)
    def body() -> None:
        cols = st.columns(len(line.units))
        for col, cell in zip(cols, line.units, strict=True):
            u = unit(cell)
            op = lv.get(uns.state_operation(u))
            phases = lv.snapshot(f"{u.prefix}/state/phase/")
            with col:
                st.subheader(cell)
                st.caption(PLANT.unit(cell).type)
                st.metric("Batch", current_batch(cell) or "idle")
                st.metric("Operation", op.v.value if op else "—")
                st.caption(
                    " · ".join(
                        f"{t.rsplit('/', 1)[1].upper()} {p.v.value}"
                        for t, p in sorted(phases.items())
                    )
                    or "no phases"
                )
        cell = st.radio("Trends for", line.units, horizontal=True, key=f"live_trend_{site}")
        batch = last_batch(cell)
        if not batch:
            st.info("No batch on this unit yet. Start one from the Demo sidebar.")
            return
        st.caption(f"{batch} on {cell} (values as published after the edge deadband)")
        grid = st.columns(2)
        for i, (title, suffixes) in enumerate(TREND_TAGS[PLANT.unit(cell).cls].items()):
            df = trend(batch, cell, suffixes)
            with grid[i % 2]:
                st.markdown(f"**{title}**")
                if df.empty:
                    st.caption("no data yet")
                else:
                    st.line_chart(df, height=220)

    body()


def page_alerts() -> None:
    st.title("Alerts")
    lv = live()

    @st.fragment(run_every=REFRESH)
    def body() -> None:
        alerts = [
            (t, p.v) for t, p in lv.snapshot(uns.ENTERPRISE).items() if "/ai/anomaly/alert/" in t
        ]
        if alerts:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "site": uns.parse(t).unit.site,
                            "unit": uns.parse(t).unit.cell,
                            "key": a.key,
                            "layer": a.layer,
                            "suggested fault": a.fault_class.value if a.fault_class else "",
                            "score": round(a.score, 2),
                            "threshold": round(a.threshold, 2),
                            "top tags": ", ".join(a.top_tags),
                            "opened (sim)": a.opened_at,
                        }
                        for t, a in alerts
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.success("No open alerts at any site.")
        scored = [c for c, u in PLANT.units.items() if u.cls in ANOMALY_PROFILES]
        cell = st.selectbox("Anomaly index for", scored, key="alerts_cell")
        batch = last_batch(cell)
        if batch:
            df = trend(batch, cell, ["ai/anomaly/score"])
            if not df.empty:
                st.caption("Anomaly index: 1.0 is the alert threshold (open after 2 windows over)")
                st.line_chart(df.clip(upper=10), height=240)

    body()
    with st.expander("Model evaluation against ground-truth labels"):
        for cls in ANOMALY_PROFILES:
            report = model_file(f"{store.anomaly_name(cls)}_eval.json")
            if not report:
                st.caption(f"{cls}: no report (run `python -m ai.anomaly.evaluate`).")
                continue
            st.markdown(f"**{cls.replace('_', ' ')}** · model {report['model']}")
            if report["per_fault"]:
                st.dataframe(pd.DataFrame(report["per_fault"]).T, width="stretch")
            st.caption(
                "False alerts per clean batch (in sample): "
                f"{report['false_alerts_per_clean_batch_in_sample']:.2f}. "
                "Faulty batches never enter training."
            )


def page_yield() -> None:
    st.title("Yield")
    lv = live()
    process = st.radio(
        "Process", list(YIELD_PROFILES), format_func=PROCESS_NAMES.get, horizontal=True
    )
    profile = YIELD_PROFILES[process]
    cell = HOME[process]
    in_days = process == "bioreactor"

    @st.fragment(run_every=REFRESH)
    def body() -> None:
        batch = last_batch(cell)
        if not batch:
            st.info("No batch on this line yet.")
            return
        rows = sql(
            "SELECT payload FROM uns_events WHERE batch_id = %s AND topic = %s ORDER BY ts",
            (batch, uns.ai_prediction(unit(cell))),
        )
        if rows:
            df = pd.DataFrame(
                [
                    {
                        "day" if in_days else "hour": p["v"]["batch_day"] * (1 if in_days else 24),
                        "P10": p["v"]["value"]["p10"],
                        "P50": p["v"]["value"]["p50"],
                        "P90": p["v"]["value"]["p90"],
                    }
                    for (p,) in rows
                ]
            ).set_index("day" if in_days else "hour")
            st.markdown(
                f"**Predicted final {profile.target}, {batch}** ({profile.unit}; the band "
                "narrows as the batch runs)"
            )
            st.line_chart(df, height=260)
        else:
            st.caption(
                "Predictions start on day 3, every 6 simulated hours."
                if in_days
                else "Predictions start when the batch reaches its first predicted operation."
            )
        rec = lv.get(uns.ai_recommendation(unit(cell)))
        st.subheader("Current recommendation")
        if rec is None:
            st.caption(
                "None: the gain the DoE response surface can vouch for is below the gate "
                f"(P10 > 0, median ≥ {profile.min_gain:g} {profile.unit})."
            )
        else:
            r = rec.v
            st.markdown(
                f"`{r.id}` · expected gain **{r.gain.p50:+.2f} {profile.unit}** "
                f"(P10 {r.gain.p10:+.2f}, P90 {r.gain.p90:+.2f}) · advisory only: apply it, "
                "if at all, from the DCS console"
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "lever": k,
                            "current": round(a.current, 3),
                            "recommended": round(a.recommended, 3),
                            "change": round(a.recommended - a.current, 3),
                            "frozen": a.frozen,
                        }
                        for k, a in r.levers.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    body()
    manifest = model_file(f"{store.yield_name(process)}.json")
    if manifest:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**What drives {profile.target}** (mean |SHAP|, from the P50 model)")
            imp = pd.Series(manifest["importance"]).sort_values(ascending=False).head(12)
            st.bar_chart(imp, horizontal=True, height=320)
        with c2:
            by = manifest["metrics"].get("progress", "day")
            st.markdown(f"**Hold-out error by batch {by}** (RMSE, {profile.unit})")
            by_day = manifest["metrics"]["by_day"]
            st.line_chart(
                pd.DataFrame(
                    {float(d): {k: v["rmse"] for k, v in r.items()} for d, r in by_day.items()}
                ).T,
                height=320,
            )
    else:
        st.caption(f"No {process} yield model yet (run `python -m ai.train_all`).")
    ev = model_file(f"{store.yield_name(process)}_eval.json")
    with st.expander("Optimizer against ground truth"):
        if ev:
            ratio = ev.get("median_predicted_over_true")
            st.caption(
                f"Stopped at {ev.get('stop', ev['day'])}: the gate opened "
                f"{ev['gate_opened']}/{len(ev['cases'])} times; "
                f"{ev['gate_opened_and_truly_worth_it']} were truly worth ≥ "
                f"{profile.min_gain:g} {profile.unit}."
                + (f" Predicted gains run {ratio:.1f}× the true gains." if ratio else "")
            )
            st.dataframe(
                pd.DataFrame(ev["cases"])[
                    ["recipe", "batch", "predicted_gain_p50", "true_gain", "gate_open"]
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("Run `python -m ai.yield_.evaluate` to produce the report.")


QUERIES = {
    "Tablet batches, the Tuas API lots they used, and how those lots were made": (
        "MATCH (t:Batch {process: 'osd'})-[c:CONSUMED]->(l:MaterialLot)"
        "<-[:PRODUCED]-(a:Batch) "
        "OPTIONAL MATCH (t)-[:RESULTED_IN]->(to:Outcome) "
        "OPTIONAL MATCH (a)-[:RESULTED_IN]->(ao:Outcome) "
        "RETURN t.id AS tablet_batch, round(to.dissolution * 10) / 10 AS dissolution, "
        "l.id AS api_lot, c.quantity_kg AS kg, round(ao.d50) AS api_d50_um, "
        "round(ao.free_sa * 1000) / 1000 AS api_free_sa, a.recipe AS api_recipe "
        "ORDER BY tablet_batch DESC LIMIT 50",
        {},
    ),
    "Slow-dissolving tablets traced back to their API": (
        "MATCH (t:Batch {process: 'osd'})-[:RESULTED_IN]->(o:Outcome) "
        "WHERE o.dissolution < $q "
        "MATCH (t)-[:CONSUMED]->(:MaterialLot)<-[:PRODUCED]-(a:Batch)-[:RESULTED_IN]->(ao) "
        "RETURN t.id AS tablet_batch, round(o.dissolution * 10) / 10 AS dissolution, "
        "collect(a.id) AS api_batches, round(avg(ao.d50)) AS api_d50_um ORDER BY dissolution",
        {"q": 86.0},
    ),
    "Alerts on low-titer batches, by tag": (
        "MATCH (b:Batch)-[:RESULTED_IN]->(o:Outcome) WHERE o.titer < $low "
        "MATCH (b)-[:HAS_EVENT]->(e:Event {type: 'alert'})-[:ON_TAG]->(t:Tag) "
        "RETURN t.name AS tag, count(e) AS alerts ORDER BY alerts DESC",
        {"low": 4.0},
    ),
    "Which phase controls, and which only watches, each RX-201 tag": (
        "MATCH (pc:PhaseClass)-[b:BOUND_TO {unit: 'RX-201'}]->(m) "
        "MATCH (m)-[:HAS_CM*0..1]->(:ControlModule)-[:HAS_TAG]->(t:Tag) "
        "RETURN t.name AS tag, pc.name AS phase, b.role AS role, b.alias AS alias "
        "ORDER BY tag, role",
        {},
    ),
    "Advice → operator action → outcome, every site": (
        "MATCH (b:Batch)-[:HAS_EVENT]->(e:Event {type: 'operator'})-[:ACTED_ON]->(r) "
        "OPTIONAL MATCH (b)-[:RESULTED_IN]->(o) "
        "RETURN b.id AS batch, b.process AS process, r.id AS recommendation, "
        "e.parameter AS lever, e.old_value AS old, e.new_value AS new, "
        "coalesce(o.titer, o.yield_pct) AS outcome",
        {},
    ),
    "Outcome by process, recipe and campaign": (
        "MATCH (b:Batch)-[:RESULTED_IN]->(o:Outcome) WHERE b.status <> 'ABORTED' "
        "RETURN b.process AS process, b.recipe AS recipe, b.campaign AS campaign, "
        "count(b) AS batches, round(avg(coalesce(o.titer, o.yield_pct)) * 100) / 100 AS mean, "
        "sum(CASE WHEN o.disposition = 'REJECTED' THEN 1 ELSE 0 END) AS rejected "
        "ORDER BY process, recipe, campaign",
        {},
    ),
    "Fault ground truth versus outcome (evaluation only)": (
        "MATCH (b:Batch)-[:HAS_INJECTION]->(f:FaultInjection) MATCH (b)-[:RESULTED_IN]->(o) "
        "RETURN b.process AS process, f.fault AS fault, count(*) AS batches, "
        "round(avg(coalesce(o.titer, o.yield_pct)) * 100) / 100 AS outcome, "
        "collect(DISTINCT o.disposition) AS dispositions ORDER BY process, fault",
        {},
    ),
}


def page_graph() -> None:
    st.title("Graph")
    name = st.selectbox("Query", list(QUERIES))
    cypher, params = QUERIES[name]
    st.code(cypher, language="cypher")
    records, _, _ = neo4j().execute_query(cypher, params)
    st.dataframe(pd.DataFrame([r.data() for r in records]), hide_index=True, width="stretch")

    st.subheader("Batch genealogy")
    st.caption("Where a batch ran, what it followed, and the lots it made and used (ADR-0020).")
    batch = st.text_input("Batch", value=last_batch(HOME["osd"]) or last_batch("BR-101") or "")
    if batch:
        records, _, _ = neo4j().execute_query(
            "MATCH (b:Batch {id: $b})-[r]->(x) "
            "RETURN type(r) AS relationship, labels(x)[0] AS node, "
            "coalesce(x.id, x.name, x.batch_id) AS id, r.quantity_kg AS kg "
            "UNION "
            "MATCH (b:Batch {id: $b})-[:PRODUCED]->(l:MaterialLot)<-[c:CONSUMED]-(t:Batch) "
            "RETURN 'LOT USED BY' AS relationship, 'Batch' AS node, t.id AS id, "
            "c.quantity_kg AS kg "
            "UNION "
            "MATCH (b:Batch {id: $b})-[:CONSUMED]->(l:MaterialLot)<-[:PRODUCED]-(a:Batch) "
            "RETURN 'LOT MADE BY' AS relationship, 'Batch' AS node, a.id AS id, null AS kg",
            {"b": batch},
        )
        st.dataframe(pd.DataFrame([r.data() for r in records]), hide_index=True, width="stretch")


def page_uns() -> None:
    st.title("UNS browser")
    lv = live()
    site = site_picker("uns_site")
    line = LINE_OF_SITE[site]

    @st.fragment(run_every=REFRESH)
    def body() -> None:
        st.caption(
            "One namespace, every tag under its equipment path. Messages received: "
            + ", ".join(f"{k} {v:,}" for k, v in sorted(lv.counts.items()))
        )
        snap = lv.snapshot()
        cell = st.radio("Equipment", line.units, horizontal=True, key=f"uns_cell_{site}")
        prefix = unit(cell).prefix + "/"
        by_class: dict[str, list[dict]] = {}
        for topic, p in sorted(snap.items()):
            if not topic.startswith(prefix):
                continue
            rest = topic.removeprefix(prefix)
            value = (
                p.v
                if isinstance(p.v, float | int | str | type(None))
                else p.v.model_dump(mode="json")
            )
            by_class.setdefault(rest.split("/")[0], []).append(
                {
                    "topic": rest,
                    "v": str(value)[:120],
                    "unit": p.unit,
                    "q": p.q.value,
                    "ts": p.ts,
                    "batch": p.batch,
                }
            )
        st.markdown(f"`{prefix}`")
        for cls in [c.value for c in TopicClass]:
            rows = by_class.get(cls, [])
            with st.expander(f"{cls}/ ({len(rows)})", expanded=cls in ("pv", "state")):
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown("**Before the edge adapter** (`edge/raw`, raw DCS tags) and its dead letter")
        raw = {t: p for t, p in snap.items() if t.startswith(f"edge/raw/{unit(cell).device}/")}
        st.dataframe(
            pd.DataFrame(
                [
                    {"raw tag": p.tag, "value": p.value, "t": p.t, "q": p.q.value}
                    for p in raw.values()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        unmapped = snap.get(uns.edge_unmapped())
        if unmapped:
            st.caption(f"edge/unmapped: {unmapped.tag} = {unmapped.value:.3f} (no UNS mapping yet)")
        status = {t: p for t, p in snap.items() if t.endswith("/status") and "/_meta/" in t}
        st.markdown("**Services** (`_meta/<service>/status`, judged on wall-clock time)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "service": p.v.service,
                        "state": p.v.state,
                        "wall": p.v.wall,
                        "processed (sim)": p.ts,
                    }
                    for p in status.values()
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    body()


# --- Ask (ADR-0017) ------------------------------------------------------------------------------

EXAMPLE_QUESTIONS = (
    "Which Tuas API lots went into the latest Freiburg tablet batch, and how did they run?",
    "Why is the predicted titer of the running batch on BR-101 falling?",
    "Which phases control the reactor temperature on RX-201, and which only monitor it?",
    "How do tablet yields compare between recipe tab-v1 and tab-v2 manufacturing batches?",
)


@st.cache_resource
def assistant() -> agent.Assistant:
    return agent.from_settings(settings())  # raises (and is not cached) when unavailable


def page_ask() -> None:
    st.title("Ask")
    st.caption(
        "Claude answers from what it reads through the plant's i3X API: the same view any "
        "i3X client gets, with no ground truth and no direct store access (ADR-0017). "
        "Advisory only: it cannot change the process or decide a disposition."
    )
    try:
        bot = assistant()
    except agent.AssistantUnavailable as exc:
        st.info(str(exc))
        return
    ss = st.session_state
    ss.setdefault("ask_history", [])
    ss.setdefault("ask_log", [])
    if st.button("New conversation", disabled=not ss.ask_log):
        ss.ask_history, ss.ask_log = [], []

    for role, text, calls in ss.ask_log:
        with st.chat_message(role):
            _render_turn(text, calls)

    question = st.chat_input("Ask about a batch, an alert, a trend or the plant model")
    if not ss.ask_log:
        for q in EXAMPLE_QUESTIONS:
            if st.button(q):
                question = q
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Reading the plant through i3X…"):
                answer = bot.ask(question, ss.ask_history)
        except anthropic.APIError as exc:
            st.error(f"Claude API error: {exc}")
            return
        _render_turn(answer.text, answer.tool_calls)
    ss.ask_history = answer.messages
    ss.ask_log += [("user", question, []), ("assistant", answer.text, answer.tool_calls)]


def _render_turn(text: str, calls: list[agent.ToolCall]) -> None:
    if calls:
        with st.expander(f"{len(calls)} i3X call(s)"):
            st.code(agent.trail(calls), language=None)
    st.markdown(text)
