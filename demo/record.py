"""Record a pharma-4 demo video: one clip per scene, simulator waits happen off camera."""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt
from playwright.sync_api import Page, sync_playwright

OUT = Path(__file__).parent / "clips"
URL = "http://localhost:8501"
W, H = 1600, 900
PREFIX = "pharmanextgen/grange-castle/upstream/suite-1"
CELL = "BR-101"


# --- broker helpers (off camera) -------------------------------------------------------


class Broker:
    def __init__(self) -> None:
        self.latest: dict[str, dict | None] = {}
        self.lock = threading.Lock()
        self.sub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.sub.username_pw_set("explorer", "dev-explorer")
        self.sub.on_message = self._on
        self.sub.connect("localhost", 1883)
        self.sub.subscribe([("_sim/clock", 0), (f"{PREFIX}/{CELL}/#", 0)])
        self.sub.loop_start()
        self.pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.pub.username_pw_set("dashboard", "dev-dashboard")
        self.pub.connect("localhost", 1883)
        self.pub.loop_start()

    def _on(self, _c, _u, msg) -> None:
        with self.lock:
            self.latest[msg.topic] = json.loads(msg.payload) if msg.payload else None

    def get(self, topic: str) -> dict | None:
        with self.lock:
            return self.latest.get(topic)

    def alerts(self) -> dict[str, dict]:
        with self.lock:
            return {t: p for t, p in self.latest.items() if "/ai/anomaly/alert/" in t and p}

    def clock(self, action: str, **kw) -> None:
        c = self.get("_sim/clock")
        ts = c["ts"] if c else datetime.now(timezone.utc).isoformat()
        body = {"v": {"action": action, **kw}, "ts": ts, "unit": None, "q": "GOOD",
                "batch": None, "src": "operator"}
        self.pub.publish("_sim/cmd/clock", json.dumps(body), qos=1).wait_for_publish()

    def wait(self, pred, timeout: float, what: str) -> None:
        end = time.time() + timeout
        while time.time() < end:
            if pred():
                print(f"  ok: {what}")
                return
            time.sleep(0.5)
        raise TimeoutError(what)


# --- on-camera helpers ------------------------------------------------------------------

CAPTION_JS = """
([title, body]) => {
  let el = document.getElementById('demo-cap');
  if (!el) {
    el = document.createElement('div'); el.id = 'demo-cap';
    el.style.cssText = `position:fixed;left:50%;bottom:28px;transform:translateX(-50%);
      z-index:2147483647;max-width:1100px;width:calc(100% - 420px);margin-left:150px;
      background:rgba(15,25,32,.92);color:#fff;border-radius:10px;padding:14px 22px;
      font:16px/1.45 'Source Sans Pro',system-ui,sans-serif;box-shadow:0 8px 30px rgba(0,0,0,.25);
      transition:opacity .35s`;
    document.documentElement.appendChild(el);
  }
  el.style.opacity = 0;
  setTimeout(() => {
    el.innerHTML = (title ? `<div style="font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#6fd3d6;margin-bottom:4px;font-weight:600">${title}</div>` : '') + body;
    el.style.opacity = 1;
  }, 300);
}
"""


def caption(pg: Page, title: str, body: str, hold: float = 5.0) -> None:
    pg.evaluate(CAPTION_JS, [title, body])
    pg.wait_for_timeout(int(hold * 1000))


def sidebar(pg: Page):
    return pg.locator("section[data-testid='stSidebar']")


def choose(pg: Page, scope, label: str, option: str) -> None:
    box = scope.locator("div[data-testid='stSelectbox']").filter(has_text=label).first
    box.click()
    pg.wait_for_timeout(500)
    pg.get_by_role("option", name=option, exact=True).click()
    pg.wait_for_timeout(1200)


def click(pg: Page, scope, name: str) -> None:
    btn = scope.get_by_role("button", name=name, exact=True).first
    btn.hover()
    pg.wait_for_timeout(400)
    btn.click()
    park(pg)
    pg.wait_for_timeout(1500)


SCROLL_JS = """
([mode, arg, ms]) => new Promise(done => {
  const el = [...document.querySelectorAll('[data-testid=stMain], [data-testid=stAppViewContainer], section.main')]
    .find(e => e.scrollHeight > e.clientHeight + 5) || document.scrollingElement;
  let target = el.scrollTop + arg;
  if (mode === 'to') {
    const hit = [...el.querySelectorAll('p, strong, h1, h2, h3, summary, span')]
      .find(n => n.textContent.trim() === arg);
    if (!hit) { done(false); return; }
    target = el.scrollTop + hit.getBoundingClientRect().top - 90;
  }
  const start = el.scrollTop, t0 = performance.now();
  const step = now => {
    const k = Math.min(1, (now - t0) / ms), e = k < .5 ? 2*k*k : 1 - Math.pow(-2*k + 2, 2) / 2;
    el.scrollTop = start + (target - start) * e;
    k < 1 ? requestAnimationFrame(step) : done(true);
  };
  requestAnimationFrame(step);
})
"""


def park(pg: Page) -> None:
    pg.mouse.move(W - 20, 70)


def scroll(pg: Page, px: int, secs: float = 1.6) -> None:
    park(pg)
    pg.evaluate(SCROLL_JS, ["by", px, int(secs * 1000)])
    pg.wait_for_timeout(700)


def scroll_to(pg: Page, text: str, secs: float = 1.6) -> None:
    park(pg)
    if not pg.evaluate(SCROLL_JS, ["to", text, int(secs * 1000)]):
        print(f"  (heading not found: {text})")
    pg.wait_for_timeout(700)


def goto(pg: Page, path: str, settle: float = 6.0) -> None:
    pg.goto(f"{URL}/{path}")
    pg.wait_for_selector("section[data-testid='stSidebar']")
    pg.wait_for_timeout(int(settle * 1000))


CARD = """
<html><body style="margin:0;height:100vh;display:grid;place-items:center;background:#0f1a20;
 color:#e6eef0;font-family:'Segoe UI',system-ui,sans-serif">
<div style="max-width:1100px;padding:0 60px">
 <div style="font-size:15px;letter-spacing:.14em;text-transform:uppercase;color:#6fd3d6;font-weight:600">{eyebrow}</div>
 <div style="font-size:54px;font-weight:700;line-height:1.15;margin:14px 0 22px">{title}</div>
 <div style="font-size:22px;line-height:1.55;color:#b5c6cc">{body}</div>
</div></body></html>
"""


class Recorder:
    def __init__(self, pw) -> None:
        self.browser = pw.chromium.launch()
        self.n = 0

    def scene(self, name: str):
        self.n += 1
        ctx = self.browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(OUT / "raw"),
            record_video_size={"width": W, "height": H},
        )
        pg = ctx.new_page()
        rec = self

        class _S:
            def __enter__(self_inner):
                print(f"scene {rec.n}: {name}")
                return pg

            def __exit__(self_inner, *exc):
                path = pg.video.path()
                ctx.close()
                Path(path).rename(OUT / f"{rec.n:02d}-{name}.webm")
                return False

        return _S()

    def still(self, name: str):
        return _Still(self, name)


class _Still:
    def __init__(self, rec, name):
        self.rec, self.name = rec, name

    def __enter__(self):
        self.rec.n += 1
        self.pg = self.rec.browser.new_page(viewport={"width": W, "height": H})
        self.pg.stills, self.pg.still_name = [], f"{self.rec.n:02d}-{self.name}"
        print(f"still {self.rec.n}: {self.name}")
        return self.pg

    def __exit__(self, *exc):
        self.pg.close()
        return False


def card(pg: Page, eyebrow: str, title: str, body: str, hold: float) -> None:
    pg.set_content(CARD.format(eyebrow=eyebrow, title=title, body=body))
    pg.wait_for_timeout(300)
    pg.stills.append(hold)
    pg.screenshot(path=str(OUT / f"{pg.still_name}-{len(pg.stills)}-{hold:g}s.png"))


# --- the demo ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    for f in [*OUT.glob("*.webm"), *OUT.glob("*.png")]:
        f.unlink()
    br = Broker()
    br.wait(lambda: br.get("_sim/clock") is not None, 10, "clock seen")
    print("  waiting for BR-101 to be idle")
    br.wait(lambda: not br.get(f"{PREFIX}/{CELL}/state/batch")["v"], 900, "BR-101 idle")
    br.clock("speed", speed=3600.0)

    with sync_playwright() as pw:
        r = Recorder(pw)

        with r.still("intro") as pg:
            card(pg, "Pharma 4.0 proof of concept", "pharma-4",
                 "A simulated fed-batch bioreactor producing a monoclonal antibody.<br>"
                 "Its data flows through an MQTT Unified Namespace into a time-series historian "
                 "and a Neo4j knowledge graph. Two AI services then watch every batch: "
                 "one for developing faults, one for final yield.", 9)
            card(pg, "What this demo shows", "From sensor to advice",
                 "1&nbsp;&nbsp;Start a batch and fast-forward to day 4<br>"
                 "2&nbsp;&nbsp;Inject a pH-probe drift, a fault the sensor itself hides<br>"
                 "3&nbsp;&nbsp;Watch the anomaly layer catch it<br>"
                 "4&nbsp;&nbsp;See the yield prediction and setpoint advice<br>"
                 "5&nbsp;&nbsp;Explore the knowledge graph and the UNS topic tree", 9)

        with r.scene("start-batch") as pg:
            goto(pg, "")
            caption(pg, "Live page",
                    "Two simulated 2000 L bioreactors, BR-101 and BR-102. Both are idle between batches.", 6)
            sb = sidebar(pg)
            caption(pg, "Demo controls",
                    "The sidebar drives the simulator through <code>_sim/cmd</code>. "
                    "It stands in for an MES and the DCS operator console.", 5)
            click(pg, sb, "Start batch")
            br.wait(lambda: (br.get(f"{PREFIX}/{CELL}/state/batch") or {}).get("v"), 30, "batch started")
            batch = br.get(f"{PREFIX}/{CELL}/state/batch")["v"]
            caption(pg, "Batch started",
                    f"Batch <b>{batch}</b> is running on BR-101 with the current recipe. "
                    "Next, fast-forward the simulated clock to day 4.", 5)
            day = sb.get_by_label("Day")
            day.fill("4")
            day.press("Enter")
            pg.wait_for_timeout(1000)
            click(pg, sb, "Run to day")
            caption(pg, "Simulated time at 3600×",
                    "One real second is one simulated hour. The whole stack runs on plant time; "
                    "services only use the wall clock to report that they are alive.", 7)

        print("  waiting for day 4 (off camera)")
        br.wait(lambda: br.get("_sim/clock")["v"]["paused"], 400, "paused at day 4")

        with r.scene("day4-inject") as pg:
            goto(pg, "", settle=8)
            caption(pg, "Day 4 · Growth",
                    "Operations run in sequence: Growth, then TempShift, then Production. "
                    "The control phases TEMP_CTRL, PH_CTRL, DO_CTRL and FEED_ADD run in parallel inside each one.", 7)
            scroll_to(pg, "Temperature (°C)")
            caption(pg, "Trends from the historian",
                    "These values have been through the edge adapter: raw DCS tags mapped onto ISA-95 topics, "
                    "with unit, batch and quality added and a deadband applied.", 7)
            sb = sidebar(pg)
            sb.locator("div[data-testid='stSelectbox']").filter(has_text="Fault").first.scroll_into_view_if_needed()
            pg.wait_for_timeout(800)
            choose(pg, sb, "Fault", "ph_probe_drift")
            caption(pg, "Inject a fault",
                    "<b>pH probe drift.</b> The probe reads higher and higher. The pH loop holds the <i>measured</i> "
                    "value at setpoint, so the <i>true</i> pH falls while the trend looks perfect.", 7)
            click(pg, sb, "Inject")
            sb.get_by_role("button", name="Resume", exact=True).scroll_into_view_if_needed()
            click(pg, sb, "Resume")
            caption(pg, "Ground truth stays hidden",
                    "The simulator logs the fault on <code>_sim/faults</code>. The broker ACL stops the AI services "
                    "from reading it, so they have to find the fault from process data alone.", 7)

        print("  waiting for an alert (off camera)")
        br.wait(lambda: br.alerts(), 600, "alert open")
        time.sleep(4)
        br.clock("pause")
        time.sleep(3)

        with r.scene("evidence") as pg:
            goto(pg, "", settle=8)
            scroll_to(pg, "Temperature (°C)")
            caption(pg, "What the data shows",
                    "The measured pH still sits on its setpoint, so the pH trend alone looks perfect.", 6)
            scroll_to(pg, "Gas flows (L/min)")
            caption(pg, "The fault signature",
                    "To hold the drifting reading at setpoint, the pH loop keeps adding CO₂: gas flow climbs "
                    "while base use stays flat. A single-tag alarm on pH would never fire.", 8)

        with r.scene("alerts") as pg:
            goto(pg, "alerts", settle=8)
            caption(pg, "Alerts page",
                    "The multivariate (PCA) and statistical (EWMA/CUSUM) layers both flag the batch. Each alert "
                    "suggests a fault class, here <b>ph_probe_drift</b>, and lists the tags that contributed most.", 8)
            scroll(pg, 450)
            caption(pg, "Anomaly index",
                    "1.0 is the alert threshold. An alert opens after 2 windows above it and clears after 4 below. "
                    "Rules, univariate statistics and PCA/Isolation Forest each score every 30-minute window.", 8)
            exp = pg.get_by_text("Model evaluation against ground-truth labels")
            exp.scroll_into_view_if_needed()
            exp.click()
            pg.wait_for_timeout(1500)
            scroll(pg, 500)
            caption(pg, "Scored offline against the truth",
                    "Evaluation compares alerts with the hidden fault labels: detection rate, lead time "
                    "before the true spec breach, and false alerts per clean batch.", 8)

        with r.scene("yield") as pg:
            goto(pg, "yield", settle=8)
            caption(pg, "Yield page",
                    "From day 3, a LightGBM ensemble predicts final titer every 6 simulated hours. "
                    "It shows a calibrated P10–P90 band that narrows as the batch runs.", 8)
            scroll(pg, 550)
            rec = br.get(f"{PREFIX}/{CELL}/ai/yield/recommendation")
            caption(pg, "Setpoint advice, only when the gain is real",
                    "A response surface fitted to the process-characterisation runs values each lever: shift day, "
                    "temperature, pH, DO and feed. Advice is published only if the gain is at least 0.1 g/L "
                    "and its P10 is above zero." + ("" if rec else " <b>No change clears that bar for this batch.</b>"), 9)
            scroll(pg, 650)
            caption(pg, "What drives titer",
                    "SHAP values rank what drives the remaining gain. The AI is advisory: an operator makes any change "
                    "through the DCS console, and the graph links it to the recommendation it followed.", 8)

        with r.scene("graph") as pg:
            goto(pg, "graph", settle=6)
            caption(pg, "Knowledge graph (Neo4j)",
                    "Context lives in the graph: equipment hierarchy, recipes and spec limits, "
                    "batch genealogy and outcomes. Time series stay in TimescaleDB.", 7)
            main = pg.locator("section[data-testid='stMain']")
            opts = main.locator("div[data-testid='stSelectbox']").first
            opts.click()
            pg.wait_for_timeout(600)
            options = pg.get_by_role("option")
            if options.count() > 1:
                options.nth(1).click()
            pg.wait_for_timeout(2500)
            caption(pg, "Ask across batches",
                    "Preset Cypher queries join process context with outcomes, "
                    "for example which phase controls or only monitors a tag.", 7)
            scroll(pg, 700)
            caption(pg, "Batch genealogy",
                    "Each batch links to its recipe, operations, phases, alerts, operator actions and outcome.", 7)

        with r.scene("uns") as pg:
            goto(pg, "uns", settle=6)
            caption(pg, "Unified Namespace browser",
                    "One ISA-95 topic tree: <code>pharmanextgen/grange-castle/upstream/suite-1/BR-101/pv/ph</code>. "
                    "Every payload carries value, plant timestamp, unit, quality, batch and source.", 8)
            scroll(pg, 600)
            caption(pg, "One owner per branch",
                    "The edge adapter owns <code>pv</code> and <code>sp</code>, the simulator owns <code>lab</code>, "
                    "<code>state</code> and <code>events</code>, and the AI services own <code>ai</code>.", 7)

        with r.still("outro") as pg:
            card(pg, "Also in the stack", "Read the plant through an open standard",
                 "An i3X 1.0 read API (port 8600) serves the same plant model, live values and history. "
                 "It never serves ground truth.<br><br>"
                 "The <b>Ask</b> page connects Claude to that API so you can question the plant in plain "
                 "language. It needs <code>ANTHROPIC_API_KEY</code> in <code>.env</code>.", 10)
            card(pg, "pharma-4", "docker compose up",
                 "Simulator · edge adapter · Mosquitto · TimescaleDB · Neo4j · anomaly and yield AI · "
                 "Streamlit · i3X<br><br>A demo and learning vehicle. Not validated for GMP use; AI output is advisory only.", 7)

    br.clock("resume")
    print("done")


if __name__ == "__main__":
    main()
