"""The three detection layers and the alert lifecycle (architecture: anomaly detection;
ADR-0013, ADR-0014). Pure: no MQTT, no databases.

- Layer 1, rules: spec limits (recipe action limits), temperature rate of change, stuck
  sensors (bit-identical published values), bad quality.
- Layer 2, stats: EWMA of each aligned feature, per operation.
- Layer 3, multivariate: PCA T² and SPE, and Isolation Forest, on aligned features.

Layers 2-3 only score Growth and Production, stay quiet through TempShift and its
settling time, and ignore signals whose controlling phase is HELD. Layer 1 always runs.

Scores become alerts through `AlertManager`: open after 2 windows over threshold, clear
after 4 under, one lifecycle per key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ai.anomaly.model import AnomalyModel
from ai.context import SCORED_OPERATIONS
from ai.features import FEATURE_TAG, FEATURES
from common.models import FaultType

EWMA_LAMBDA = 0.2
OPEN_AFTER = 2
CLEAR_AFTER = 4
ROC_LIMIT = 1.5  # °C/h outside TempShift
STATS_FEATURES = (
    "temp_err_mean", "temp_err_std", "ph_err_mean", "do_err_mean", "agitation_mean",
    "o2_mean", "co2_mean", "base_rate", "hours_since_bolus", "pressure_mean", "weight_slope",
)  # fmt: skip
_F = {f: i for i, f in enumerate(FEATURES)}


@dataclass(frozen=True, slots=True)
class Limit:
    sp_relative: bool
    low: float
    high: float


# Rule inputs: feature for the window mean and the signal's name.
LIMIT_FEATURES = {
    "temperature": "temp_err_mean",
    "ph": "ph_err_mean",
    "do": "do_err_mean",
    "pressure": "pressure_mean",
}


# --- per-window inputs ------------------------------------------------------------------------


@dataclass
class Window:
    """Everything the detector needs about one window, precomputed or live."""

    end: int  # minute since batch origin (window is [end-30, end))
    operation: str | None
    age_h: float
    settling: bool
    held_signals: frozenset[str]
    x: np.ndarray  # raw features
    stuck: list[str]
    uncertain: list[str]
    sp: dict[str, float]  # current setpoints (for absolute limits on SP-relative features)


@dataclass
class Channels:
    """Model scores per window (vectorised over a batch)."""

    z: np.ndarray  # aligned features, NaN where not scored
    t2: np.ndarray
    spe: np.ndarray
    iforest: np.ndarray
    spe_contrib: np.ndarray  # per-feature squared residuals


def score_channels(model: AnomalyModel, windows: list[Window]) -> Channels:
    n, k = len(windows), len(FEATURES)
    z = np.full((n, k), np.nan)
    t2, spe, iff = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    contrib = np.zeros((n, k))
    for op in SCORED_OPERATIONS:
        rows = [
            i
            for i, w in enumerate(windows)
            if w.operation == op and not w.settling and op in model.ops
        ]
        if rows:
            ages = np.array([windows[i].age_h for i in rows])
            rows = [i for i, ok in zip(rows, model.scaler.covered(op, ages), strict=True) if ok]
        if not rows:
            continue
        X = np.stack([windows[i].x for i in rows])
        ages = np.array([windows[i].age_h for i in rows])
        Z = model.scaler.transform(op, ages, X)
        for j, i in enumerate(rows):  # neutralise features of held loops
            for f, tag in FEATURE_TAG.items():
                if tag in windows[i].held_signals:
                    Z[j, _F[f]] = 0.0
        m = model.ops[op]
        z[rows] = Z
        t2[rows] = m.t2(Z)
        res = m.residual(Z)
        contrib[rows] = res**2
        spe[rows] = contrib[rows].sum(axis=1)
        iff[rows] = m.iforest_score(Z)
    return Channels(z=z, t2=t2, spe=spe, iforest=iff, spe_contrib=contrib)


# --- alert lifecycle --------------------------------------------------------------------------


@dataclass
class Evidence:
    """What drove a key's score in one window."""

    layer: str
    score: float
    threshold: float
    top_tags: list[str]
    fault_class: FaultType | None
    ratio: float = 1.0  # how far past its threshold (>= 1 when fired); the anomaly index


@dataclass
class AlertChange:
    key: str
    state: str  # OPEN or CLEARED
    alert_id: str
    evidence: Evidence
    end: int
    opened_end: int


@dataclass
class _KeyState:
    over: int = 0
    under: int = 0
    open_id: str | None = None
    opened_end: int = 0
    evidence: Evidence | None = None


@dataclass
class AlertManager:
    batch_id: str
    keys: dict[str, _KeyState] = field(default_factory=dict)
    counter: int = 0

    def update(self, end: int, fired: dict[str, Evidence]) -> list[AlertChange]:
        changes: list[AlertChange] = []
        for key in set(self.keys) | set(fired):
            st = self.keys.setdefault(key, _KeyState())
            ev = fired.get(key)
            if ev is not None:
                st.over, st.under = st.over + 1, 0
                st.evidence = (
                    ev if st.evidence is None or ev.score >= st.evidence.score else st.evidence
                )
                if st.open_id is None and st.over >= OPEN_AFTER:
                    self.counter += 1
                    st.open_id, st.opened_end = f"{self.batch_id}-A{self.counter:03d}", end
                    changes.append(AlertChange(key, "OPEN", st.open_id, ev, end, end))
            else:
                st.over, st.under = 0, st.under + 1
                if st.open_id is not None and st.under >= CLEAR_AFTER:
                    changes.append(
                        AlertChange(key, "CLEARED", st.open_id, st.evidence, end, st.opened_end)
                    )
                    st.open_id, st.evidence = None, None
                elif st.open_id is None:
                    st.evidence = None
        return changes

    @property
    def open_keys(self) -> list[str]:
        return [k for k, s in self.keys.items() if s.open_id]


# --- the detector ------------------------------------------------------------------------------


@dataclass
class Detector:
    """One unit's detector for one batch. Feed windows in time order."""

    model: AnomalyModel
    batch_id: str
    limits: dict[str, Limit]
    alerts: AlertManager = field(init=False)
    ewma: dict[str, float] = field(default_factory=dict)
    last_op: str | None = None

    def __post_init__(self) -> None:
        self.alerts = AlertManager(self.batch_id)

    def step(
        self, w: Window, ch: Channels, i: int
    ) -> tuple[float, list[AlertChange], dict[str, float]]:
        """Score window `i` of `ch`. Returns (anomaly index, alert changes, raw channels)."""
        fired: dict[str, Evidence] = {}
        raw: dict[str, float] = {}
        self._rules(w, fired)
        scored = w.operation in SCORED_OPERATIONS and not w.settling and not np.isnan(ch.t2[i])
        if w.operation != self.last_op or not scored:
            self.ewma.clear()
        self.last_op = w.operation
        if scored:
            self._stats(ch.z[i], w, fired, raw)
            self._multivariate(ch, i, fired, raw)
        index = max((e.ratio for e in fired.values()), default=0.0)
        if scored and not fired:
            index = max(
                (raw[c] / self.model.thresholds[c] for c in raw if c in self.model.thresholds),
                default=0.0,
            )
        return index, self.alerts.update(w.end, fired), raw

    # layer 1
    def _rules(self, w: Window, fired: dict[str, Evidence]) -> None:
        for s in w.stuck:
            fired[f"rules-stuck_{s}"] = Evidence("rules", 3.0, 3.0, [s], FaultType.STUCK_SENSOR)
        for s in w.uncertain:
            fired[f"rules-quality_{s}"] = Evidence("rules", 1.0, 1.0, [s], None)
        if w.operation is None or w.operation in ("Setup", "Harvest"):
            return
        for sig, feat in LIMIT_FEATURES.items():
            lim = self.limits.get(sig)
            if lim is None:
                continue
            err = float(w.x[_F[feat]])
            value = err if lim.sp_relative else err + w.sp.get(sig, 0.0)
            if not lim.low <= value <= lim.high:
                bound = lim.low if value < lim.low else lim.high
                span = max(lim.high - lim.low, 1e-6)
                fired[f"rules-limit_{sig}"] = Evidence(
                    "rules", value, bound, [sig], None, 1.0 + abs(value - bound) / span
                )
        slope = float(w.x[_F["temp_slope"]])
        if w.operation != "TempShift" and not w.settling and abs(slope) > ROC_LIMIT:
            fired["rules-roc_temperature"] = Evidence(
                "rules",
                abs(slope),
                ROC_LIMIT,
                ["temperature"],
                FaultType.TEMP_CONTROL_LOSS,
                abs(slope) / ROC_LIMIT,
            )

    # layer 2
    def _stats(
        self, z: np.ndarray, w: Window, fired: dict[str, Evidence], raw: dict[str, float]
    ) -> None:
        best: dict[str, tuple[float, float, str]] = {}
        for f in STATS_FEATURES:
            tag = FEATURE_TAG[f]
            if tag in w.held_signals:
                self.ewma.pop(f, None)
                continue
            e = EWMA_LAMBDA * z[_F[f]] + (1 - EWMA_LAMBDA) * self.ewma.get(f, 0.0)
            self.ewma[f] = e
            channel = f"stats:{f}"
            raw[channel] = abs(e)
            thr = self.model.thresholds.get(channel)
            if (
                thr is not None
                and abs(e) > thr
                and (tag not in best or abs(e) / thr > best[tag][0] / best[tag][1])
            ):
                best[tag] = (abs(e), thr, f)
        for tag, (score, thr, f) in best.items():
            fired[f"stats-{tag}"] = Evidence(
                "stats", score, thr, [tag], suggest(self._signs(z), f), score / thr
            )

    # layer 3
    def _multivariate(
        self, ch: Channels, i: int, fired: dict[str, Evidence], raw: dict[str, float]
    ) -> None:
        contrib = ch.spe_contrib[i]
        order = np.argsort(-contrib)
        top: list[str] = []
        for j in order:
            tag = FEATURE_TAG[FEATURES[j]]
            if tag not in top:
                top.append(tag)
            if len(top) == 3:
                break
        signs = self._signs(ch.z[i])
        lead = FEATURES[int(order[0])]
        for channel, key, layer in (
            ("mspc:t2", "mspc-t2", "mspc"),
            ("mspc:spe", "mspc-spe", "mspc"),
            ("iforest:score", "iforest-score", "iforest"),
        ):
            value = {"mspc:t2": ch.t2, "mspc:spe": ch.spe, "iforest:score": ch.iforest}[channel][i]
            raw[channel] = float(value)
            thr = self.model.thresholds.get(channel)
            if thr is not None and value > thr:
                fired[key] = Evidence(
                    layer, float(value), thr, top, suggest(signs, lead), float(value) / thr
                )

    @staticmethod
    def _signs(z: np.ndarray) -> dict[str, float]:
        return {f: float(z[_F[f]]) for f in FEATURES}


def suggest(z: dict[str, float], lead: str) -> FaultType | None:
    """A fault class from the pattern of aligned features (explainable, not learned)."""
    if z["hours_since_bolus"] > 3 or z["feed_rate"] < -3:
        return FaultType.FEED_PUMP_FAILURE
    if z["do_err_mean"] < -3 and z["ph_err_mean"] < -2:
        return FaultType.CONTAMINATION
    if z["temp_err_std"] > 3 or abs(z["temp_slope"]) > 3:
        return FaultType.TEMP_CONTROL_LOSS
    if (z["agitation_mean"] > 2 or z["o2_mean"] > 2) and z["do_err_mean"] < 0:
        return FaultType.DO_SPARGER_FOULING
    if z["co2_mean"] > 2 or z["base_rate"] < -2:
        return FaultType.PH_PROBE_DRIFT
    return None
