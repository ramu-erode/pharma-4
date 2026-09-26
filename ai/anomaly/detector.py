"""The three detection layers and the alert lifecycle (architecture: anomaly detection;
ADR-0013, ADR-0015). Pure: no MQTT, no databases.

- Layer 1, rules: spec limits (recipe action limits), temperature rate of change, stuck
  sensors (bit-identical published values), bad quality.
- Layer 2, stats: EWMA of each aligned feature, per operation.
- Layer 3, multivariate: PCA T² and SPE, and Isolation Forest, on aligned features.

Layers 2-3 only score the profile's operations (the bioreactor: Growth and Production),
stay quiet while settling (TempShift; off the bioreactor, the first window of each
operation), and ignore signals whose controlling phase is HELD. Layer 1 always runs.
Everything class-specific comes from the model's profile (ADR-0021).

Scores become alerts through `AlertManager`: open after 2 windows over threshold, clear
after 4 under, one lifecycle per key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ai.anomaly.model import AnomalyModel
from ai.profiles import BIOREACTOR, AnomalyProfile
from common.models import FaultType

EWMA_LAMBDA = 0.2
OPEN_AFTER = 2
CLEAR_AFTER = 4
ROC_LIMIT = BIOREACTOR.roc.limit if BIOREACTOR.roc else 1.5  # °C/h outside TempShift
STATS_FEATURES = BIOREACTOR.stats_features
LIMIT_FEATURES = BIOREACTOR.limit_features


@dataclass(frozen=True, slots=True)
class Limit:
    sp_relative: bool
    low: float
    high: float
    operations: tuple[str, ...] | None = None  # None: every operation


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
    profile = model.profile
    fi = profile.feature_index
    n, k = len(windows), len(profile.features)
    z = np.full((n, k), np.nan)
    t2, spe, iff = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    contrib = np.zeros((n, k))
    for op in profile.scored_ops:
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
            for f, tag in profile.feature_tag.items():
                if tag in windows[i].held_signals:
                    Z[j, fi[f]] = 0.0
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
        self.profile: AnomalyProfile = self.model.profile

    def step(
        self, w: Window, ch: Channels, i: int
    ) -> tuple[float, list[AlertChange], dict[str, float]]:
        """Score window `i` of `ch`. Returns (anomaly index, alert changes, raw channels)."""
        fired: dict[str, Evidence] = {}
        raw: dict[str, float] = {}
        self._rules(w, fired)
        scored = (
            w.operation in self.profile.scored_ops and not w.settling and not np.isnan(ch.t2[i])
        )
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
        profile, fi = self.profile, self.profile.feature_index
        if w.operation is None or w.operation in profile.rules_skip_ops:
            return
        for sig, feat in profile.limit_features.items():
            lim = self.limits.get(sig)
            if lim is None or (lim.operations and w.operation not in lim.operations):
                continue
            if profile.settle_new_operation and w.settling:
                continue  # the window still holds the previous operation's values
            err = float(w.x[fi[feat]])
            value = err if lim.sp_relative else err + w.sp.get(sig, 0.0)
            if not lim.low <= value <= lim.high:
                bound = lim.low if value < lim.low else lim.high
                span = max(lim.high - lim.low, 1e-6)
                fired[f"rules-limit_{sig}"] = Evidence(
                    "rules", value, bound, [sig], None, 1.0 + abs(value - bound) / span
                )
        roc = profile.roc
        if roc is None:
            return
        slope = float(w.x[fi[roc.feature]])
        if w.operation not in roc.exempt and not w.settling and abs(slope) > roc.limit:
            fired[f"rules-roc_{roc.tag}"] = Evidence(
                "rules", abs(slope), roc.limit, [roc.tag], roc.fault, abs(slope) / roc.limit
            )

    # layer 2
    def _stats(
        self, z: np.ndarray, w: Window, fired: dict[str, Evidence], raw: dict[str, float]
    ) -> None:
        profile, fi = self.profile, self.profile.feature_index
        best: dict[str, tuple[float, float, str]] = {}
        for f in profile.stats_features:
            tag = profile.feature_tag[f]
            if tag in w.held_signals:
                self.ewma.pop(f, None)
                continue
            e = EWMA_LAMBDA * z[fi[f]] + (1 - EWMA_LAMBDA) * self.ewma.get(f, 0.0)
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
        for tag, (score, thr, _f) in best.items():
            fired[f"stats-{tag}"] = Evidence(
                "stats", score, thr, [tag], profile.suggest(self._signs(z)), score / thr
            )

    # layer 3
    def _multivariate(
        self, ch: Channels, i: int, fired: dict[str, Evidence], raw: dict[str, float]
    ) -> None:
        profile = self.profile
        contrib = ch.spe_contrib[i]
        order = np.argsort(-contrib)
        top: list[str] = []
        for j in order:
            tag = profile.feature_tag[profile.features[j]]
            if tag not in top:
                top.append(tag)
            if len(top) == 3:
                break
        signs = self._signs(ch.z[i])
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
                    layer, float(value), thr, top, profile.suggest(signs), float(value) / thr
                )

    def _signs(self, z: np.ndarray) -> dict[str, float]:
        return {f: float(z[i]) for f, i in self.profile.feature_index.items()}


def suggest(z: dict[str, float], lead: str | None = None) -> FaultType | None:
    """The bioreactor's fault-class heuristic (explainable, not learned); each profile
    has its own (`ai.profiles`)."""
    return BIOREACTOR.suggest(z)
