"""MQTT connection handling shared by every service.

- Connects with the service's own credentials (one ACL user per service).
- Registers a retained last-will of `offline` on `_meta/<service>/status` and publishes
  a wall-clock heartbeat every `heartbeat_wall_s` (ADR-0009).
- Publishes models with QoS/retain from `uns.delivery()`; nothing else chooses them.
- Decodes incoming messages with `models.decode()`. Bad payloads are logged and counted,
  never raised into the network loop.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import paho.mqtt.client as paho
from paho.mqtt.enums import CallbackAPIVersion
from pydantic import BaseModel, ValidationError

from common import models, uns
from common.settings import Settings, get_settings

log = logging.getLogger(__name__)

Handler = Callable[[str, BaseModel | None], None]


@dataclass
class Stats:
    received: int = 0
    published: int = 0
    invalid: int = 0
    handler_errors: int = 0


@dataclass
class _Subscription:
    pattern: str
    handler: Handler
    qos: int


@dataclass
class UnsClient:
    """A connected MQTT client for one service. Use `connect()` to create."""

    service: str
    src: models.Src
    settings: Settings = field(default_factory=get_settings)
    stats: Stats = field(default_factory=Stats)
    presence: bool = True  # heartbeat + last-will; False for short-lived tools (simulator.ctl)

    def __post_init__(self) -> None:
        # Unique client id: two connections as the same user must not evict each other.
        client_id = f"{self.service}-{uuid.uuid4().hex[:8]}"
        self._client = paho.Client(CallbackAPIVersion.VERSION2, client_id=client_id)
        username, password = self.settings.mqtt_credentials(self.service)
        self._client.username_pw_set(username, password)
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect
        self._subs: list[_Subscription] = []
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._last_ts: datetime | None = None
        self._status_topic = uns.meta_status(self.service)
        if self.presence:
            will = self._heartbeat("offline")
            self._client.will_set(self._status_topic, models.encode(will), qos=1, retain=True)

    # -- lifecycle ------------------------------------------------------------------------

    def start(self, wait_s: float = 30.0) -> UnsClient:
        self._client.connect_async(self.settings.mqtt_host, self.settings.mqtt_port, keepalive=30)
        self._client.loop_start()
        if not self._connected.wait(wait_s):
            raise ConnectionError(
                f"{self.service}: no MQTT connection to "
                f"{self.settings.mqtt_host}:{self.settings.mqtt_port} after {wait_s}s"
            )
        if self.presence:
            threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True).start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._connected.is_set() and self.presence:
            info = self._publish_raw(self._status_topic, models.encode(self._heartbeat("offline")))
            info.wait_for_publish(timeout=5)
        self._client.disconnect()
        self._client.loop_stop()

    def __enter__(self) -> UnsClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- publishing -----------------------------------------------------------------------

    def publish(self, topic: str, payload: BaseModel) -> paho.MQTTMessageInfo:
        """Publish a model with the topic's delivery policy.

        The model must be the one `models.model_for(topic)` names; a mismatch is a bug in
        the caller, so it raises.
        """
        expected = models.model_for(topic)
        if not isinstance(payload, expected):
            raise TypeError(f"{topic} carries {expected.__name__}, got {type(payload).__name__}")
        ts = getattr(payload, "ts", None)
        if isinstance(ts, datetime):
            self.mark_processed(ts)
        return self._publish_raw(topic, models.encode(payload))

    def clear_retained(self, topic: str) -> paho.MQTTMessageInfo:
        """Remove a retained message (alerts, recommendations: ADR-0013/0014)."""
        d = uns.delivery(topic)
        if not d.retain:
            raise ValueError(f"{topic} is not retained; nothing to clear")
        return self._publish_raw(topic, b"")

    def _publish_raw(self, topic: str, data: bytes) -> paho.MQTTMessageInfo:
        d = uns.delivery(topic)
        info = self._client.publish(topic, data, qos=d.qos, retain=d.retain)
        self.stats.published += 1
        return info

    # -- subscribing ----------------------------------------------------------------------

    def subscribe(self, pattern: str, handler: Handler, qos: int = 1) -> None:
        """Subscribe; `handler(topic, model_or_None)` runs on the network thread.

        `None` means an empty (retained-clear) message. Handlers must be quick: hand work
        to a queue if it is slow.
        """
        with self._lock:
            self._subs.append(_Subscription(pattern, handler, qos))
        if self._connected.is_set():
            self._client.subscribe(pattern, qos=qos)

    def mark_processed(self, ts: datetime) -> None:
        """Record the latest simulated time this service has handled (heartbeat `ts`)."""
        if self._last_ts is None or ts > self._last_ts:
            self._last_ts = ts

    # -- callbacks ------------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            log.error("%s: MQTT connect refused: %s", self.service, reason_code)
            return
        with self._lock:
            subs = list(self._subs)
        for s in subs:
            client.subscribe(s.pattern, qos=s.qos)
        if self.presence:
            self._publish_raw(self._status_topic, models.encode(self._heartbeat("online")))
        self._connected.set()
        log.info("%s: connected to MQTT", self.service)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        self._connected.clear()
        if not self._stop.is_set():
            log.warning("%s: MQTT disconnected (%s); reconnecting", self.service, reason_code)

    def _on_message(self, client, userdata, msg: paho.MQTTMessage) -> None:
        self.stats.received += 1
        try:
            payload = models.decode(msg.topic, msg.payload)
        except (ValidationError, uns.TopicError, models.UnknownTopicError, ValueError) as exc:
            self.stats.invalid += 1
            log.warning("%s: invalid message on %s: %s", self.service, msg.topic, exc)
            return
        with self._lock:
            subs = list(self._subs)
        for s in subs:
            if paho.topic_matches_sub(s.pattern, msg.topic):
                try:
                    s.handler(msg.topic, payload)
                except Exception:
                    self.stats.handler_errors += 1
                    log.exception("%s: handler failed for %s", self.service, msg.topic)

    # -- heartbeat ------------------------------------------------------------------------

    def _heartbeat(self, state: str) -> models.HeartbeatPayload:
        wall = models.now_utc()
        return models.HeartbeatPayload(
            v=models.Heartbeat(service=self.service, state=state, wall=wall),
            ts=self._last_ts or wall,
            unit=None,
            batch=None,
            src=self.src,
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.settings.heartbeat_wall_s):
            if self._connected.is_set():
                self._publish_raw(self._status_topic, models.encode(self._heartbeat("online")))


def connect(
    service: str, src: models.Src, settings: Settings | None = None, presence: bool = True
) -> UnsClient:
    """Create and start a client for `service`, blocking until connected."""
    return UnsClient(
        service=service, src=src, settings=settings or get_settings(), presence=presence
    ).start()
