"""i3X subscriptions: sync queues fed by the broker (ADR-0016). Streaming is not offered.

Semantics from the i3X 1.0 guide, "Subscribe Methods":

- a subscription belongs to one `clientId`; any other client gets "not found"
- registering an object queues its current value at once (like an MQTT retained message),
  then every change; registering twice is a no-op
- `sync` wraps everything queued since the last call in a new batch with the next
  sequence number, and returns every batch not yet acknowledged
- `lastSequenceNumber = n` acknowledges batches up to n; `-1` clears everything
- when the queue is full the oldest updates are dropped, and the next sync says so (206)
- a subscription not synced within the TTL is deleted
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

Update = dict[str, Any]  # {elementId, value, quality, timestamp}


@dataclass
class Subscription:
    subscription_id: str
    client_id: str
    display_name: str
    last_seen: float
    monitored: dict[str, tuple[int, list[str]]] = field(default_factory=dict)  # id -> depth, ids
    pending: deque[Update] = field(default_factory=deque)
    batches: list[tuple[int, list[Update]]] = field(default_factory=list)
    next_seq: int = 1
    dropped: int = 0

    @property
    def watched(self) -> set[str]:
        return {e for _, ids in self.monitored.values() for e in ids}

    def describe(self) -> dict[str, Any]:
        return {
            "subscriptionId": self.subscription_id,
            "displayName": self.display_name,
            "monitoredObjects": [
                {"elementId": e, "maxDepth": depth} for e, (depth, _) in self.monitored.items()
            ],
        }


class Hub:
    def __init__(
        self,
        max_queue: int = 10_000,
        ttl_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_queue = max_queue
        self.ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._subs: dict[str, Subscription] = {}

    def create(self, client_id: str, display_name: str | None) -> Subscription:
        sid = secrets.token_urlsafe(24)
        with self._lock:
            self._reap()
            sub = Subscription(sid, client_id, display_name or sid, self._clock())
            self._subs[sid] = sub
            return sub

    def get(self, client_id: str, subscription_id: str) -> Subscription | None:
        """The subscription, or None if it does not exist *for this client*."""
        with self._lock:
            self._reap()
            sub = self._subs.get(subscription_id)
            return sub if sub is not None and sub.client_id == client_id else None

    def delete(self, client_id: str, subscription_id: str) -> bool:
        with self._lock:
            sub = self._subs.get(subscription_id)
            if sub is None or sub.client_id != client_id:
                return False
            del self._subs[subscription_id]
            return True

    def register(
        self, sub: Subscription, element_id: str, depth: int, ids: list[str], initial: list[Update]
    ) -> None:
        with self._lock:
            if element_id in sub.monitored:
                return  # the spec: a second registration succeeds and is ignored
            sub.monitored[element_id] = (depth, ids)
            self._enqueue(sub, initial)

    def unregister(self, sub: Subscription, element_id: str) -> None:
        with self._lock:
            sub.monitored.pop(element_id, None)  # queued values stay (the spec: SHOULD NOT delete)

    def publish(self, updates: Iterable[Update]) -> None:
        """Queue updates on every subscription watching their elements."""
        updates = list(updates)
        with self._lock:
            for sub in self._subs.values():
                watched = sub.watched
                self._enqueue(sub, [u for u in updates if u["elementId"] in watched])

    def sync(
        self, sub: Subscription, last_sequence_number: int | None
    ) -> tuple[list[dict[str, Any]], int]:
        """Acknowledge, batch what is pending, return unacknowledged batches and drop count."""
        with self._lock:
            sub.last_seen = self._clock()
            if last_sequence_number == -1:
                sub.batches.clear()
                sub.pending.clear()
            elif last_sequence_number is not None:
                sub.batches = [b for b in sub.batches if b[0] > last_sequence_number]
            if sub.pending:
                sub.batches.append((sub.next_seq, list(sub.pending)))
                sub.next_seq += 1
                sub.pending.clear()
            dropped, sub.dropped = sub.dropped, 0
            return [{"sequenceNumber": s, "updates": u} for s, u in sub.batches], dropped

    def _enqueue(self, sub: Subscription, updates: list[Update]) -> None:
        sub.pending.extend(updates)
        excess = sum(len(u) for _, u in sub.batches) + len(sub.pending) - self.max_queue
        while excess > 0:  # drop oldest first: unacknowledged batches, then pending
            if sub.batches:
                seq, first = sub.batches[0]
                cut = min(excess, len(first))
                sub.batches[0] = (seq, first[cut:])
                if not sub.batches[0][1]:
                    sub.batches.pop(0)
            else:
                cut = excess
                for _ in range(cut):
                    sub.pending.popleft()
            sub.dropped += cut
            excess -= cut

    def _reap(self) -> None:
        cutoff = self._clock() - self.ttl_s
        for sid in [s for s, sub in self._subs.items() if sub.last_seen < cutoff]:
            del self._subs[sid]
