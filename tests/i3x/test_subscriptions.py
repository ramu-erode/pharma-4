"""Sync subscriptions: scoping, sequence acknowledgement, overflow and TTL (ADR-0016)."""

from i3x.subscriptions import Hub


def upd(eid: str, v: float) -> dict:
    return {"elementId": eid, "value": v, "quality": "Good", "timestamp": "2026-03-01T00:00:00Z"}


def make(clock=None, **kw):
    t = [0.0]
    hub = Hub(clock=clock or (lambda: t[0]), **kw)
    return hub, t


def test_register_queues_the_current_value_then_changes():
    hub, _ = make()
    sub = hub.create("c1", None)
    hub.register(sub, "a", 1, ["a"], [upd("a", 1)])
    hub.publish([upd("a", 2), upd("b", 9)])
    batches, dropped = hub.sync(sub, None)
    assert batches == [{"sequenceNumber": 1, "updates": [upd("a", 1), upd("a", 2)]}]
    assert dropped == 0


def test_registering_twice_is_ignored():
    hub, _ = make()
    sub = hub.create("c1", None)
    hub.register(sub, "a", 1, ["a"], [upd("a", 1)])
    hub.register(sub, "a", 1, ["a"], [upd("a", 1)])
    assert len(hub.sync(sub, None)[0][0]["updates"]) == 1


def test_unacknowledged_batches_are_returned_again_until_acknowledged():
    hub, _ = make()
    sub = hub.create("c1", None)
    hub.register(sub, "a", 1, ["a"], [upd("a", 1)])
    assert [b["sequenceNumber"] for b in hub.sync(sub, None)[0]] == [1]
    hub.publish([upd("a", 2)])
    assert [b["sequenceNumber"] for b in hub.sync(sub, None)[0]] == [1, 2]
    assert hub.sync(sub, 2)[0] == []


def test_minus_one_clears_everything():
    hub, _ = make()
    sub = hub.create("c1", None)
    hub.register(sub, "a", 1, ["a"], [upd("a", 1)])
    hub.sync(sub, None)
    hub.publish([upd("a", 2)])
    assert hub.sync(sub, -1)[0] == []
    hub.publish([upd("a", 3)])
    assert hub.sync(sub, None)[0][0]["sequenceNumber"] == 2  # numbering keeps increasing


def test_overflow_drops_oldest_and_reports_it():
    hub, _ = make(max_queue=3)
    sub = hub.create("c1", None)
    hub.register(sub, "a", 1, ["a"], [])
    hub.publish([upd("a", i) for i in range(5)])
    batches, dropped = hub.sync(sub, None)
    assert dropped == 2 and [u["value"] for u in batches[0]["updates"]] == [2, 3, 4]
    assert hub.sync(sub, None)[1] == 0  # reported once


def test_subscriptions_are_scoped_to_their_client():
    hub, _ = make()
    sub = hub.create("c1", None)
    assert hub.get("c2", sub.subscription_id) is None
    assert not hub.delete("c2", sub.subscription_id)
    assert hub.get("c1", sub.subscription_id) is sub


def test_unregistered_elements_stop_queueing_but_keep_what_was_queued():
    hub, _ = make()
    sub = hub.create("c1", None)
    hub.register(sub, "a", 1, ["a"], [upd("a", 1)])
    hub.unregister(sub, "a")
    hub.publish([upd("a", 2)])
    assert [u["value"] for u in hub.sync(sub, None)[0][0]["updates"]] == [1]


def test_idle_subscriptions_expire():
    hub, t = make(ttl_s=60)
    sub = hub.create("c1", None)
    t[0] = 30
    hub.sync(sub, None)
    t[0] = 80
    assert hub.get("c1", sub.subscription_id) is sub
    t[0] = 91
    assert hub.get("c1", sub.subscription_id) is None
