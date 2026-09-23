import json

import pytest

from common import models as m
from common import uns
from historian.core import Batch, EventRow, LabelRow, TagRow, route
from tests.samples import BATCH, TS, UNIT, samples


@pytest.mark.parametrize(("topic", "payload"), samples(), ids=lambda x: str(x)[:60])
def test_every_topic_class_has_a_home(topic, payload):
    row = route(topic, payload)
    p = uns.parse(topic)
    if p.kind is uns.TopicKind.SIM_FAULTS:
        assert isinstance(row, LabelRow)
    elif p.kind is not uns.TopicKind.UNS:
        assert row is None  # _meta, edge/*, _sim/cmd and _sim/clock are not history
    elif isinstance(payload, m.ScalarPayload):
        assert isinstance(row, TagRow) and row.value == payload.v and row.topic == topic
    else:
        assert isinstance(row, EventRow)
        assert json.loads(row.payload) == json.loads(m.encode(payload))


def test_scalar_row_keeps_quality_batch_and_time():
    p = m.ScalarPayload(v=7.01, ts=TS, unit="pH", q="UNCERTAIN", batch=BATCH, src="edge")
    row = route(uns.pv(UNIT, "ph"), p)
    assert (row.ts, row.batch_id, row.quality) == (TS, BATCH, "UNCERTAIN")


def test_retained_clear_is_not_history():
    assert route(uns.ai_alert(UNIT, "stats-co2_flow"), None) is None


def test_batch_groups_rows():
    b = Batch()
    for topic, payload in samples():
        row = route(topic, payload)
        if row is not None:
            b.add(row)
    assert len(b.tags) == 4 and len(b.labels) == 1 and len(b.events) > 5
    assert len(b) == len(b.tags) + len(b.events) + len(b.labels)
