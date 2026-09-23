import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from common import models as m
from common import uns
from tests.samples import BATCH, TS, UNIT, samples


@pytest.mark.parametrize(("topic", "payload"), samples(), ids=lambda x: str(x)[:60])
def test_every_topic_class_roundtrips(topic, payload):
    assert isinstance(payload, m.model_for(topic))
    data = m.encode(payload)
    assert m.decode(topic, data) == payload


@pytest.mark.parametrize(("topic", "payload"), samples(), ids=lambda x: str(x)[:60])
def test_six_keys_everywhere_except_edge(topic, payload):
    body = json.loads(m.encode(payload))
    if topic.startswith("edge/"):
        assert set(body) == {"tag", "value", "t", "q"}
    else:
        assert set(body) == {"v", "ts", "unit", "q", "batch", "src"}


def test_ts_format():
    body = json.loads(m.encode(samples()[0][1]))
    assert body["ts"] == "2026-09-22T10:15:05.123Z"


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError):
        m.ScalarPayload(v=1.0, ts=datetime(2026, 1, 1), unit=None, batch=None, src="sim")


@pytest.mark.parametrize("bad", ["B26-0142", "b2026-0142", "B2026-142", "2026-0142"])
def test_batch_id_format(bad):
    with pytest.raises(ValidationError):
        m.ScalarPayload(v=1.0, ts=TS, unit=None, batch=bad, src="sim")


def test_empty_payload_is_a_clear():
    assert m.decode(uns.ai_alert(UNIT, "stats-co2_flow"), b"") is None


def test_wrong_v_type_rejected():
    topic = uns.state_operation(UNIT)
    data = json.dumps(
        {
            "v": "Brewing",
            "ts": "2026-09-22T10:15:05.000Z",
            "unit": None,
            "q": "GOOD",
            "batch": BATCH,
            "src": "sim",
        }
    ).encode()
    with pytest.raises(ValidationError):
        m.decode(topic, data)


def test_batch_event_discriminator():
    topic = uns.events(UNIT, "batch")
    data = json.dumps(
        {
            "v": {
                "kind": "OPERATION_CHANGE",
                "batch_id": BATCH,
                "previous": "Growth",
                "current": "TempShift",
            },
            "ts": "2026-09-22T10:15:05.000Z",
            "unit": None,
            "q": "GOOD",
            "batch": BATCH,
            "src": "sim",
        }
    ).encode()
    decoded = m.decode(topic, data)
    assert isinstance(decoded.v, m.OperationChanged)


def test_models_are_frozen_and_strict():
    p = samples()[0][1]
    with pytest.raises(ValidationError):
        p.v = 1.0
    with pytest.raises(ValidationError):
        m.ScalarPayload(v=1.0, ts=TS, unit=None, batch=None, src="sim", extra=1)
