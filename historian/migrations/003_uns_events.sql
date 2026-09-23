-- Structured UNS messages (state, events, alerts, predictions, recommendations), as the
-- full six-key payload. History of what was said, not relationships (ADR-0004).
CREATE TABLE uns_events (
    ts        timestamptz NOT NULL,
    topic     text        NOT NULL,
    batch_id  text,
    payload   jsonb       NOT NULL
);

CREATE INDEX uns_events_batch ON uns_events (batch_id, ts);
CREATE INDEX uns_events_topic ON uns_events (topic, ts);
