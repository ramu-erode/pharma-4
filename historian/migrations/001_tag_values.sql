-- Every scalar UNS value (pv, sp, lab, ai score). ADR-0004.
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE tag_values (
    ts        timestamptz      NOT NULL,
    topic     text             NOT NULL,
    batch_id  text,
    value     double precision NOT NULL,
    quality   text             NOT NULL
);

SELECT create_hypertable('tag_values', by_range('ts', INTERVAL '7 days'));

CREATE INDEX tag_values_topic_ts ON tag_values (topic, ts DESC);
CREATE INDEX tag_values_batch ON tag_values (batch_id, ts);
