-- One-minute continuous aggregate, for dashboard charts only (features use raw rows).
-- Real-time mode: unmaterialised rows are unioned in at query time. That covers live data,
-- whose simulated time runs ahead of the wall clock (ADR-0009), and the history.
-- Only the last 30 days are materialised: deadbanded data arrives at about one row a
-- minute, so materialising all history would store an uncompressed copy of it
-- (measured: 2.5 GB for 200 batches, against 107 MB compressed raw).
CREATE MATERIALIZED VIEW tag_values_1m
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT time_bucket(INTERVAL '1 minute', ts) AS bucket,
       topic,
       batch_id,
       avg(value) AS avg,
       min(value) AS min,
       max(value) AS max,
       last(value, ts) AS last
FROM tag_values
GROUP BY bucket, topic, batch_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'tag_values_1m',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '10 minutes'
);
