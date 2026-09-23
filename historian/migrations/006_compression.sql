-- Keep the laptop footprint small (implementation plan budget: <= 3 GB).
ALTER TABLE tag_values SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'topic',
    timescaledb.compress_orderby = 'ts'
);

SELECT add_compression_policy('tag_values', INTERVAL '7 days');
