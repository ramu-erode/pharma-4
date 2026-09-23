-- Pre-attributed intervals, projected from the graph by graph-sync (ADR-0011).
-- A training join is `ts BETWEEN t_start AND t_end` on topic; no binding logic in SQL.
-- phase/role are null when no bound phase ran: operation carries the fallback.
CREATE TABLE tag_attribution (
    topic        text        NOT NULL,
    batch_id     text        NOT NULL,
    operation    text        NOT NULL,
    phase        text,
    role         text,
    phase_state  text,
    t_start      timestamptz NOT NULL,
    t_end        timestamptz NOT NULL
);

CREATE INDEX tag_attribution_lookup ON tag_attribution (topic, t_start, t_end);
CREATE INDEX tag_attribution_batch ON tag_attribution (batch_id);
