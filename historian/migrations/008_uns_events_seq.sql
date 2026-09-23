-- Arrival order. Several state changes share one timestamp (an operation ends, the next
-- starts, its phases start), and replaying them into the graph needs the original order.
ALTER TABLE uns_events ADD COLUMN seq bigserial;
CREATE INDEX uns_events_seq ON uns_events (ts, seq);
