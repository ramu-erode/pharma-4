-- What the one-shot bootstrap has already done (ADR-0010). Lets an interrupted backfill
-- be redone exactly (same plan end date, same batches) instead of being skipped.
CREATE TABLE bootstrap_state (
    key         text PRIMARY KEY,
    value       jsonb       NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);
