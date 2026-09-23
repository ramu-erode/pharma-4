-- Ground truth for evaluation only (ADR-0012). Never read by the AI services.
CREATE TABLE fault_labels (
    id        text PRIMARY KEY,
    cell      text        NOT NULL,
    batch_id  text        NOT NULL,
    fault     text        NOT NULL,
    onset     timestamptz NOT NULL,
    "end"     timestamptz,
    params    jsonb       NOT NULL DEFAULT '{}'
);

CREATE INDEX fault_labels_batch ON fault_labels (batch_id);
