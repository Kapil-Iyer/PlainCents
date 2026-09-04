-- Phase 12B: track intentionally-excluded rows (credits/deposits, unsupported
-- currency) separately from malformed rows_unparseable, so the preview/
-- result can distinguish "this row was recognized and correctly skipped"
-- from "this row could not be parsed at all" (Phase 12A.5 §17/§24).
ALTER TABLE import_batches ADD COLUMN rows_skipped_credit INTEGER NOT NULL DEFAULT 0;
ALTER TABLE import_batches ADD COLUMN rows_skipped_currency INTEGER NOT NULL DEFAULT 0;
