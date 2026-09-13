-- PlainCents: spending eligibility -- internal/self account transfers must
-- not count as spending.
--
-- WHY THIS EXISTS
-- ---------------
-- Every transaction persisted so far has been summed into spend totals,
-- forecast input, and the Power BI category summary purely because it was a
-- negative-amount debit row -- there was no concept, anywhere in the schema,
-- of "this specific row does not represent consumption." A same-owner
-- account-to-account transfer (RBC chequing -> RBC savings, an "online
-- banking transfer") is exactly such a row: real money movement, zero
-- consumption.
--
-- `transaction_type` is the new, minimal, orthogonal concept:
--   'spending'          -- counts toward every spend total, forecast input,
--                          and category summary (the default; every
--                          pre-existing row backfills to this value, which
--                          is correct -- see the migration runner's own
--                          backward-compatibility guarantee).
--   'internal_transfer' -- a structurally-detected same-owner account
--                          transfer (backend.services.transfer_eligibility).
--                          Still visible in Transactions (labeled "Internal
--                          transfer"), still exported in transactions.csv,
--                          but excluded from every spend/forecast/category
--                          aggregate.
--
-- No CHECK constraint, deliberately -- same precedent as
-- 005_transaction_decision_source.sql's decision_source column: an
-- open-ended TEXT value means a future third value never requires a schema
-- migration, only an application-level change.
--
-- staged_transactions gets the same column, NULLABLE with no default (same
-- pattern as merchant_key/decision_source in 004_staged_decision_columns.sql)
-- -- IngestionService computes and stages it explicitly at Preview time, so
-- Confirm re-validates rather than re-decides, exactly like every other
-- decision field on that table.
ALTER TABLE transactions ADD COLUMN transaction_type TEXT NOT NULL DEFAULT 'spending';
ALTER TABLE staged_transactions ADD COLUMN transaction_type TEXT;

-- import_batches gets a count of how many staged rows were structurally
-- excluded as internal transfers, alongside the existing rows_skipped_credit
-- / rows_skipped_currency counts (002_import_exclusion_counts.sql) -- so
-- Preview/Result can truthfully say "N internal transfers excluded" rather
-- than silently dropping them from the spend total with no explanation.
ALTER TABLE import_batches ADD COLUMN rows_internal_transfer INTEGER NOT NULL DEFAULT 0;

-- v_transactions_effective is SELECT t.* so it picks the new column up
-- automatically, but the view was created against the old column list and
-- SQLite caches that, so it is recreated here (same pattern as every prior
-- migration that touches `transactions`).
DROP VIEW IF EXISTS v_transactions_effective;
CREATE VIEW v_transactions_effective AS
SELECT
    t.*,
    COALESCE(t.confirmed_category, t.predicted_category) AS effective_category,
    (t.confirmed_category IS NOT NULL) AS is_manual_override
FROM transactions t;
