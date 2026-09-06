-- PlainCents: persist the raw classifier opinion (model_category) on
-- `transactions`, the same pattern migration 005 used for decision_source.
--
-- model_category already exists on staged_transactions (migration 004) and
-- on CategoryDecision (backend/services/category_decision.py) for every
-- model/abstention-path row, but IngestionService.commit_import() discarded
-- it at insert -- Confirm only ever kept predicted_category/decision_source.
-- That was fine while nothing downstream needed the raw model opinion after
-- Confirm; it is no longer fine now that the product wants to show a
-- low-confidence abstention's advisory suggestion ("Suggested: Transport")
-- on a CONFIRMED transaction, not just in Preview.
--
-- DATA SEMANTICS (see backend/services/category_decision.py, frontend
-- CategoryBadge.tsx): model_category is advisory model metadata only.
--   - For a low-confidence abstention: predicted_category stays "Other"
--     (the persisted, served decision); model_category stores the model's
--     best non-abstained guess; confirmed_category stays NULL until the
--     user explicitly accepts/corrects it.
--   - effective_category (COALESCE(confirmed_category, predicted_category))
--     is completely unaffected by this column -- model_category must NEVER
--     become effective_category automatically. Accepting the suggestion
--     ("Use {model_category}") writes confirmed_category through the
--     existing PATCH /transactions/{id} path, exactly like any other manual
--     correction; it does not read or write model_category itself.
--
-- Additive-only, backward-safe: NULL for every pre-existing row (never
-- recorded) and for manual entries (TransactionService.create_manual()
-- never calls decide()/decide_batch(), so there is no model opinion to
-- store) and for structural/ambiguous-e-transfer rows (the model is never
-- called on those paths -- CategoryDecision.model_category is already None
-- there, same as it is for decision_source's "no reason to record" cases).
ALTER TABLE transactions ADD COLUMN model_category TEXT;

-- v_transactions_effective is SELECT t.* so it picks the new column up
-- automatically, but the view was created against the old column list and
-- SQLite caches that, so it is recreated here (same pattern as migrations
-- 003_merchant_key.sql and 005_transaction_decision_source.sql). Reproduces
-- the view exactly as migration 005 left it -- only `transactions`' own
-- column set changes (via t.*); effective_category/is_manual_override are
-- untouched.
DROP VIEW IF EXISTS v_transactions_effective;
CREATE VIEW v_transactions_effective AS
SELECT
    t.*,
    COALESCE(t.confirmed_category, t.predicted_category) AS effective_category,
    (t.confirmed_category IS NOT NULL) AS is_manual_override
FROM transactions t;
