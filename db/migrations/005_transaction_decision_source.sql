-- PlainCents: persist WHY a confirmed transaction's predicted_category is
-- what it is, so the distinction survives Confirm/reload -- not just Preview.
--
-- decision_source already exists on staged_transactions (migration 004) but
-- was never carried onto `transactions` -- IngestionService.commit_import()
-- computed/staged it, then discarded it at insert. That was fine while
-- nothing downstream needed the reason after Confirm; it is no longer fine
-- now that the product wants to distinguish, e.g.:
--
--   CASE 1: a genuine miscellaneous purchase        -> effective_category = Other
--   CASE 2: a purposeless/ambiguous E-Transfer       -> effective_category = Other,
--                                                       decision_source = 'ambiguous_e_transfer'
--
-- Both cases already look identical in predicted_category/confirmed_category
-- alone -- decision_source is what tells them apart, and it must be a
-- STORED fact about how the row was originally decided, not something
-- recomputed from merchant text on every read (this codebase's HITL model
-- already treats predicted_category itself the same way: frozen at
-- decide-time, not silently reinterpreted later if the policy changes).
--
-- Additive-only, backward-safe: NULL for every pre-existing row (their
-- reason was simply never recorded) and for manual entries (TransactionService
-- .create_manual() never calls decide()/decide_batch() at all -- see its own
-- docstring -- so there is no decision-path "reason" to store for those).
--
-- Values: 'model' | 'structural_other' | 'low_confidence_other' |
--         'gazetteer' | 'ambiguous_e_transfer' | NULL
-- (backend.services.category_decision's SOURCE_* constants -- no CHECK
-- constraint here, same as staged_transactions.decision_source, so adding a
-- new source value later never requires a migration.)
ALTER TABLE transactions ADD COLUMN decision_source TEXT;

-- v_transactions_effective is SELECT t.* so it picks the new column up
-- automatically, but the view was created against the old column list and
-- SQLite caches that, so it is recreated here (same pattern as migration
-- 003_merchant_key.sql).
DROP VIEW IF EXISTS v_transactions_effective;
CREATE VIEW v_transactions_effective AS
SELECT
    t.*,
    COALESCE(t.confirmed_category, t.predicted_category) AS effective_category,
    (t.confirmed_category IS NOT NULL) AS is_manual_override
FROM transactions t;
