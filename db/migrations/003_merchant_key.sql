-- PlainCents ML-G: stable merchant identity for correction memory.
--
-- WHY: correction memory matched a prior user correction by the EXACT
-- `merchant` string. Real bank descriptions embed a per-transaction card
-- suffix / store number / reference code, so the same merchant produces a
-- different `merchant` string every time and the exact match essentially
-- never fired. See backend/services/merchant_identity.py.
--
-- `merchant_key` is a deterministic, bank-scoped identity derived from the
-- merchant-identity tokens that survive boilerplate and reference-noise
-- removal. It is NULL for rows whose text names nothing (a generic
-- e-transfer, an ABM withdrawal), which is exactly what keeps unrelated
-- transfers from collapsing into one shared memory entry.
--
-- The `merchant` column itself is untouched: display, search and dedup keys
-- all keep using the exact text the bank sent. This column is for lookup
-- only.
--
-- Existing rows are backfilled in Python at startup (the derivation needs
-- application logic, not SQL) -- see backend/db/backfill.py. That backfill
-- is idempotent and only touches rows where merchant_key IS NULL.

ALTER TABLE transactions ADD COLUMN merchant_key TEXT;

CREATE INDEX IF NOT EXISTS idx_transactions_merchant_key
    ON transactions(merchant_key, confirmed_category);

-- v_transactions_effective is SELECT t.* so it picks the new column up
-- automatically, but the view was created against the old column list and
-- SQLite caches that, so it is recreated here.
DROP VIEW IF EXISTS v_transactions_effective;
CREATE VIEW v_transactions_effective AS
SELECT
    t.*,
    COALESCE(t.confirmed_category, t.predicted_category) AS effective_category,
    (t.confirmed_category IS NOT NULL) AS is_manual_override
FROM transactions t;
