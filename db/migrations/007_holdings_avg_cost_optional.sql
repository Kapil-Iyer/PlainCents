-- PlainCents: make holdings.avg_cost optional (nullable).
--
-- PRODUCT DECISION: Ticker and Shares are required to track a holding;
-- Average Cost is not. A user who knows "I own 10 MSFT shares" but not
-- their exact cost basis must still be able to add the holding -- current
-- price and market value are still honest and computable from shares alone.
-- Cost basis and unrealized P&L are NEVER fabricated from current price,
-- demo price, or any default -- they stay genuinely NULL until the user
-- supplies (or calculates, via the purchase-lot helper) a real average
-- cost. See PortfolioService._to_response for the read-side null-safety
-- this enables.
--
-- SQLite has no ALTER TABLE ... ALTER COLUMN to drop a NOT NULL/CHECK
-- constraint, so this is the standard rebuild-and-swap: create the new
-- table shape, copy every existing row across unchanged (every existing
-- avg_cost value is preserved verbatim -- this migration never edits data,
-- only relaxes a constraint), drop the old table, rename the new one into
-- place, and recreate its index. No FK anywhere in this schema references
-- `holdings` (verified: no other migration declares
-- "REFERENCES holdings"), so there is nothing else to repoint.
CREATE TABLE holdings_new (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT NOT NULL,
    shares              REAL NOT NULL CHECK (shares > 0),
    avg_cost            REAL CHECK (avg_cost IS NULL OR avg_cost >= 0),
    data_mode           TEXT NOT NULL CHECK (data_mode IN ('demo','real')),
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO holdings_new (id, ticker, shares, avg_cost, data_mode, created_at, updated_at)
SELECT id, ticker, shares, avg_cost, data_mode, created_at, updated_at FROM holdings;

DROP TABLE holdings;
ALTER TABLE holdings_new RENAME TO holdings;

CREATE INDEX IF NOT EXISTS idx_holdings_data_mode ON holdings(data_mode);
