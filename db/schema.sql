-- PlainCents SQLite Schema — 6 tables
-- All tables are written to and queried during normal pipeline execution.

-- Table 1: transactions
CREATE TABLE IF NOT EXISTS transactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    date            TEXT NOT NULL,
    merchant        TEXT NOT NULL,
    amount          REAL NOT NULL,
    category        TEXT NOT NULL,
    cluster_id      INTEGER,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: predictions
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    category        TEXT NOT NULL,
    month_offset    INTEGER NOT NULL,
    forecast_month  TEXT NOT NULL,
    predicted_amount REAL NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Table 3: portfolio
CREATE TABLE IF NOT EXISTS portfolio (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    shares          REAL NOT NULL,
    avg_cost        REAL NOT NULL,
    current_price   REAL,
    pnl             REAL
);

-- Table 4: price_cache
-- Check before every yfinance call. Fetch only if fetched_at > 1 hour ago.
CREATE TABLE IF NOT EXISTS price_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    current_price   REAL NOT NULL,
    fetched_at      DATETIME NOT NULL
);

-- Table 5: monthly_summary
-- UPSERT: INSERT OR REPLACE to prevent UNIQUE constraint errors on re-runs.
CREATE TABLE IF NOT EXISTS monthly_summary (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    month               TEXT NOT NULL UNIQUE,
    total_spend         REAL NOT NULL,
    category_spend_json TEXT,
    forecast_next_month REAL,
    portfolio_value     REAL,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Table 6: forecast_vs_actual
-- Written when actual month data arrives. Seed script pre-populates 3 months for demo.
CREATE TABLE IF NOT EXISTS forecast_vs_actual (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    category        TEXT NOT NULL,
    forecast_month  TEXT NOT NULL,
    predicted_value REAL NOT NULL,
    actual_value    REAL NOT NULL,
    absolute_error  REAL,
    pct_error       REAL
);
