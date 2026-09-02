-- PlainCents V2 — Initial schema
-- Source of truth: TRD §4.11 (docs/V2_TRD.md). Verbatim per the frozen DDL.
-- db/migrations/ is the SOLE schema source of truth for V2 (TRD §4.12) —
-- there is no separate schema_v2.sql file.

CREATE TABLE IF NOT EXISTS app_state (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    mode            TEXT NOT NULL CHECK (mode IN ('EMPTY','DEMO','REAL')) DEFAULT 'EMPTY',
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT OR IGNORE INTO app_state (id, mode) VALUES (1, 'EMPTY');

CREATE TABLE IF NOT EXISTS import_batches (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_source         TEXT NOT NULL,
    original_filename   TEXT,
    status              TEXT NOT NULL CHECK (status IN ('previewing','confirmed','failed')),
    data_mode           TEXT NOT NULL CHECK (data_mode IN ('demo','real')) DEFAULT 'real',
    rows_valid          INTEGER NOT NULL DEFAULT 0,
    rows_unparseable    INTEGER NOT NULL DEFAULT 0,
    rows_duplicate      INTEGER NOT NULL DEFAULT 0,
    rows_imported       INTEGER NOT NULL DEFAULT 0,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    confirmed_at        DATETIME
);
CREATE INDEX IF NOT EXISTS idx_import_batches_status ON import_batches(status);

CREATE TABLE IF NOT EXISTS staged_transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    import_batch_id     INTEGER NOT NULL REFERENCES import_batches(id) ON DELETE CASCADE,
    date                TEXT NOT NULL,
    raw_description     TEXT,
    merchant            TEXT NOT NULL,
    amount              REAL NOT NULL,
    predicted_category  TEXT,
    dedup_key           TEXT NOT NULL,
    is_duplicate        INTEGER NOT NULL DEFAULT 0,
    is_valid            INTEGER NOT NULL DEFAULT 1,
    invalid_reason      TEXT
);
CREATE INDEX IF NOT EXISTS idx_staged_txn_batch ON staged_transactions(import_batch_id);

CREATE TABLE IF NOT EXISTS transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    date                TEXT NOT NULL,
    raw_description     TEXT,
    merchant            TEXT NOT NULL,
    amount              REAL NOT NULL,
    bank_source         TEXT,
    predicted_category  TEXT NOT NULL,
    confirmed_category  TEXT,
    import_batch_id     INTEGER REFERENCES import_batches(id) ON DELETE SET NULL,
    data_mode           TEXT NOT NULL CHECK (data_mode IN ('demo','real')),
    dedup_key           TEXT NOT NULL,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dedup_key)
);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_transactions_data_mode ON transactions(data_mode);
CREATE INDEX IF NOT EXISTS idx_transactions_mode_date ON transactions(data_mode, date);

CREATE VIEW IF NOT EXISTS v_transactions_effective AS
SELECT
    t.*,
    COALESCE(t.confirmed_category, t.predicted_category) AS effective_category,
    (t.confirmed_category IS NOT NULL) AS is_manual_override
FROM transactions t;

CREATE TABLE IF NOT EXISTS forecast_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    months_available    INTEGER NOT NULL,
    months_required     INTEGER NOT NULL DEFAULT 12,
    is_stale            INTEGER NOT NULL DEFAULT 0,
    stale_reason        TEXT,
    data_mode           TEXT NOT NULL CHECK (data_mode IN ('demo','real')),
    model_impl_version  TEXT
);
CREATE INDEX IF NOT EXISTS idx_forecast_runs_mode_time ON forecast_runs(data_mode, generated_at DESC);

CREATE TABLE IF NOT EXISTS forecast_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_run_id     INTEGER NOT NULL REFERENCES forecast_runs(id) ON DELETE CASCADE,
    category            TEXT NOT NULL,
    forecast_month      TEXT NOT NULL,
    month_offset        INTEGER NOT NULL CHECK (month_offset IN (1,2,3)),
    predicted_amount    REAL,
    is_available        INTEGER NOT NULL DEFAULT 1,
    unavailable_reason  TEXT,
    UNIQUE (forecast_run_id, category, forecast_month)
);

CREATE TABLE IF NOT EXISTS holdings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT NOT NULL,
    shares              REAL NOT NULL CHECK (shares > 0),
    avg_cost            REAL NOT NULL CHECK (avg_cost >= 0),
    data_mode           TEXT NOT NULL CHECK (data_mode IN ('demo','real')),
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_holdings_data_mode ON holdings(data_mode);

CREATE TABLE IF NOT EXISTS price_cache (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT NOT NULL,
    current_price       REAL NOT NULL,
    fetched_at          DATETIME NOT NULL,
    UNIQUE (ticker)
);
