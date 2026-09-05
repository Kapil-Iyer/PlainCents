"""
Phase 1 database/migration tests.
Requirements #1-6 from the Phase 1 implementation prompt.
"""
import sqlite3
from pathlib import Path

import pytest

from backend.db.migration_runner import apply_migrations

ROOT = Path(__file__).resolve().parent.parent.parent.parent
MIGRATIONS_DIR = ROOT / "db" / "migrations"

EXPECTED_TABLES = {
    "app_state", "import_batches", "staged_transactions", "transactions",
    "forecast_runs", "forecast_predictions", "holdings", "price_cache",
    "schema_migrations",
}
EXPECTED_VIEWS = {"v_transactions_effective"}
EXPECTED_INDEXES = {
    "idx_import_batches_status", "idx_staged_txn_batch", "idx_transactions_date",
    "idx_transactions_data_mode", "idx_transactions_mode_date",
    "idx_forecast_runs_mode_time", "idx_holdings_data_mode",
}


def _fresh_conn(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def test_1_fresh_db_migration_succeeds(db_path):
    conn = _fresh_conn(db_path)
    applied = apply_migrations(conn, migrations_dir=MIGRATIONS_DIR)
    # 002 added Phase 12B's import exclusion-count columns; 003/004 added
    # ML-G's stable merchant identity and the staged decision columns that
    # let Preview persist the same decision Confirm stores.
    assert applied == [1, 2, 3, 4]
    conn.close()


def test_2_migrations_run_twice_no_duplicate_application(db_path):
    conn = _fresh_conn(db_path)
    first = apply_migrations(conn, migrations_dir=MIGRATIONS_DIR)
    second = apply_migrations(conn, migrations_dir=MIGRATIONS_DIR)
    assert first == [1, 2, 3, 4]
    assert second == []  # nothing new applied the second time
    conn.close()


def test_3_schema_migrations_tracks_versions(conn):
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert [r["version"] for r in rows] == [1, 2, 3, 4]


def test_4_all_required_tables_views_indexes_exist(conn):
    tables = {
        r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    views = {
        r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'view'"
        ).fetchall()
    }
    indexes = {
        r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }
    assert EXPECTED_TABLES.issubset(tables)
    assert EXPECTED_VIEWS.issubset(views)
    assert EXPECTED_INDEXES.issubset(indexes)


def test_5_foreign_keys_pragma_is_on(conn):
    row = conn.execute("PRAGMA foreign_keys").fetchone()
    assert row[0] == 1


def test_6_invalid_fk_insert_fails(conn):
    # transactions.import_batch_id references import_batches(id); 999 doesn't exist.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO transactions
                (date, merchant, amount, predicted_category, import_batch_id, data_mode, dedup_key)
            VALUES ('2026-01-01', 'TEST MERCHANT', 10.0, 'Other', 999, 'real', 'dk-fk-test')
            """
        )
