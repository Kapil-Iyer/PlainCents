"""
Phase 4: SQLite connection helper and query/insert functions for the 6-table schema.
No pipeline imports. Uses config.DB_PATH for database location.
"""
import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from config import DB_PATH

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    """Return a connection to the PlainCents SQLite database, creating tables if needed."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    return conn


# ── INSERT functions ──────────────────────────────────────


def insert_transactions(conn: sqlite3.Connection, df: pd.DataFrame, session_id: str) -> int:
    """Bulk insert transactions from DataFrame. Returns row count inserted."""
    if df is None or df.empty:
        logger.warning("insert_transactions: empty DataFrame, skipping.")
        return 0
    rows = []
    for _, r in df.iterrows():
        rows.append((
            session_id,
            str(r["date"]),
            str(r["merchant"]),
            float(r["amount"]),
            str(r["category"]),
            int(r["cluster_id"]) if "cluster_id" in r and pd.notna(r.get("cluster_id")) else None,
        ))
    conn.executemany(
        "INSERT INTO transactions (session_id, date, merchant, amount, category, cluster_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info("Inserted %d transactions (session_id=%s).", len(rows), session_id)
    return len(rows)


def insert_predictions(conn: sqlite3.Connection, df: pd.DataFrame, session_id: str) -> int:
    """Bulk insert forecast predictions from DataFrame."""
    if df is None or df.empty:
        logger.warning("insert_predictions: empty DataFrame, skipping.")
        return 0
    rows = []
    for _, r in df.iterrows():
        rows.append((
            session_id,
            str(r["category"]),
            int(r["month_offset"]),
            str(r["forecast_month"]),
            float(r["predicted_amount"]),
        ))
    conn.executemany(
        "INSERT INTO predictions (session_id, category, month_offset, forecast_month, predicted_amount) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info("Inserted %d predictions (session_id=%s).", len(rows), session_id)
    return len(rows)


def insert_portfolio(conn: sqlite3.Connection, rows: list[dict], session_id: str) -> int:
    """Insert portfolio holdings. Each dict: ticker, shares, avg_cost, current_price, pnl."""
    if not rows:
        logger.warning("insert_portfolio: empty list, skipping.")
        return 0
    values = []
    for r in rows:
        values.append((
            session_id,
            r["ticker"],
            r["shares"],
            r["avg_cost"],
            r.get("current_price"),
            r.get("pnl"),
        ))
    conn.executemany(
        "INSERT INTO portfolio (session_id, ticker, shares, avg_cost, current_price, pnl) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        values,
    )
    conn.commit()
    logger.info("Inserted %d portfolio rows (session_id=%s).", len(values), session_id)
    return len(values)


def upsert_monthly_summary(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """
    UPSERT monthly_summary rows. Unique key: month.
    Each dict: month, total_spend, category_spend_json, forecast_next_month, portfolio_value.
    """
    if not rows:
        logger.warning("upsert_monthly_summary: empty list, skipping.")
        return 0
    values = []
    for r in rows:
        cat_json = r.get("category_spend_json")
        if isinstance(cat_json, dict):
            cat_json = json.dumps(cat_json)
        values.append((
            r["month"],
            r["total_spend"],
            cat_json,
            r.get("forecast_next_month"),
            r.get("portfolio_value"),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO monthly_summary "
        "(month, total_spend, category_spend_json, forecast_next_month, portfolio_value) "
        "VALUES (?, ?, ?, ?, ?)",
        values,
    )
    conn.commit()
    logger.info("Upserted %d monthly_summary rows.", len(values))
    return len(values)


def insert_forecast_vs_actual(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert forecast_vs_actual rows for monitoring."""
    if not rows:
        logger.warning("insert_forecast_vs_actual: empty list, skipping.")
        return 0
    values = []
    for r in rows:
        values.append((
            r["category"],
            r["forecast_month"],
            r["predicted_value"],
            r["actual_value"],
            r.get("absolute_error"),
            r.get("pct_error"),
        ))
    conn.executemany(
        "INSERT INTO forecast_vs_actual "
        "(category, forecast_month, predicted_value, actual_value, absolute_error, pct_error) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        values,
    )
    conn.commit()
    logger.info("Inserted %d forecast_vs_actual rows.", len(values))
    return len(values)


def upsert_price_cache(conn: sqlite3.Connection, ticker: str, price: float, fetched_at: str) -> None:
    """UPSERT a single price_cache row. Unique key: ticker."""
    conn.execute(
        "INSERT OR REPLACE INTO price_cache (ticker, current_price, fetched_at) "
        "VALUES (?, ?, ?)",
        (ticker, price, fetched_at),
    )
    conn.commit()
    logger.info("Upserted price_cache: %s = %.2f @ %s", ticker, price, fetched_at)


# ── QUERY helpers ─────────────────────────────────────────


def get_transactions(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM transactions ORDER BY date", conn)


def get_predictions(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM predictions ORDER BY forecast_month, category", conn)


def get_monthly_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM monthly_summary ORDER BY month", conn)


def get_forecast_accuracy(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM forecast_vs_actual ORDER BY forecast_month, category", conn)


def get_portfolio(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM portfolio ORDER BY ticker", conn)


def get_price_cache(conn: sqlite3.Connection, ticker: str) -> dict | None:
    """Return the cached price row for a ticker, or None if not found."""
    row = conn.execute(
        "SELECT ticker, current_price, fetched_at FROM price_cache WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    if row is None:
        return None
    return {"ticker": row[0], "current_price": row[1], "fetched_at": row[2]}
