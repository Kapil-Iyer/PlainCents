"""
Phase 5: Portfolio tracking with cache-first yfinance logic.
TTL 1 hour on price_cache. PnL computed from current_price - avg_cost.
"""
import logging
import math
from datetime import datetime

import pandas as pd
import yfinance as yf

from db.database import (
    get_connection,
    get_price_cache,
    insert_portfolio,
    upsert_price_cache,
)

logger = logging.getLogger(__name__)

TTL_SECONDS = 3600


def get_cached_price(conn, ticker: str):
    """
    Return cached price if present and fresh (fetched_at within TTL).
    Returns float or None. Uses datetime.now() for consistency (never utcnow).
    """
    result = get_price_cache(conn, ticker)
    if result is None:
        logger.info("cache miss: %s", ticker)
        return None

    fetched_at = datetime.fromisoformat(result["fetched_at"])
    now = datetime.now()
    delta_seconds = (now - fetched_at).total_seconds()

    if delta_seconds <= TTL_SECONDS:
        logger.info("cache hit: %s", ticker)
        return result["current_price"]

    logger.info("stale cache: %s", ticker)
    return None


def fetch_price(conn, ticker: str):
    """
    Cache-first: return cached price if valid. Else fetch from yfinance, upsert cache, return price.
    Never raises. Returns None on any failure.
    """
    cached = get_cached_price(conn, ticker)
    if cached is not None:
        return cached

    try:
        fast_info = yf.Ticker(ticker).fast_info
        try:
            raw = fast_info["lastPrice"]
        except (KeyError, TypeError, AttributeError):
            raw = getattr(fast_info, "last_price", None)
        if raw is None:
            logger.warning("invalid price for %s: missing lastPrice", ticker)
            return None
        price = float(raw)
        if math.isnan(price) or price <= 0:
            logger.warning("invalid price for %s: nan or <= 0", ticker)
            return None
        fetched_at = datetime.now().isoformat()
        upsert_price_cache(conn, ticker, price, fetched_at)
        logger.info("fetched from yfinance: %s = %.2f", ticker, price)
        return price
    except Exception as e:
        logger.warning("yfinance error for %s: %s", ticker, e)
        return None


def build_portfolio(conn, holdings: list[dict], session_id: str) -> pd.DataFrame:
    """
    For each holding (ticker, shares, avg_cost), fetch current price (cache-first),
    compute PnL, persist to portfolio table. Returns DataFrame of rows written.
    Portfolio is append-only; session_id distinguishes runs.
    """
    if holdings is None or len(holdings) == 0:
        logger.warning("No holdings provided — returning empty DataFrame")
        return pd.DataFrame()

    rows = []
    for holding in holdings:
        ticker = holding["ticker"]
        price = fetch_price(conn, ticker)
        if price is None:
            logger.warning("skipping %s — no price available", ticker)
            continue
        pnl = (price - holding["avg_cost"]) * holding["shares"]
        rows.append({
            "ticker": ticker,
            "shares": holding["shares"],
            "avg_cost": holding["avg_cost"],
            "current_price": price,
            "pnl": round(pnl, 2),
        })

    if not rows:
        logger.warning("No valid prices — returning empty DataFrame")
        return pd.DataFrame()

    insert_portfolio(conn, rows, session_id)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    HOLDINGS = [
        {"ticker": "AAPL", "shares": 15, "avg_cost": 145.00},
        {"ticker": "MSFT", "shares": 10, "avg_cost": 280.00},
        {"ticker": "SPY", "shares": 20, "avg_cost": 420.00},
        {"ticker": "BNS.TO", "shares": 50, "avg_cost": 65.00},
    ]

    conn = get_connection()

    print("=== Run 1: should call yfinance ===")
    df1 = build_portfolio(conn, HOLDINGS, session_id="TEST_001")
    print(df1.to_string(index=False))

    print("\n=== Run 2: should use cache ===")
    df2 = build_portfolio(conn, HOLDINGS, session_id="TEST_002")
    print(df2.to_string(index=False))
    print("Second run complete — cache used")

    conn.close()
