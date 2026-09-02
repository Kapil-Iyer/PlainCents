"""PriceCacheRepository — persistence only (TRD §8, §4.8). Only the latest
observation per ticker is stored, matching V1's shape. No TTL/staleness
gating here — that is a service-layer read-vs-refresh decision (TRD §13.3),
not something this repository decides."""
import sqlite3

from backend.repositories.money import round_money


class PriceCacheRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def get_last_known(self, ticker: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM price_cache WHERE ticker = ?", (ticker,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_latest(self, ticker: str, price: float, fetched_at: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO price_cache (ticker, current_price, fetched_at) "
            "VALUES (?, ?, ?)",
            (ticker, round_money(price), fetched_at),
        )
