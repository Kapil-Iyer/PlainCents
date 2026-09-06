"""
PowerBIExportService (PATCH D: on-demand Power BI "current-state" export).

Generates a ZIP of CSVs on demand, entirely in memory (no temp files, no
disk artifact) from the LIVE V2 database via the same repositories every
other read endpoint uses. This is a deliberate rewrite, not a port: V1's
viz/powerbi_export.py (still preserved untouched per docs/V2_TRD.md's
LEGACY/PRESERVE table) writes CSV files to disk keyed by `session_id`, a
concept that does not exist in V2 -- there is no "session" here, only the
one live database and its current data_mode. Only the *idea* of a flat,
Power-BI-friendly CSV bundle is reused; none of that module's code path.

Every query here is scoped by the caller-supplied `data_mode`, exactly like
the route-decides/service-executes convention transactions.py, dashboard.py
and holdings.py already follow (never one-off logic that could disagree with
what the rest of the app shows for the same mode). EMPTY mode (`data_mode is
None`) still produces a valid ZIP -- four CSVs with headers and no rows --
rather than an error, matching the rest of the app's "honest empty state,
never a crash" convention.

Field selection is deliberately narrow (privacy-safe): raw_description (the
untouched bank text, which can carry masked account/reference numbers),
merchant_key (an internal matching key), decision_source/model_category (
internal/advisory debug metadata never surfaced as a decided value anywhere
else in the app) are all excluded. Only what a spending report actually
needs is included: the effective category a transaction is actually filed
under, never the system's raw or advisory opinions about it.
"""
from __future__ import annotations

import io
import sqlite3
import zipfile
from datetime import date

import pandas as pd

from backend.repositories.transaction_repository import TransactionRepository
from backend.repositories.forecast_repository import ForecastRepository
from backend.services.portfolio_service import PortfolioService

# Column order/names, chosen to read naturally once opened in Power BI --
# not a database dump. Renaming `effective_category` to `category` here
# specifically because that column IS the one and only category value a
# Power BI report should ever group or filter by.
_TRANSACTION_COLUMNS = ["date", "merchant", "amount", "bank_source", "category", "is_manual_override"]
_CATEGORY_SUMMARY_COLUMNS = ["month", "category", "total_spend"]
_PORTFOLIO_COLUMNS = [
    "ticker", "shares", "avg_cost", "current_price", "current_value", "pnl", "price_last_updated",
]
_FORECAST_COLUMNS = [
    "category", "forecast_month", "month_offset", "predicted_amount", "is_available",
    "generated_at", "is_stale",
]


def _to_csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    """A DataFrame built even from an empty `rows` list still carries the
    right header row (`columns=columns` on an empty frame), so a mode with
    no data yet exports a valid, honestly-empty CSV rather than nothing."""
    df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
    df = df.reindex(columns=columns)
    return df.to_csv(index=False).encode("utf-8")


class PowerBIExportService:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._txn_repo = TransactionRepository(conn)
        self._forecast_repo = ForecastRepository(conn)
        self._portfolio_service = PortfolioService(conn)

    def _transactions_csv(self, data_mode: str | None) -> bytes:
        rows = self._txn_repo.list(data_mode=data_mode, sort="date")
        renamed = [
            {
                "date": r["date"],
                "merchant": r["merchant"],
                "amount": r["amount"],
                "bank_source": r["bank_source"],
                "category": r["effective_category"],
                "is_manual_override": bool(r["is_manual_override"]),
            }
            for r in rows
        ]
        return _to_csv_bytes(renamed, _TRANSACTION_COLUMNS)

    def _category_summary_csv(self, data_mode: str | None) -> bytes:
        rows = self._txn_repo.aggregate_by_month_category(data_mode=data_mode)
        return _to_csv_bytes(rows, _CATEGORY_SUMMARY_COLUMNS)

    def _portfolio_csv(self, data_mode: str | None) -> bytes:
        # PortfolioService.get_holdings_with_prices is the same DB-only,
        # never-calls-fetch_price read path GET /api/holdings uses (TRD
        # §13.2) -- this export can never trigger a live market lookup.
        rows = self._portfolio_service.get_holdings_with_prices(data_mode)
        return _to_csv_bytes(rows, _PORTFOLIO_COLUMNS)

    def _forecast_csv(self, data_mode: str | None) -> bytes:
        run = self._forecast_repo.get_latest_run(data_mode=data_mode)
        if run is None:
            return _to_csv_bytes([], _FORECAST_COLUMNS)
        predictions = self._forecast_repo.get_predictions(run["id"])
        rows = [
            {
                "category": p["category"],
                "forecast_month": p["forecast_month"],
                "month_offset": p["month_offset"],
                "predicted_amount": p["predicted_amount"],
                "is_available": bool(p["is_available"]),
                "generated_at": run["generated_at"],
                "is_stale": bool(run["is_stale"]),
            }
            for p in predictions
        ]
        return _to_csv_bytes(rows, _FORECAST_COLUMNS)

    def build_export_zip(self, data_mode: str | None) -> bytes:
        """The whole export: four CSVs (transactions, category_summary,
        portfolio, forecast) zipped together in memory. Nothing is written
        to disk and nothing here calls yfinance or re-runs the categorizer
        or forecaster -- purely a read of already-persisted, already-decided
        data, generated fresh on every call so it always reflects the
        current live state."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("transactions.csv", self._transactions_csv(data_mode))
            zf.writestr("category_summary.csv", self._category_summary_csv(data_mode))
            zf.writestr("portfolio.csv", self._portfolio_csv(data_mode))
            zf.writestr("forecast.csv", self._forecast_csv(data_mode))
        return buffer.getvalue()


def export_filename(as_of: date | None = None) -> str:
    """`plaincents_export_YYYY-MM-DD.zip` -- a fresh, dated name each day so
    re-downloading the same day overwrites a browser's prior download of it,
    while a different day's export is never silently mistaken for today's."""
    day = as_of or date.today()
    return f"plaincents_export_{day.isoformat()}.zip"
