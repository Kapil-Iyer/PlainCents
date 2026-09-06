"""
DashboardService (TRD §7.6; Build Plan Phase 6).

get_summary() computes the Overview dashboard's numbers via live SQL
aggregation only — reads TransactionRepository, no ML calls, no yfinance
calls, no pipeline.forecast/pipeline.portfolio calls (TRD §22.H). It reuses
TransactionRepository.aggregate_by_month_category()/.list() (Phase 1) rather
than adding new repository methods, per Build Plan's "reuse existing
repository/service conventions."

forecast_summary/portfolio_summary are always None here — see
backend/schemas/dashboard.py's docstring for why (Build Plan Phase 6, item 12).
"""
import sqlite3
from datetime import date

from backend.repositories.transaction_repository import TransactionRepository
from backend.services.date_windows import elapsed_window, shift_month

# PRD §11.7 only requires current-vs-previous calendar month; the trend chart
# and recent-transactions list are additional visualizations the same section
# calls for ("a spending trend over time", "a list of recent transactions").
# Neither count is frozen by the PRD/TRD, so these are Build-Plan-Phase-6
# implementation choices, not spec values.
_TREND_MONTHS = 6
_RECENT_TRANSACTIONS_LIMIT = 5


def _month_str(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    """Shift a (year, month) pair by `delta` months (may be negative),
    wrapping the year — the one piece of arithmetic month-boundary edge
    cases (e.g. "current month is January") depend on getting right.

    Kept as a thin alias of date_windows.shift_month (rather than removed)
    so this module's existing internal call sites don't all need touching;
    the shared implementation now lives in date_windows.py."""
    return shift_month(year, month, delta)


def _change_pct(current: float, previous: float) -> float | None:
    """Percent change from previous to current. Undefined (None) when the
    previous period had zero spend and the current period doesn't — there is
    no meaningful percentage of a zero baseline. Both zero is reported as
    0.0% (no change), not None, since that comparison is well-defined."""
    if previous == 0:
        return 0.0 if current == 0 else None
    return round((current - previous) / previous * 100, 1)


def _category_breakdown(rows: list[dict], total_current: float) -> list[dict]:
    items = [
        {
            "category": r["category"],
            "total_spend": round(r["total_spend"], 2),
            "pct_of_total": round(r["total_spend"] / total_current * 100, 1) if total_current else 0.0,
        }
        for r in rows
    ]
    items.sort(key=lambda item: item["total_spend"], reverse=True)
    return items


def _spending_trend(today: date, monthly_totals: dict[str, float]) -> list[dict]:
    """Trailing `_TREND_MONTHS` months ending at the current calendar month,
    zero-filled for months with no transactions. This is real, computed
    information (a month genuinely had $0 spend), not fabricated data."""
    points = []
    for offset in range(_TREND_MONTHS - 1, -1, -1):
        year, month = _shift_month(today.year, today.month, -offset)
        m = f"{year:04d}-{month:02d}"
        points.append({"month": m, "total_spend": round(monthly_totals.get(m, 0.0), 2)})
    return points


class DashboardService:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._repo = TransactionRepository(conn)

    def get_summary(
        self,
        data_mode: str | None,
        app_mode: str,
        reference_date: date | None = None,
    ) -> dict:
        """
        `data_mode` is the repository-level filter ('real'/'demo'/None for
        EMPTY, from resolve_data_mode_filter — computed by the route, same as
        transactions.py) that decides which rows are read. `app_mode` is the
        EMPTY/DEMO/REAL app state, echoed back verbatim as the response's
        `data_mode` field (TRD §6). `reference_date` defaults to today; tests
        inject a fixed date to exercise month-boundary edge cases.
        """
        today = reference_date or date.today()
        window = elapsed_window(today)
        current_month = window.current_month
        previous_month = window.previous_month

        trend_start_year, trend_start_month = _shift_month(
            today.year, today.month, -(_TREND_MONTHS - 1)
        )
        trend_start = f"{trend_start_year:04d}-{trend_start_month:02d}-01"

        rows = self._repo.aggregate_by_month_category(data_mode=data_mode, date_from=trend_start)
        # aggregate_by_month_category has no upper bound applied here; filter
        # out anything after the current calendar month defensively (rows
        # aren't expected to be future-dated, but the trend/current-month
        # math must not silently include them if they exist).
        rows = [r for r in rows if r["month"] <= current_month]

        monthly_totals: dict[str, float] = {}
        current_month_rows: list[dict] = []
        for row in rows:
            monthly_totals[row["month"]] = monthly_totals.get(row["month"], 0.0) + row["total_spend"]
            if row["month"] == current_month:
                current_month_rows.append(row)

        total_spend_current = round(monthly_totals.get(current_month, 0.0), 2)
        # Full previous calendar month -- a genuinely useful standalone
        # number ("you spent $X last month total"), kept as-is.
        total_spend_previous = round(monthly_totals.get(previous_month, 0.0), 2)

        # The FAIR comparison basis: the previous month's spend through the
        # SAME day-of-month the current (possibly partial) month has reached.
        # Without this, a $0 first day of the month reads as "-100% vs last
        # month" against the full previous month's total, which is not a
        # meaningful statement about pace. This is the number `change_pct`
        # is computed against; `total_spend_previous` above stays the full
        # month for its own separate, honest standalone meaning.
        prev_to_date_rows = self._repo.aggregate_by_month_category(
            data_mode=data_mode, date_from=window.previous_start, date_to=window.previous_comparable_end,
        )
        total_spend_previous_to_date = round(sum(r["total_spend"] for r in prev_to_date_rows), 2)

        recent_transactions = self._repo.list(
            data_mode=data_mode, sort="-date", limit=_RECENT_TRANSACTIONS_LIMIT
        )

        return {
            "period": {"current": current_month, "previous": previous_month},
            "total_spend_current": total_spend_current,
            "total_spend_previous": total_spend_previous,
            "total_spend_previous_to_date": total_spend_previous_to_date,
            "comparable_day": window.comparable_day,
            "change_pct": _change_pct(total_spend_current, total_spend_previous_to_date),
            "category_breakdown": _category_breakdown(current_month_rows, total_spend_current),
            "spending_trend": _spending_trend(today, monthly_totals),
            "recent_transactions": recent_transactions,
            "forecast_summary": None,
            "portfolio_summary": None,
            "data_mode": app_mode,
        }
