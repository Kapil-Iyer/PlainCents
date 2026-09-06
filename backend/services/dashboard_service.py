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
from backend.services.date_windows import analysis_window, shift_month

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


def _spending_trend(anchor_year: int, anchor_month: int, monthly_totals: dict[str, float]) -> list[dict]:
    """Trailing `_TREND_MONTHS` months ENDING at the analysis month (the
    selected month, which defaults to the current calendar month), zero-filled
    for months with no transactions. This is real, computed information (a
    month genuinely had $0 spend), not fabricated data.

    Always full monthly totals -- historical context, deliberately NOT
    MTD-aligned, regardless of whether the anchor month is still in
    progress (see module docstring / date_windows.analysis_window). When
    the anchor month IS the current in-progress month, its own point here
    is the same as every other month's: a plain sum of whatever rows exist
    for it (never fabricated) -- honest by construction as long as no
    future-dated rows exist (see demo_seed_data.py's own day-of-month cap
    for the current month)."""
    points = []
    for offset in range(_TREND_MONTHS - 1, -1, -1):
        year, month = _shift_month(anchor_year, anchor_month, -offset)
        m = f"{year:04d}-{month:02d}"
        points.append({"month": m, "total_spend": round(monthly_totals.get(m, 0.0), 2)})
    return points


class DashboardService:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._repo = TransactionRepository(conn)

    def list_available_months(self, data_mode: str | None) -> list[str]:
        """Backs the analysis-month selector -- only months actually
        represented in the data, never an arbitrary calendar pick."""
        return self._repo.list_distinct_months(data_mode=data_mode)

    def get_summary(
        self,
        data_mode: str | None,
        app_mode: str,
        reference_date: date | None = None,
        analysis_month: str | None = None,
    ) -> dict:
        """
        `data_mode` is the repository-level filter ('real'/'demo'/None for
        EMPTY, from resolve_data_mode_filter — computed by the route, same as
        transactions.py) that decides which rows are read. `app_mode` is the
        EMPTY/DEMO/REAL app state, echoed back verbatim as the response's
        `data_mode` field (TRD §6). `reference_date` defaults to today; tests
        inject a fixed date to exercise month-boundary edge cases.

        `analysis_month` ("YYYY-MM") is the ONE shared clock driving this
        card, Spending Pace, and Category Movers together (product decision:
        one selector, not one per card) -- defaults to `reference_date`'s own
        month, reproducing prior behavior exactly. See
        backend.services.date_windows.analysis_window for the two resulting
        regimes: current-incomplete-month (day-aligned MTD vs MTD) or a
        fully-completed historical month (full month vs full month).

        BUG FIX (previously `total_spend_current` was NOT date-capped —
        see date_windows.py's module docstring): both `total_spend_current`
        and `category_breakdown` are now queried through `window.current_end`
        (today, when the analysis month is still in progress; the month's
        own last day otherwise), so the numerator `change_pct` divides by is
        symmetric with the (already-capped) `total_spend_previous_to_date`
        denominator, and with what Spending Pace / Category Movers show for
        the same analysis month.
        """
        today = reference_date or date.today()
        window = analysis_window(today, analysis_month)
        current_month = window.selected_month
        previous_month = window.previous_month

        trend_end_year, trend_end_month = int(current_month[:4]), int(current_month[5:7])
        trend_start_year, trend_start_month = _shift_month(
            trend_end_year, trend_end_month, -(_TREND_MONTHS - 1)
        )
        trend_start = f"{trend_start_year:04d}-{trend_start_month:02d}-01"

        # Spending Trend: full monthly totals, historical context, never
        # capped at `window.current_end` -- deliberately a SEPARATE query
        # from the current-period figures below (see _spending_trend's own
        # docstring for why this must stay uncapped).
        trend_rows = self._repo.aggregate_by_month_category(data_mode=data_mode, date_from=trend_start)
        trend_rows = [r for r in trend_rows if r["month"] <= current_month]
        monthly_totals: dict[str, float] = {}
        for row in trend_rows:
            monthly_totals[row["month"]] = monthly_totals.get(row["month"], 0.0) + row["total_spend"]
        total_spend_previous = round(monthly_totals.get(previous_month, 0.0), 2)

        # The CURRENT period's own figures -- queried through `current_end`
        # (this is the actual bug fix: previously sourced from the same
        # unbounded `monthly_totals` dict as the trend chart above, so an
        # in-progress month's total silently included every row dated later
        # in that month, however far in the future).
        current_rows = self._repo.aggregate_by_month_category(
            data_mode=data_mode, date_from=window.current_start, date_to=window.current_end,
        )
        total_spend_current = round(sum(r["total_spend"] for r in current_rows), 2)

        # The FAIR comparison basis: the previous period's spend through the
        # SAME relative point the current (possibly partial) period has
        # reached -- day-aligned for an in-progress month, full-month for a
        # completed one (window.previous_end already encodes which). Without
        # this, a $0 first day of the month reads as "-100% vs last month"
        # against the full previous month's total, which is not a meaningful
        # statement about pace. `total_spend_previous` above stays the full
        # previous CALENDAR month regardless, for its own separate, honest
        # standalone meaning ("you spent $X last month, period").
        prev_to_date_rows = self._repo.aggregate_by_month_category(
            data_mode=data_mode, date_from=window.previous_start, date_to=window.previous_end,
        )
        total_spend_previous_to_date = round(sum(r["total_spend"] for r in prev_to_date_rows), 2)

        recent_transactions = self._repo.list(
            data_mode=data_mode, sort="-date", limit=_RECENT_TRANSACTIONS_LIMIT
        )

        return {
            "period": {"current": current_month, "previous": previous_month},
            "is_current_incomplete": window.is_current_incomplete,
            "total_spend_current": total_spend_current,
            "total_spend_previous": total_spend_previous,
            "total_spend_previous_to_date": total_spend_previous_to_date,
            "comparable_day": window.comparable_day,
            "change_pct": _change_pct(total_spend_current, total_spend_previous_to_date),
            "category_breakdown": _category_breakdown(current_rows, total_spend_current),
            "spending_trend": _spending_trend(trend_end_year, trend_end_month, monthly_totals),
            "recent_transactions": recent_transactions,
            "forecast_summary": None,
            "portfolio_summary": None,
            "data_mode": app_mode,
        }
