"""Dashboard schemas (TRD §6 DashboardSummary, §5.8; Build Plan Phase 6)."""
from typing import Literal

from pydantic import BaseModel

from backend.schemas.transaction import TransactionResponse


class DashboardPeriod(BaseModel):
    """Calendar-month identifiers, 'YYYY-MM' (PRD §11.7: the dashboard's
    default period is the current calendar month vs. the previous calendar
    month — not a rolling 30-day window)."""

    current: str
    previous: str


class CategoryBreakdownItem(BaseModel):
    """One effective-category's share of the current month's spend."""

    category: str
    total_spend: float
    pct_of_total: float


class SpendingTrendPoint(BaseModel):
    """One month's total spend in the trailing trend window.

    `has_data=False` (`total_spend=None`) is the one exception: the current,
    still-in-progress calendar month when NOTHING has been imported for it
    yet. That is a genuine gap ("nothing imported"), not a computed $0, and
    the frontend must render it as a gap/no-data marker rather than drawing
    the line down to zero -- see DashboardService._spending_trend. Every
    other point (including a fully-completed historical month with genuinely
    $0 spend) keeps `has_data=True` with its real computed total."""

    month: str
    total_spend: float | None
    has_data: bool = True


class DashboardSummaryResponse(BaseModel):
    """TRD §6 DashboardSummary.

    `forecast_summary`/`portfolio_summary` are always None in Phase 6 — their
    backing services (ForecastService.run_forecast, PortfolioService) don't
    exist until Phases 7/8 (Build Plan Phase 6, item 12: "no placeholder fake
    data"). They stay in the schema now so those phases only need to start
    populating a field that already exists, not add one.
    """

    period: DashboardPeriod
    # True when `period.current` is still in progress (the analysis month
    # equals today's own calendar month) -- False for a fully-completed
    # historical month the user selected. Drives whether the frontend labels
    # this "day 1 through today" (MTD-aligned) or a full calendar-month
    # comparison -- see backend.services.date_windows.analysis_window.
    is_current_incomplete: bool
    # Current-month-default fix: True when `period.current` actually has at
    # least one imported transaction (of either transaction_type). False
    # means the analysis month was EXPLICITLY selected (including the true
    # current calendar month, by a user or by resolve_default_analysis_month
    # finding no other populated month to fall back to) despite having no
    # data at all -- the frontend must show honest "no transactions imported
    # for {month} yet" copy in that case, never "$0 spent" / a pace
    # comparison. See DashboardService.get_summary's own docstring.
    current_month_has_data: bool = True
    total_spend_current: float
    total_spend_previous: float
    # Previous month's spend, capped at the SAME day-of-month the current
    # (possibly partial) month has reached -- the fair basis `change_pct` is
    # computed against. `total_spend_previous` above stays the full previous
    # calendar month, a separate and still-honest standalone figure.
    total_spend_previous_to_date: float
    comparable_day: int
    change_pct: float | None
    category_breakdown: list[CategoryBreakdownItem]
    spending_trend: list[SpendingTrendPoint]
    recent_transactions: list[TransactionResponse]
    forecast_summary: dict | None = None
    portfolio_summary: dict | None = None
    data_mode: Literal["EMPTY", "DEMO", "REAL"]


class AvailableMonthsResponse(BaseModel):
    """Backs the ONE shared analysis-month selector (Change KPI, Spending
    Pace, Category Movers) -- only months a user actually has data in,
    newest first, never an arbitrary/empty calendar picker."""

    months: list[str]
