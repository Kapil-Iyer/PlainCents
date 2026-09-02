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
    """One month's total spend in the trailing trend window."""

    month: str
    total_spend: float


class DashboardSummaryResponse(BaseModel):
    """TRD §6 DashboardSummary.

    `forecast_summary`/`portfolio_summary` are always None in Phase 6 — their
    backing services (ForecastService.run_forecast, PortfolioService) don't
    exist until Phases 7/8 (Build Plan Phase 6, item 12: "no placeholder fake
    data"). They stay in the schema now so those phases only need to start
    populating a field that already exists, not add one.
    """

    period: DashboardPeriod
    total_spend_current: float
    total_spend_previous: float
    change_pct: float | None
    category_breakdown: list[CategoryBreakdownItem]
    spending_trend: list[SpendingTrendPoint]
    recent_transactions: list[TransactionResponse]
    forecast_summary: dict | None = None
    portfolio_summary: dict | None = None
    data_mode: Literal["EMPTY", "DEMO", "REAL"]
