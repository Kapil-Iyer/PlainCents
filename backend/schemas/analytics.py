"""Analytics schemas (ML-G). Every field is a live aggregation over stored
transactions; nothing here is fabricated or extrapolated."""
from __future__ import annotations

from pydantic import BaseModel


class CategoryTrendPoint(BaseModel):
    month: str
    total_spend: float
    by_category: dict[str, float]


class CategoryTrendResponse(BaseModel):
    """Monthly spend per effective category over a trailing window."""

    months: list[str]
    categories: list[str]
    points: list[CategoryTrendPoint]


class TopMerchantItem(BaseModel):
    merchant: str
    merchant_key: str | None
    total_spend: float
    transaction_count: int
    average_transaction: float
    category: str | None
    last_seen: str | None
    pct_of_total: float


class TopMerchantsResponse(BaseModel):
    items: list[TopMerchantItem]
    total_spend: float
    distinct_merchants: int
    top_n_share_pct: float
    months: int


class CategoryMover(BaseModel):
    category: str
    current: float
    previous: float
    change: float
    change_pct: float | None


class CategoryMoversResponse(BaseModel):
    """Per-category contributions to the month-over-month change in total
    spend. The `change` values sum exactly to `total_change`."""

    current_month: str
    previous_month: str
    total_current: float
    total_previous: float
    total_change: float
    movers: list[CategoryMover]


class SpendPacePoint(BaseModel):
    day: int
    # None past today (current month) or past the previous month's real
    # length -- a genuine gap, never a flat zero that would read as "spent
    # nothing that day".
    current_cumulative: float | None
    previous_cumulative: float | None


class SpendPaceResponse(BaseModel):
    current_month: str
    previous_month: str
    day_of_month: int
    current_to_date: float
    previous_same_point: float
    difference: float
    points: list[SpendPacePoint]


class ForecastAccuracyItem(BaseModel):
    forecast_month: str
    category: str
    predicted: float
    actual: float
    error: float
    generated_at: str | None


class ForecastAccuracyResponse(BaseModel):
    """Forecast-vs-actual, from GENUINE historical snapshots only.

    `available` is False (with a `reason`) until at least one forecast run
    exists that was generated before the month it predicted, for a month that
    has since completed. No prediction is ever recomputed after the fact and
    presented as if it had been made at the time.
    """

    available: bool
    reason: str | None
    items: list[ForecastAccuracyItem]
    months_covered: list[str]
    total_predicted: float
    total_actual: float
    wape: float | None
