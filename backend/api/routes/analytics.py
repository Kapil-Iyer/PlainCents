"""
Analytics routes (ML-G).

Same mode-resolution convention as dashboard.py and transactions.py: the
route resolves the app's EMPTY/DEMO/REAL state into a repository-level
data_mode filter once, and passes it into the service. The service never
decides for itself which data it is allowed to see.
"""
import sqlite3

from fastapi import APIRouter, Depends, Query

from backend.api.deps import get_db
from backend.repositories.mode_filter import resolve_data_mode_filter
from backend.schemas.analytics import (
    CategoryMoversResponse,
    CategoryTrendResponse,
    ForecastAccuracyResponse,
    SpendPaceResponse,
    TopMerchantsResponse,
)
from backend.services.analytics_service import (
    DEFAULT_TOP_MERCHANTS,
    DEFAULT_TREND_MONTHS,
    MAX_TOP_MERCHANTS,
    MAX_TREND_MONTHS,
    AnalyticsService,
)
from backend.services.app_state_service import AppStateService

router = APIRouter(prefix="/api/analytics")


def _service(conn: sqlite3.Connection) -> tuple[AnalyticsService, str | None]:
    mode = AppStateService(conn).get_mode()
    return AnalyticsService(conn), resolve_data_mode_filter(mode)


@router.get("/category-trend", response_model=CategoryTrendResponse)
def get_category_trend(
    months: int = Query(DEFAULT_TREND_MONTHS, ge=1, le=MAX_TREND_MONTHS),
    conn: sqlite3.Connection = Depends(get_db),
) -> CategoryTrendResponse:
    service, data_mode = _service(conn)
    return CategoryTrendResponse(**service.category_trend(data_mode, months=months))


@router.get("/top-merchants", response_model=TopMerchantsResponse)
def get_top_merchants(
    limit: int = Query(DEFAULT_TOP_MERCHANTS, ge=1, le=MAX_TOP_MERCHANTS),
    months: int = Query(DEFAULT_TREND_MONTHS, ge=1, le=MAX_TREND_MONTHS),
    conn: sqlite3.Connection = Depends(get_db),
) -> TopMerchantsResponse:
    service, data_mode = _service(conn)
    return TopMerchantsResponse(**service.top_merchants(data_mode, limit=limit, months=months))


_MONTH_QUERY = Query(
    default=None,
    description=(
        "'YYYY-MM' analysis month -- the ONE shared clock also driving the "
        "Dashboard's Change KPI and (via the sibling endpoint) Spend Pace / "
        "Category Movers together. Defaults to the current calendar month."
    ),
)


@router.get("/category-movers", response_model=CategoryMoversResponse)
def get_category_movers(
    month: str | None = _MONTH_QUERY,
    conn: sqlite3.Connection = Depends(get_db),
) -> CategoryMoversResponse:
    service, data_mode = _service(conn)
    return CategoryMoversResponse(**service.category_movers(data_mode, analysis_month=month))


@router.get("/spend-pace", response_model=SpendPaceResponse)
def get_spend_pace(
    month: str | None = _MONTH_QUERY,
    conn: sqlite3.Connection = Depends(get_db),
) -> SpendPaceResponse:
    service, data_mode = _service(conn)
    return SpendPaceResponse(**service.spend_pace(data_mode, analysis_month=month))


@router.get("/forecast-accuracy", response_model=ForecastAccuracyResponse)
def get_forecast_accuracy(
    conn: sqlite3.Connection = Depends(get_db),
) -> ForecastAccuracyResponse:
    service, data_mode = _service(conn)
    return ForecastAccuracyResponse(**service.forecast_accuracy(data_mode))
