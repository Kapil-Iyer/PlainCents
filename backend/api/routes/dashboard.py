"""Dashboard route (TRD §5.8; Build Plan Phase 6)."""
import sqlite3

from fastapi import APIRouter, Depends, Query

from backend.api.deps import get_db
from backend.repositories.mode_filter import resolve_data_mode_filter
from backend.schemas.dashboard import AvailableMonthsResponse, DashboardSummaryResponse
from backend.services.app_state_service import AppStateService
from backend.services.dashboard_service import DashboardService

router = APIRouter()


@router.get("/api/dashboard/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary(
    month: str | None = Query(
        default=None,
        description=(
            "'YYYY-MM' analysis month -- the ONE shared clock also driving "
            "Spending Pace and Category Movers. Defaults to the current "
            "calendar month (prior behavior, unchanged)."
        ),
    ),
    conn: sqlite3.Connection = Depends(get_db),
) -> DashboardSummaryResponse:
    # Same mode-resolution pattern as transactions.py: the route decides
    # which data_mode is currently active and passes the resulting filter
    # into the service, rather than the service deciding for itself.
    mode = AppStateService(conn).get_mode()
    data_mode = resolve_data_mode_filter(mode)
    service = DashboardService(conn)
    summary = service.get_summary(data_mode=data_mode, app_mode=mode, analysis_month=month)
    return DashboardSummaryResponse(**summary)


@router.get("/api/dashboard/available-months", response_model=AvailableMonthsResponse)
def get_available_months(
    conn: sqlite3.Connection = Depends(get_db),
) -> AvailableMonthsResponse:
    """Backs the analysis-month selector -- only months the active
    data_mode actually has transactions in, newest first."""
    mode = AppStateService(conn).get_mode()
    data_mode = resolve_data_mode_filter(mode)
    service = DashboardService(conn)
    return AvailableMonthsResponse(months=service.list_available_months(data_mode))
