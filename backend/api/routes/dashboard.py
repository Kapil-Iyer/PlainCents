"""Dashboard route (TRD §5.8; Build Plan Phase 6)."""
import sqlite3

from fastapi import APIRouter, Depends

from backend.api.deps import get_db
from backend.repositories.mode_filter import resolve_data_mode_filter
from backend.schemas.dashboard import DashboardSummaryResponse
from backend.services.app_state_service import AppStateService
from backend.services.dashboard_service import DashboardService

router = APIRouter()


@router.get("/api/dashboard/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary(
    conn: sqlite3.Connection = Depends(get_db),
) -> DashboardSummaryResponse:
    # Same mode-resolution pattern as transactions.py: the route decides
    # which data_mode is currently active and passes the resulting filter
    # into the service, rather than the service deciding for itself.
    mode = AppStateService(conn).get_mode()
    data_mode = resolve_data_mode_filter(mode)
    service = DashboardService(conn)
    summary = service.get_summary(data_mode=data_mode, app_mode=mode)
    return DashboardSummaryResponse(**summary)
