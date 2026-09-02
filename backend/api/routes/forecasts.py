"""Forecast routes (TRD Section 5.6; Build Plan Phase 7).

GET /status and GET /latest are DB reads only (ForecastService never fits on
those paths). POST /run is the only endpoint that trains — 422 (via
ForecastColdStartError, handled by the shared AppError envelope) if
months_available < 12.
"""
import sqlite3

from fastapi import APIRouter, Depends

from backend.api.deps import get_db
from backend.repositories.mode_filter import resolve_data_mode_filter
from backend.schemas.forecast import ForecastRunResponse, ForecastStatusResponse
from backend.services.app_state_service import AppStateService
from backend.services.forecast_service import ForecastService

router = APIRouter()


def _data_mode(conn: sqlite3.Connection) -> str | None:
    # Same mode-resolution pattern as dashboard.py/transactions.py: the
    # route decides which data_mode is currently active and passes the
    # resulting filter into the service, rather than the service deciding
    # for itself.
    return resolve_data_mode_filter(AppStateService(conn).get_mode())


@router.get("/api/forecasts/status", response_model=ForecastStatusResponse)
def get_forecast_status(conn: sqlite3.Connection = Depends(get_db)) -> ForecastStatusResponse:
    service = ForecastService(conn)
    return ForecastStatusResponse(**service.check_status(_data_mode(conn)))


@router.get("/api/forecasts/latest")
def get_latest_forecast(conn: sqlite3.Connection = Depends(get_db)) -> dict:
    # TRD Section 5.6: 200 with ForecastRunResponse, or 200 with
    # {status: "no_forecast_yet"} if none exists — a union shape, so this
    # route returns a plain dict rather than a single Pydantic response_model.
    service = ForecastService(conn)
    latest = service.get_latest(_data_mode(conn))
    if latest is None:
        return {"status": "no_forecast_yet"}
    return ForecastRunResponse(**latest).model_dump(mode="json")


@router.post("/api/forecasts/run", response_model=ForecastRunResponse)
def run_forecast(conn: sqlite3.Connection = Depends(get_db)) -> ForecastRunResponse:
    service = ForecastService(conn)
    run = service.run_forecast(_data_mode(conn))
    return ForecastRunResponse(**run)
