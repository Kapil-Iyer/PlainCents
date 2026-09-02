"""Forecast schemas (TRD Section 6 ForecastStatusResponse/ForecastRunResponse/
ForecastPrediction, Section 5.6; Build Plan Phase 7)."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ForecastStatusResponse(BaseModel):
    """TRD Section 5.6: 200 always; cold_start is a normal read, never an
    error. latest_run_id/is_stale are only populated once status == 'ready'."""

    status: Literal["ready", "cold_start", "no_forecast_yet"]
    months_available: int
    months_required: int
    latest_run_id: int | None = None
    is_stale: bool | None = None


class ForecastPrediction(BaseModel):
    # is_available/predicted_amount arrive already Python-typed from
    # ForecastService (bool(...)/None coercion happens there), matching
    # TransactionResponse's convention of doing type coercion in the layer
    # that reads sqlite3.Row rather than duplicating it in every schema.
    category: str
    forecast_month: str
    month_offset: int
    predicted_amount: float | None
    is_available: bool
    unavailable_reason: str | None


class ForecastRunResponse(BaseModel):
    run_id: int
    generated_at: datetime
    is_stale: bool
    stale_reason: str | None
    months_available: int
    predictions: list[ForecastPrediction]
