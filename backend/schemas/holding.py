"""Holding schemas (TRD §6, §5.7, §13)."""
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


def _validate_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if not ticker:
        raise ValueError("ticker must not be blank")
    return ticker


class HoldingCreate(BaseModel):
    ticker: str
    shares: float = Field(gt=0)
    avg_cost: float = Field(ge=0)

    _validate_ticker = field_validator("ticker")(_validate_ticker)


class HoldingUpdate(BaseModel):
    # Ticker is deliberately absent: HoldingRepository.update() only ever
    # applies {"shares", "avg_cost"} (TRD §13.1 plain CRUD via
    # HoldingRepository as already implemented) — a ticker change is not a
    # supported update, so the schema doesn't expose a field that would be
    # silently dropped by the repository.
    shares: float | None = Field(default=None, gt=0)
    avg_cost: float | None = Field(default=None, ge=0)


class HoldingResponse(BaseModel):
    id: int
    ticker: str
    shares: float
    avg_cost: float
    current_price: float | None = None
    current_value: float | None = None
    pnl: float | None = None
    price_last_updated: str | None = None
    created_at: datetime
    updated_at: datetime


class RefreshedTicker(BaseModel):
    ticker: str
    price: float


class FailedTicker(BaseModel):
    ticker: str
    error: str


class RefreshPricesResponse(BaseModel):
    refreshed: list[RefreshedTicker]
    failed: list[FailedTicker]
