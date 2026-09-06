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
    # PRODUCT DECISION: Ticker and Shares are required; average cost is not.
    # A user who knows "I own 10 MSFT shares" but not their exact cost basis
    # must still be able to add the holding -- current price and market
    # value are still honest and computable from shares alone (see
    # PortfolioService._to_response). Omitting avg_cost (or sending it as
    # null) is a real, deliberate "unknown" state, never coerced to 0.
    avg_cost: float | None = Field(default=None, ge=0)

    _validate_ticker = field_validator("ticker")(_validate_ticker)


class HoldingUpdate(BaseModel):
    # Ticker is deliberately absent: HoldingRepository.update() only ever
    # applies {"shares", "avg_cost"} (TRD §13.1 plain CRUD via
    # HoldingRepository as already implemented) — a ticker change is not a
    # supported update, so the schema doesn't expose a field that would be
    # silently dropped by the repository.
    shares: float | None = Field(default=None, gt=0)
    # `None` here is ambiguous by design between "field omitted" (the route
    # uses `model_dump(exclude_unset=True)`, so an omitted field never
    # reaches the repository at all) and "explicitly set to null" (a real
    # request to clear a previously-known cost basis) -- Pydantic v2 tracks
    # which fields were actually present in the request body separately
    # from their value, so both cases behave correctly with no extra code
    # here. ge=0 only constrains a non-null value; None always passes.
    avg_cost: float | None = Field(default=None, ge=0)


class HoldingResponse(BaseModel):
    id: int
    ticker: str
    shares: float
    # None means "cost basis not recorded yet" -- never fabricated from
    # current_price/a demo value/0. See PortfolioService._to_response for
    # how pnl derives from this (also None whenever avg_cost is None).
    avg_cost: float | None = None
    current_price: float | None = None
    current_value: float | None = None
    pnl: float | None = None
    price_last_updated: str | None = None
    # True only when `price_last_updated` is the fixed sentinel
    # DemoService.load_demo() stamps on a never-actually-fetched seeded
    # price (see PortfolioService._to_response) -- lets the frontend show
    # an honest "Demo snapshot" label instead of implying a real (if old)
    # fetch. False for every real holding and for a demo holding that has
    # since been refreshed with a genuine price.
    price_is_demo_snapshot: bool = False
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
