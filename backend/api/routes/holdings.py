"""Holdings CRUD + manual price refresh routes (TRD §5.7, Build Plan Phase 8)."""
import sqlite3

from fastapi import APIRouter, Depends

from backend.api.deps import get_db
from backend.repositories.mode_filter import resolve_data_mode_filter
from backend.schemas.holding import (
    HoldingCreate,
    HoldingResponse,
    HoldingUpdate,
    RefreshPricesResponse,
)
from backend.services.app_state_service import AppStateService
from backend.services.portfolio_service import PortfolioService

router = APIRouter()


@router.get("/api/holdings", response_model=list[HoldingResponse])
def list_holdings(conn: sqlite3.Connection = Depends(get_db)) -> list[HoldingResponse]:
    # TRD §13.2: DB/cache only, never fetch_price/yfinance.
    mode = AppStateService(conn).get_mode()
    data_mode = resolve_data_mode_filter(mode)
    service = PortfolioService(conn)
    return [HoldingResponse(**row) for row in service.get_holdings_with_prices(data_mode)]


@router.post("/api/holdings", response_model=HoldingResponse, status_code=201)
def create_holding(
    payload: HoldingCreate, conn: sqlite3.Connection = Depends(get_db)
) -> HoldingResponse:
    service = PortfolioService(conn)
    row = service.create_holding(payload.model_dump())
    return HoldingResponse(**row)


@router.patch("/api/holdings/{holding_id}", response_model=HoldingResponse)
def update_holding(
    holding_id: int, payload: HoldingUpdate, conn: sqlite3.Connection = Depends(get_db)
) -> HoldingResponse:
    service = PortfolioService(conn)
    fields = payload.model_dump(exclude_unset=True)
    row = service.update_holding(holding_id, fields)
    return HoldingResponse(**row)


@router.delete("/api/holdings/{holding_id}")
def delete_holding(holding_id: int, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    service = PortfolioService(conn)
    service.delete_holding(holding_id)
    return {"id": holding_id, "deleted": True}


@router.post("/api/holdings/refresh-prices", response_model=RefreshPricesResponse)
def refresh_prices(conn: sqlite3.Connection = Depends(get_db)) -> RefreshPricesResponse:
    # TRD §5.7/§13.2: the only endpoint permitted to call fetch_price/yfinance.
    mode = AppStateService(conn).get_mode()
    data_mode = resolve_data_mode_filter(mode)
    service = PortfolioService(conn)
    result = service.refresh_prices(data_mode)
    return RefreshPricesResponse(**result)
