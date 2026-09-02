"""Transaction CRUD routes (TRD §5.4, Build Plan Phase 3)."""
import sqlite3

from fastapi import APIRouter, Depends, Query

from backend.api.deps import get_categorization_service, get_db
from backend.repositories.mode_filter import resolve_data_mode_filter
from backend.schemas.transaction import (
    TransactionCreate,
    TransactionListResponse,
    TransactionResponse,
    TransactionUpdate,
)
from backend.services.app_state_service import AppStateService
from backend.services.categorization_service import CategorizationService
from backend.services.transaction_service import TransactionService

router = APIRouter()


def _service(
    conn: sqlite3.Connection, categorization_service: CategorizationService
) -> TransactionService:
    return TransactionService(conn, categorization_service)


@router.get("/api/transactions", response_model=TransactionListResponse)
def list_transactions(
    date_from: str | None = None,
    date_to: str | None = None,
    category: str | None = None,
    search: str | None = None,
    sort: str = "date",
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_db),
    categorization_service: CategorizationService = Depends(get_categorization_service),
) -> TransactionListResponse:
    mode = AppStateService(conn).get_mode()
    data_mode = resolve_data_mode_filter(mode)
    service = _service(conn, categorization_service)
    result = service.list(
        data_mode=data_mode,
        date_from=date_from,
        date_to=date_to,
        category=category,
        search=search,
        sort=sort,
        page=page,
        page_size=page_size,
    )
    return TransactionListResponse(**result)


@router.post("/api/transactions", response_model=TransactionResponse, status_code=201)
def create_transaction(
    payload: TransactionCreate,
    conn: sqlite3.Connection = Depends(get_db),
    categorization_service: CategorizationService = Depends(get_categorization_service),
) -> TransactionResponse:
    service = _service(conn, categorization_service)
    row = service.create_manual(payload.model_dump())
    return TransactionResponse(**row)


@router.get("/api/transactions/{transaction_id}", response_model=TransactionResponse)
def get_transaction(
    transaction_id: int,
    conn: sqlite3.Connection = Depends(get_db),
    categorization_service: CategorizationService = Depends(get_categorization_service),
) -> TransactionResponse:
    service = _service(conn, categorization_service)
    return TransactionResponse(**service.get(transaction_id))


@router.patch("/api/transactions/{transaction_id}", response_model=TransactionResponse)
def update_transaction(
    transaction_id: int,
    payload: TransactionUpdate,
    conn: sqlite3.Connection = Depends(get_db),
    categorization_service: CategorizationService = Depends(get_categorization_service),
) -> TransactionResponse:
    service = _service(conn, categorization_service)
    fields = payload.model_dump(exclude_unset=True)
    row = service.update(transaction_id, fields)
    return TransactionResponse(**row)


@router.delete("/api/transactions/{transaction_id}")
def delete_transaction(
    transaction_id: int,
    conn: sqlite3.Connection = Depends(get_db),
    categorization_service: CategorizationService = Depends(get_categorization_service),
) -> dict:
    service = _service(conn, categorization_service)
    service.delete(transaction_id)
    return {"id": transaction_id, "deleted": True}
