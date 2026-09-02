"""GET /api/health (TRD §5.1). categorization_model reports the real
CategorizationService status (Phase 3); db status reflects whether the
shared connection can execute a trivial query."""
import sqlite3

from fastapi import APIRouter, Depends

from backend.api.deps import get_categorization_service, get_db
from backend.schemas.common import HealthResponse
from backend.services.app_state_service import AppStateService
from backend.services.categorization_service import CategorizationService

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
def get_health(
    conn: sqlite3.Connection = Depends(get_db),
    categorization_service: CategorizationService = Depends(get_categorization_service),
) -> HealthResponse:
    try:
        conn.execute("SELECT 1")
        db_status = "ok"
        data_mode = AppStateService(conn).get_mode()
    except sqlite3.Error:
        db_status = "error"
        data_mode = "EMPTY"

    return HealthResponse(
        db=db_status,
        categorization_model=categorization_service.status,
        data_mode=data_mode,
    )
