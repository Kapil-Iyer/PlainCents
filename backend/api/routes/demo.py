"""
Demo endpoints (TRD §5.2, Build Plan §2.5).

GET /api/demo/status is functional now — it reads the real app_state.mode.
POST /api/demo/load and DELETE /api/demo/clear are explicitly NOT implemented
until Phase 9: they return 501 so no later phase mistakes the stub for a
working demo-load/clear implementation.
"""
import sqlite3

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from backend.api.deps import get_db
from backend.schemas.common import DemoStatusResponse
from backend.services.app_state_service import AppStateService

router = APIRouter()


@router.get("/api/demo/status", response_model=DemoStatusResponse)
def get_demo_status(conn: sqlite3.Connection = Depends(get_db)) -> DemoStatusResponse:
    service = AppStateService(conn)
    mode = service.get_mode()
    return DemoStatusResponse(mode=mode, can_load_demo=(mode == "EMPTY"))


@router.post("/api/demo/load")
def load_demo() -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": "not_implemented",
            "message": "Demo load is not implemented until Phase 9.",
            "details": {},
        },
    )


@router.delete("/api/demo/clear")
def clear_demo() -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": "not_implemented",
            "message": "Demo clear is not implemented until Phase 9.",
            "details": {},
        },
    )
