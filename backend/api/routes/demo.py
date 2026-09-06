"""
Demo endpoints (TRD §5.2, §14; Build Plan Phase 9).

GET /api/demo/status reads app_state.mode directly (no DemoService needed —
it's a pure read, same as before Phase 9). POST /api/demo/load and
DELETE /api/demo/clear now delegate to the real DemoService: load seeds a
full deterministic demo dataset and flips EMPTY -> DEMO (409 if not EMPTY);
clear deletes every demo-flagged row and flips back to EMPTY (200,
idempotent even if already empty).
"""
import sqlite3

from fastapi import APIRouter, Depends

from backend.api.deps import get_db
from backend.schemas.common import (
    DemoClearResponse,
    DemoLoadResponse,
    DemoStatusResponse,
    RealDataClearResponse,
)
from backend.services.app_state_service import AppStateService
from backend.services.demo_service import DemoService

router = APIRouter()


@router.get("/api/demo/status", response_model=DemoStatusResponse)
def get_demo_status(conn: sqlite3.Connection = Depends(get_db)) -> DemoStatusResponse:
    service = AppStateService(conn)
    mode = service.get_mode()
    return DemoStatusResponse(mode=mode, can_load_demo=(mode == "EMPTY"))


@router.post("/api/demo/load", response_model=DemoLoadResponse)
def load_demo(conn: sqlite3.Connection = Depends(get_db)) -> DemoLoadResponse:
    service = DemoService(conn)
    result = service.load_demo()
    return DemoLoadResponse(**result)


@router.delete("/api/demo/clear", response_model=DemoClearResponse)
def clear_demo(conn: sqlite3.Connection = Depends(get_db)) -> DemoClearResponse:
    service = DemoService(conn)
    result = service.clear_demo()
    return DemoClearResponse(**result)


@router.delete("/api/demo/clear-real-data", response_model=RealDataClearResponse)
def clear_real_data(conn: sqlite3.Connection = Depends(get_db)) -> RealDataClearResponse:
    """In-app, user-facing equivalent of scripts/reset_real_data.py: deletes
    every data_mode='real' row and flips app_state.mode back to 'EMPTY', so
    Load Demo Data becomes available -- without needing shell access to the
    running instance (e.g. once deployed). Namespaced under /api/demo/
    because it exists to unblock the demo/real mutual-exclusion state
    machine DemoService already owns, the mirror image of /api/demo/clear."""
    service = DemoService(conn)
    result = service.clear_real_data()
    return RealDataClearResponse(**result)
