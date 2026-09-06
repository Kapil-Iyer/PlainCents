"""Shared response schemas (TRD §6, §15)."""
from typing import Any, Literal

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """TRD §15 error envelope: {"error": "stable_snake_case_code", "message": "...", "details": {}}."""

    error: str
    message: str
    details: dict[str, Any] = {}


class HealthResponse(BaseModel):
    """TRD §5.1 — GET /api/health."""

    db: Literal["ok", "error"]
    categorization_model: Literal["loaded", "missing", "error"]
    data_mode: Literal["EMPTY", "DEMO", "REAL"]


class DemoStatusResponse(BaseModel):
    """TRD §5.2 — GET /api/demo/status."""

    mode: Literal["EMPTY", "DEMO", "REAL"]
    can_load_demo: bool


class DemoLoadResponse(BaseModel):
    """TRD §5.2 — POST /api/demo/load: '200 + summary on success'."""

    mode: Literal["DEMO"]
    summary: dict[str, int]


class DemoClearResponse(BaseModel):
    """TRD §5.2 — DELETE /api/demo/clear: '200 on success (idempotent: 200
    even if already empty)'. `mode` reflects the mode after clearing — 'EMPTY'
    in the documented/expected case; see DemoService.clear_demo()'s REAL-mode
    defense-in-depth note for the one case it can differ."""

    mode: Literal["EMPTY", "REAL"]
    cleared: bool
    summary: dict[str, int]


class RealDataClearResponse(BaseModel):
    """DELETE /api/demo/clear-real-data: the mirror image of
    DemoClearResponse -- in-app equivalent of scripts/reset_real_data.py,
    for a user who doesn't have shell access to the running instance (e.g.
    a deployed environment). 200 on success, idempotent even if there was no
    real data. `mode` reflects the mode after clearing -- 'EMPTY' in the
    documented/expected case; see DemoService.clear_real_data()'s DEMO-mode
    defense-in-depth note for the one case it can differ."""

    mode: Literal["EMPTY", "DEMO"]
    cleared: bool
    summary: dict[str, int]
