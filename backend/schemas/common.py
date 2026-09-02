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
