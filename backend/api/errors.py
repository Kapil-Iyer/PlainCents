"""
Domain exception hierarchy (TRD §15).

Routes/services raise these instead of FastAPI's HTTPException so that every
error — regardless of where it originates — is normalized into the same
{"error", "message", "details"} envelope by error_handlers.py.
"""
from typing import Any


class AppError(Exception):
    """Base domain exception. status_code/error_code drive the TRD §15 envelope."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class BadRequestError(AppError):
    status_code = 400
    error_code = "bad_request"


class NotFoundError(AppError):
    status_code = 404
    error_code = "not_found"


class ConflictError(AppError):
    status_code = 409
    error_code = "conflict"


class ValidationError(AppError):
    status_code = 422
    error_code = "validation_error"


class DemoConflictError(ConflictError):
    """TRD §5.2/§5.3: a real import attempted while app_state.mode == 'DEMO'.
    Distinct error_code (not the generic "conflict") so the frontend can
    recognize this specific case and offer the confirm-clear-demo flow."""

    error_code = "demo_conflict"


class ForecastColdStartError(ValidationError):
    """TRD Section 5.6/Section 15: POST /api/forecasts/run attempted while
    months_available < 12 (the frozen cold-start threshold, TRD Section
    12.5). Distinct error_code from the generic "validation_error" so the
    frontend can recognize this specific case and render the cold-start
    explanation rather than a generic form-validation message. Checking
    status (GET /api/forecasts/status) during cold-start is NOT an error —
    it returns 200 with a structured payload; only an explicit attempt to
    *generate* while ineligible is rejected."""

    error_code = "cold_start"


class ServiceUnavailableError(AppError):
    status_code = 503
    error_code = "service_unavailable"


class CategorizationUnavailableError(ServiceUnavailableError):
    """TRD §11.3: the categorization model is missing/errored at a
    prediction-dependent write. No transaction row may ever be written with
    predicted_category=NULL, so this must be raised before any insert."""

    error_code = "categorization_unavailable"


class InternalError(AppError):
    status_code = 500
    error_code = "internal_error"
