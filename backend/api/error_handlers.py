"""
Exception handlers mapping domain exceptions (and FastAPI's own validation/
HTTP exceptions) onto the TRD §15 error envelope and status codes.

Registered onto the FastAPI app via register_error_handlers(app).
"""
import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from backend.api.errors import AppError

logger = logging.getLogger("backend")


def _envelope(error: str, message: str, details: dict | None = None) -> dict:
    return {"error": error, "message": message, "details": details or {}}


async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.error_code, exc.message, exc.details),
    )


def _json_safe_validation_errors(exc: RequestValidationError) -> list[dict]:
    # Pydantic v2 puts the raw exception object (e.g. a ValueError raised by
    # a @field_validator) under error["ctx"]["error"], which json.dumps
    # cannot serialize. Strip "ctx" (an implementation detail — the "msg"
    # field already contains that exception's message) rather than leaking
    # a raw Python exception repr to the client.
    errors = []
    for error in exc.errors():
        error = dict(error)
        error.pop("ctx", None)
        errors.append(error)
    return errors


async def _handle_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # FastAPI's own schema-validation errors are normalized into the same
    # envelope (TRD §15) rather than FastAPI's default {"detail": [...]}.
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_envelope(
            "validation_error",
            "Request validation failed.",
            {"errors": _json_safe_validation_errors(exc)},
        ),
    )


async def _handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    # Fallback for any plain HTTPException raised (or renamed via a library),
    # so the envelope stays consistent even outside our own AppError tree.
    error_code = {
        400: "bad_request",
        404: "not_found",
        409: "conflict",
        422: "validation_error",
        503: "service_unavailable",
    }.get(exc.status_code, "error")
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(error_code, detail),
    )


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # TRD §15: 500s return a generic message, never a stack trace to the client.
    logger.exception("Unhandled exception while processing %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_envelope("internal_error", "An unexpected error occurred."),
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(RequestValidationError, _handle_request_validation_error)
    app.add_exception_handler(HTTPException, _handle_http_exception)
    app.add_exception_handler(Exception, _handle_unexpected_error)
