"""
Error-handler unit tests (Build Plan Phase 2, item 8): each TRD §15 mapped
status code, using deliberately-raised domain exceptions on throwaway routes,
plus FastAPI's own 422 schema-validation errors normalized into the same
envelope, and a deliberately-triggered 500 with no stack trace leaked.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from backend.api.error_handlers import register_error_handlers
from backend.api.errors import (
    BadRequestError,
    ConflictError,
    NotFoundError,
    ServiceUnavailableError,
    ValidationError,
)


@pytest.fixture
def error_test_app() -> TestClient:
    app = FastAPI()
    register_error_handlers(app)

    class Payload(BaseModel):
        name: str

    @app.get("/boom/bad-request")
    def bad_request():
        raise BadRequestError("bad input")

    @app.get("/boom/not-found")
    def not_found():
        raise NotFoundError("missing thing")

    @app.get("/boom/conflict")
    def conflict():
        raise ConflictError("already in that state")

    @app.get("/boom/validation")
    def validation():
        raise ValidationError("domain validation failed")

    @app.get("/boom/service-unavailable")
    def service_unavailable():
        raise ServiceUnavailableError("model unavailable")

    @app.get("/boom/unexpected")
    def unexpected():
        raise RuntimeError("something broke")

    @app.post("/boom/schema")
    def schema(payload: Payload):
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


ENVELOPE_KEYS = {"error", "message", "details"}


@pytest.mark.parametrize(
    "path,expected_status,expected_error",
    [
        ("/boom/bad-request", 400, "bad_request"),
        ("/boom/not-found", 404, "not_found"),
        ("/boom/conflict", 409, "conflict"),
        ("/boom/validation", 422, "validation_error"),
        ("/boom/service-unavailable", 503, "service_unavailable"),
    ],
)
def test_domain_exceptions_map_to_envelope(
    error_test_app: TestClient, path, expected_status, expected_error
):
    response = error_test_app.get(path)

    assert response.status_code == expected_status
    body = response.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["error"] == expected_error


def test_unexpected_exception_returns_generic_500(error_test_app: TestClient):
    response = error_test_app.get("/boom/unexpected")

    assert response.status_code == 500
    body = response.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["error"] == "internal_error"
    # No stack trace / exception details leaked to the client (TRD §15).
    assert "RuntimeError" not in body["message"]
    assert "Traceback" not in str(body)


def test_fastapi_schema_validation_error_uses_same_envelope(error_test_app: TestClient):
    response = error_test_app.post("/boom/schema", json={})

    assert response.status_code == 422
    body = response.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["error"] == "validation_error"
