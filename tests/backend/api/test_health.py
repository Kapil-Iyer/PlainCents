"""GET /api/health (Build Plan Phase 2, updated Phase 3 for the real
CategorizationService status)."""
import sqlite3

from fastapi.testclient import TestClient

from backend.api.deps import get_categorization_service
from backend.main import app
from backend.services.categorization_service import CategorizationService

from .conftest import MISSING_MODEL_PATH


def test_health_returns_expected_shape(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "db": "ok",
        "categorization_model": "loaded",
        "data_mode": "EMPTY",
    }


def test_health_has_no_extra_fields(client):
    # TRD §5.1: no row counts/paths/stack traces leaked.
    response = client.get("/api/health")
    body = response.json()
    assert set(body.keys()) == {"db", "categorization_model", "data_mode"}


def test_health_reports_missing_model(conn: sqlite3.Connection):
    from backend.api.deps import get_db

    missing_service = CategorizationService(MISSING_MODEL_PATH)
    app.dependency_overrides[get_db] = lambda: conn
    app.dependency_overrides[get_categorization_service] = lambda: missing_service
    try:
        # Not a context manager: avoids running the lifespan hook, which
        # would open the real default plaincents_v2.db (see api/conftest.py).
        test_client = TestClient(app)
        response = test_client.get("/api/health")
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_categorization_service, None)

    assert response.status_code == 200
    assert response.json()["categorization_model"] == "missing"
