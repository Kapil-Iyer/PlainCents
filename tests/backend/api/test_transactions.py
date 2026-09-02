"""Transaction CRUD API tests (Build Plan Phase 3, item 8): full CRUD via
HTTP, a 503 on missing-model manual-create, and effective-category
correctness after a correction."""
import sqlite3

from fastapi.testclient import TestClient

from backend.api.deps import get_categorization_service, get_db
from backend.main import app
from backend.services.categorization_service import CategorizationService

from .conftest import MISSING_MODEL_PATH


def _sample(**overrides):
    data = {"date": "2026-01-15", "merchant": "TIM HORTONS", "amount": 4.50}
    data.update(overrides)
    return data


def test_create_transaction_sets_predicted_category(client: TestClient):
    response = client.post("/api/transactions", json=_sample())

    assert response.status_code == 201
    body = response.json()
    assert body["predicted_category"]
    assert body["confirmed_category"] is None
    assert body["effective_category"] == body["predicted_category"]
    assert body["is_manual_override"] is False


def test_get_transaction_by_id(client: TestClient):
    created = client.post("/api/transactions", json=_sample()).json()

    response = client.get(f"/api/transactions/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_missing_transaction_returns_404_envelope(client: TestClient):
    response = client.get("/api/transactions/999999")

    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "not_found"


def test_list_transactions_returns_created_row(client: TestClient):
    client.post("/api/transactions", json=_sample())

    response = client.get("/api/transactions")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["page"] == 1


def test_list_transactions_empty_mode_returns_empty_list(client: TestClient):
    response = client.get("/api/transactions")

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "page": 1, "page_size": 50}


def test_patch_correction_updates_effective_category_not_predicted(client: TestClient):
    created = client.post("/api/transactions", json=_sample()).json()
    original_predicted = created["predicted_category"]

    response = client.patch(
        f"/api/transactions/{created['id']}", json={"confirmed_category": "Healthcare"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_category"] == original_predicted
    assert body["confirmed_category"] == "Healthcare"
    assert body["effective_category"] == "Healthcare"
    assert body["is_manual_override"] is True


def test_patch_missing_transaction_returns_404(client: TestClient):
    response = client.patch("/api/transactions/999999", json={"amount": 1.0})
    assert response.status_code == 404


def test_patch_empty_body_returns_400(client: TestClient):
    created = client.post("/api/transactions", json=_sample()).json()
    response = client.patch(f"/api/transactions/{created['id']}", json={})
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_delete_transaction(client: TestClient):
    created = client.post("/api/transactions", json=_sample()).json()

    response = client.delete(f"/api/transactions/{created['id']}")
    assert response.status_code == 200
    assert response.json() == {"id": created["id"], "deleted": True}

    follow_up = client.get(f"/api/transactions/{created['id']}")
    assert follow_up.status_code == 404


def test_delete_missing_transaction_returns_404(client: TestClient):
    response = client.delete("/api/transactions/999999")
    assert response.status_code == 404


def test_create_rejects_invalid_confirmed_category(client: TestClient):
    response = client.post("/api/transactions", json=_sample(confirmed_category="Not A Real Category"))
    assert response.status_code == 422


def test_create_manual_returns_503_when_model_missing(conn: sqlite3.Connection):
    missing_service = CategorizationService(MISSING_MODEL_PATH)
    app.dependency_overrides[get_db] = lambda: conn
    app.dependency_overrides[get_categorization_service] = lambda: missing_service
    try:
        test_client = TestClient(app)
        response = test_client.post("/api/transactions", json=_sample())
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_categorization_service, None)

    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "categorization_unavailable"

    # No row was written with the model unavailable.
    row = conn.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()
    assert row["n"] == 0
