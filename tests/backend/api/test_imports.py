"""
Import API tests (Build Plan Phase 4, item 8): upload -> preview -> confirm
via HTTP, dedup, idempotent confirm, model-missing preview-200/confirm-503,
and DEMO-mode 409.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.deps import get_categorization_service, get_db
from backend.main import app
from backend.repositories.app_state_repository import AppStateRepository
from backend.services.categorization_service import CategorizationService

from .conftest import MISSING_MODEL_PATH, TEST_MODEL_PATH

TD_CSV_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "td_csv"


def _upload(client: TestClient, filename: str, bank: str = "TD"):
    content = (TD_CSV_DIR / filename).read_bytes()
    return client.post(
        "/api/imports",
        files={"file": (filename, content, "text/csv")},
        data={"bank": bank},
    )


def test_upload_clean_valid_returns_200_preview(client: TestClient):
    response = _upload(client, "clean_valid.csv")

    assert response.status_code == 200
    body = response.json()
    assert body["rows_valid"] == 12
    assert body["rows_unparseable"] == 0
    assert body["rows_duplicate"] == 0
    assert body["categorization_available"] is True
    assert body["status"] == "previewing"
    assert len(body["sample_rows"]) == 10


def test_upload_unrecognized_format_returns_400(client: TestClient):
    response = _upload(client, "unrecognized_format.csv")
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_upload_headerless_td_fixture(client: TestClient):
    response = _upload(client, "headerless_positional.csv")
    assert response.status_code == 200
    body = response.json()
    assert body["rows_valid"] == 4
    assert body["rows_unparseable"] == 1


def test_confirm_persists_transactions_visible_via_list(client: TestClient):
    preview = _upload(client, "clean_valid.csv").json()

    confirm_response = client.post(f"/api/imports/{preview['batch_id']}/confirm")
    assert confirm_response.status_code == 200
    result = confirm_response.json()
    assert result["rows_imported"] == 12
    assert result["status"] == "confirmed"

    list_response = client.get("/api/transactions", params={"page_size": 50})
    body = list_response.json()
    assert body["total"] == 12
    assert all(item["predicted_category"] for item in body["items"])


def test_confirm_twice_is_idempotent(client: TestClient):
    preview = _upload(client, "clean_valid.csv").json()
    first = client.post(f"/api/imports/{preview['batch_id']}/confirm").json()
    second = client.post(f"/api/imports/{preview['batch_id']}/confirm").json()
    assert first == second

    list_response = client.get("/api/transactions", params={"page_size": 50})
    assert list_response.json()["total"] == 12


def test_reimport_same_file_flags_all_as_duplicate(client: TestClient):
    first_preview = _upload(client, "clean_valid.csv").json()
    client.post(f"/api/imports/{first_preview['batch_id']}/confirm")

    second_preview = _upload(client, "clean_valid.csv").json()
    assert second_preview["rows_duplicate"] == 12

    second_result = client.post(f"/api/imports/{second_preview['batch_id']}/confirm").json()
    assert second_result["rows_imported"] == 0
    assert second_result["rows_skipped_duplicate"] == 12


def test_preview_returns_200_with_categorization_unavailable_when_model_missing(
    conn,
):
    missing_service = CategorizationService(MISSING_MODEL_PATH)
    app.dependency_overrides[get_db] = lambda: conn
    app.dependency_overrides[get_categorization_service] = lambda: missing_service
    try:
        test_client = TestClient(app)
        response = _upload(test_client, "clean_valid.csv")
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_categorization_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["categorization_available"] is False
    assert all(r["predicted_category"] is None for r in body["sample_rows"])


def test_confirm_returns_503_and_commits_nothing_when_model_missing(conn):
    available_service = CategorizationService(TEST_MODEL_PATH)
    app.dependency_overrides[get_db] = lambda: conn
    app.dependency_overrides[get_categorization_service] = lambda: available_service
    try:
        staging_client = TestClient(app)
        preview = _upload(staging_client, "clean_valid.csv").json()
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_categorization_service, None)

    missing_service = CategorizationService(MISSING_MODEL_PATH)
    app.dependency_overrides[get_db] = lambda: conn
    app.dependency_overrides[get_categorization_service] = lambda: missing_service
    try:
        confirming_client = TestClient(app)
        response = confirming_client.post(f"/api/imports/{preview['batch_id']}/confirm")
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_categorization_service, None)

    assert response.status_code == 503
    assert response.json()["error"] == "categorization_unavailable"
    row = conn.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()
    assert row["n"] == 0


def test_import_returns_409_when_mode_is_demo(client: TestClient, conn):
    AppStateRepository(conn).set_mode("DEMO")
    response = _upload(client, "clean_valid.csv")
    assert response.status_code == 409
    assert response.json()["error"] == "demo_conflict"


def test_get_import_list_and_detail(client: TestClient):
    preview = _upload(client, "clean_valid.csv").json()

    list_response = client.get("/api/imports")
    assert list_response.status_code == 200
    batches = list_response.json()
    assert any(b["id"] == preview["batch_id"] for b in batches)

    detail_response = client.get(f"/api/imports/{preview['batch_id']}")
    assert detail_response.status_code == 200
    assert detail_response.json()["id"] == preview["batch_id"]
    assert detail_response.json()["status"] == "previewing"


def test_get_import_missing_batch_returns_404(client: TestClient):
    response = client.get("/api/imports/999999")
    assert response.status_code == 404
