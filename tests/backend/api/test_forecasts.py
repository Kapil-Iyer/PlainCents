"""
Forecast API tests (Build Plan Phase 7, item 8; TRD Section 17.3): cold-start
(200 for status, 422 for an explicit run attempt), run creation/retention,
the staleness decision table's real hook (extending Phase 3's stubbed
version), and the call-count assertion proving GET /status and GET /latest
never trigger a fit.

Transactions are seeded directly via TransactionRepository (forecast_fixtures,
deterministic, no CSV) rather than through the real POST /api/transactions
endpoint for speed across many rows — since that bypasses
TransactionService's own EMPTY->REAL transition, app_state.mode is set
explicitly via AppStateRepository afterward so the routes' mode resolution
(AppStateService.get_mode() -> resolve_data_mode_filter()) sees 'REAL', not
the default 'EMPTY'.
"""
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.repositories.app_state_repository import AppStateRepository

from tests.backend.services.forecast_fixtures import seed_months

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "td_csv"


def _seed_real(conn, months, categories):
    seed_months(conn, months, categories)
    AppStateRepository(conn).set_mode("REAL")
    conn.commit()


# -- status: cold start is always 200 ------------------------------------------


def test_status_cold_start_on_fresh_database(client: TestClient):
    response = client.get("/api/forecasts/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "cold_start"
    assert body["months_available"] == 0
    # ML-G: three completed months, the 3-month rolling mean's window.
    assert body["months_required"] == 3
    assert body["latest_run_id"] is None


def test_status_cold_start_with_partial_history(client: TestClient, conn):
    _seed_real(conn, 2, ["Food & Dining"])

    response = client.get("/api/forecasts/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "cold_start"
    assert body["months_available"] == 2


def test_status_no_forecast_yet_when_eligible_but_never_generated(client: TestClient, conn):
    _seed_real(conn, 12, ["Food & Dining"])

    response = client.get("/api/forecasts/status")

    assert response.status_code == 200
    assert response.json()["status"] == "no_forecast_yet"


# -- run: 422 during cold start, never 500/silent -------------------------------


def test_run_forecast_rejected_with_422_during_cold_start(client: TestClient, conn):
    _seed_real(conn, 2, ["Food & Dining"])

    response = client.post("/api/forecasts/run")

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "cold_start"
    assert "months_available" in body["details"]

    # Nothing was persisted.
    assert client.get("/api/forecasts/status").json()["status"] == "cold_start"


# -- run: creation, latest/status reflect it, retention -------------------------


def test_run_forecast_creates_a_run_reflected_by_status_and_latest(client: TestClient, conn):
    _seed_real(conn, 12, ["Food & Dining"])

    run_response = client.post("/api/forecasts/run")
    assert run_response.status_code == 200
    run_body = run_response.json()
    assert run_body["is_stale"] is False
    assert len(run_body["predictions"]) > 0

    status_body = client.get("/api/forecasts/status").json()
    assert status_body["status"] == "ready"
    assert status_body["latest_run_id"] == run_body["run_id"]
    assert status_body["is_stale"] is False

    latest_body = client.get("/api/forecasts/latest").json()
    assert latest_body["run_id"] == run_body["run_id"]
    assert latest_body["predictions"] == run_body["predictions"]


def test_latest_returns_no_forecast_yet_shape_when_none_exists(client: TestClient, conn):
    _seed_real(conn, 12, ["Food & Dining"])

    response = client.get("/api/forecasts/latest")

    assert response.status_code == 200
    assert response.json() == {"status": "no_forecast_yet"}


def test_run_retention_two_consecutive_runs_are_both_queryable(client: TestClient, conn):
    _seed_real(conn, 12, ["Food & Dining"])

    first = client.post("/api/forecasts/run").json()
    second = client.post("/api/forecasts/run").json()

    assert first["run_id"] != second["run_id"]
    assert client.get("/api/forecasts/latest").json()["run_id"] == second["run_id"]


# -- reads never fit -------------------------------------------------------------


def test_status_and_latest_never_trigger_a_fit(client: TestClient, conn):
    _seed_real(conn, 12, ["Food & Dining"])
    client.post("/api/forecasts/run")  # one legitimate fit, so there's a run to read

    with patch("backend.services.forecast_service.train_and_predict") as mock_fit:
        client.get("/api/forecasts/status")
        client.get("/api/forecasts/latest")

    mock_fit.assert_not_called()


# -- staleness: the real hook (TRD Section 7.2/Section 12.4) --------------------


def test_amount_edit_marks_the_forecast_stale(client: TestClient, conn):
    _seed_real(conn, 12, ["Food & Dining"])
    client.post("/api/forecasts/run")

    txn = client.get("/api/transactions?page_size=1").json()["items"][0]
    patch_response = client.patch(f"/api/transactions/{txn['id']}", json={"amount": txn["amount"] + 1})
    assert patch_response.status_code == 200

    assert client.get("/api/forecasts/status").json()["is_stale"] is True


def test_merchant_only_edit_does_not_mark_the_forecast_stale(client: TestClient, conn):
    _seed_real(conn, 12, ["Food & Dining"])
    client.post("/api/forecasts/run")

    txn = client.get("/api/transactions?page_size=1").json()["items"][0]
    client.patch(f"/api/transactions/{txn['id']}", json={"merchant": "RENAMED MERCHANT"})

    assert client.get("/api/forecasts/status").json()["is_stale"] is False


def test_transaction_delete_marks_the_forecast_stale(client: TestClient, conn):
    # Two categories per month so deleting one row doesn't also drop
    # months_available below 12 (which would flip status to cold_start and
    # make is_stale meaningless) — this test is specifically about the
    # staleness hook firing, not about the cold-start boundary.
    _seed_real(conn, 12, ["Food & Dining", "Transport"])
    client.post("/api/forecasts/run")

    txn = client.get("/api/transactions?page_size=1").json()["items"][0]
    client.delete(f"/api/transactions/{txn['id']}")

    assert client.get("/api/forecasts/status").json()["is_stale"] is True


def test_import_confirm_marks_the_forecast_stale(client: TestClient, conn):
    _seed_real(conn, 12, ["Food & Dining"])
    client.post("/api/forecasts/run")

    file_bytes = (FIXTURES_DIR / "clean_valid.csv").read_bytes()
    preview = client.post(
        "/api/imports",
        files={"file": ("clean_valid.csv", file_bytes, "text/csv")},
        data={"bank": "TD"},
    ).json()
    confirm_response = client.post(f"/api/imports/{preview['batch_id']}/confirm")
    assert confirm_response.status_code == 200

    assert client.get("/api/forecasts/status").json()["is_stale"] is True
