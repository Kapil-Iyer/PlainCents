"""GET /api/demo/status, POST /api/demo/load, DELETE /api/demo/clear
(Build Plan Phase 2 status read; Phase 9 real load/clear + full demo->real
sequence, TRD §4.5, §5.2, §14)."""
from pathlib import Path

import sqlite3

from fastapi.testclient import TestClient

from backend.repositories.app_state_repository import AppStateRepository

TD_CSV_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "td_csv"


def _upload(client: TestClient, filename: str = "clean_valid.csv", bank: str = "TD"):
    content = (TD_CSV_DIR / filename).read_bytes()
    return client.post(
        "/api/imports",
        files={"file": (filename, content, "text/csv")},
        data={"bank": bank},
    )


def test_demo_status_returns_empty_on_fresh_db(client):
    response = client.get("/api/demo/status")

    assert response.status_code == 200
    assert response.json() == {"mode": "EMPTY", "can_load_demo": True}


def test_demo_status_reflects_real_mode(client, conn: sqlite3.Connection):
    AppStateRepository(conn).set_mode("REAL")
    conn.commit()

    response = client.get("/api/demo/status")

    assert response.status_code == 200
    assert response.json() == {"mode": "REAL", "can_load_demo": False}


# -- load ---------------------------------------------------------------


def test_demo_load_succeeds_from_empty(client: TestClient):
    response = client.post("/api/demo/load")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "DEMO"
    assert body["summary"]["transactions"] > 0
    assert body["summary"]["holdings"] > 0
    assert body["summary"]["forecast_predictions"] > 0

    status = client.get("/api/demo/status").json()
    assert status == {"mode": "DEMO", "can_load_demo": False}


def test_demo_load_populates_dashboard_forecast_and_holdings_immediately(client: TestClient):
    client.post("/api/demo/load")

    dashboard = client.get("/api/dashboard/summary").json()
    assert dashboard["data_mode"] == "DEMO"
    assert dashboard["total_spend_current"] >= 0  # populated, not the EMPTY shape's absence

    # Forecast is populated WITHOUT calling POST /api/forecasts/run.
    forecast_status = client.get("/api/forecasts/status").json()
    assert forecast_status["status"] == "ready"
    latest = client.get("/api/forecasts/latest").json()
    assert len(latest["predictions"]) == 24  # 8 categories x 3 months

    holdings = client.get("/api/holdings").json()
    assert len(holdings) > 0


def test_demo_load_rejected_when_real_data_exists(client: TestClient, conn: sqlite3.Connection):
    client.post("/api/transactions", json={"date": "2026-01-15", "merchant": "TIM HORTONS", "amount": 4.5})
    assert AppStateRepository(conn).get_mode() == "REAL"

    response = client.post("/api/demo/load")

    assert response.status_code == 409
    assert response.json()["error"] == "demo_conflict"
    # No real data was touched or deleted.
    assert len(client.get("/api/transactions").json()["items"]) == 1
    assert AppStateRepository(conn).get_mode() == "REAL"


def test_demo_load_rejected_when_already_demo(client: TestClient):
    client.post("/api/demo/load")

    response = client.post("/api/demo/load")

    assert response.status_code == 409
    assert response.json()["error"] == "demo_conflict"


def test_demo_load_never_calls_forecast_training(client: TestClient):
    from unittest.mock import patch

    # Same patch target test_forecasts.py's "no fit on read" tests use — the
    # name ForecastService actually calls, not the module it's imported from.
    # Demo seeding must never reach ForecastService.run_forecast() at all
    # (its prebuilt forecast is computed by demo_seed_data's plain arithmetic
    # instead — see generate_demo_forecast()'s docstring).
    with patch("backend.services.forecast_service.train_and_predict") as mock_train:
        response = client.post("/api/demo/load")

    assert response.status_code == 200
    mock_train.assert_not_called()


# -- clear ----------------------------------------------------------------


def test_demo_clear_removes_all_demo_rows_and_returns_to_empty(client: TestClient, conn: sqlite3.Connection):
    client.post("/api/demo/load")

    response = client.delete("/api/demo/clear")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "EMPTY"
    assert body["cleared"] is True
    assert body["summary"]["transactions"] > 0

    status = client.get("/api/demo/status").json()
    assert status == {"mode": "EMPTY", "can_load_demo": True}
    assert client.get("/api/transactions").json()["items"] == []
    assert client.get("/api/holdings").json() == []

    row = conn.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()
    assert row["n"] == 0
    row = conn.execute("SELECT COUNT(*) AS n FROM forecast_runs").fetchone()
    assert row["n"] == 0
    row = conn.execute("SELECT COUNT(*) AS n FROM price_cache").fetchone()
    assert row["n"] == 0


def test_demo_clear_is_idempotent_when_already_empty(client: TestClient):
    response = client.delete("/api/demo/clear")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "EMPTY"
    assert body["cleared"] is True


# -- full demo -> real sequence (TRD §14.4, Build Plan Phase 9 item 6) -----


def test_full_demo_to_real_sequence(client: TestClient, conn: sqlite3.Connection):
    # EMPTY -> Load Demo -> DEMO
    load_response = client.post("/api/demo/load")
    assert load_response.status_code == 200
    assert AppStateRepository(conn).get_mode() == "DEMO"

    # Attempt real import while DEMO -> 409 demo_conflict
    conflict_response = _upload(client)
    assert conflict_response.status_code == 409
    assert conflict_response.json()["error"] == "demo_conflict"

    # User confirms: DELETE /api/demo/clear -> EMPTY
    clear_response = client.delete("/api/demo/clear")
    assert clear_response.status_code == 200
    assert clear_response.json()["mode"] == "EMPTY"
    assert AppStateRepository(conn).get_mode() == "EMPTY"
    assert client.get("/api/transactions").json()["items"] == []

    # Retry the same import -> succeeds, mode becomes REAL
    retry_response = _upload(client)
    assert retry_response.status_code == 200
    preview = retry_response.json()
    confirm_response = client.post(f"/api/imports/{preview['batch_id']}/confirm")
    assert confirm_response.status_code == 200
    assert confirm_response.json()["rows_imported"] > 0

    assert AppStateRepository(conn).get_mode() == "REAL"
    transactions = client.get("/api/transactions").json()["items"]
    assert len(transactions) == confirm_response.json()["rows_imported"]
    # No demo rows leaked into the REAL view.
    for txn in transactions:
        assert txn["merchant"]  # sanity: real rows read back fine


def test_repeated_load_clear_cycle_does_not_accumulate_rows(client: TestClient, conn: sqlite3.Connection):
    first_load = client.post("/api/demo/load").json()
    client.delete("/api/demo/clear")
    second_load = client.post("/api/demo/load").json()

    assert first_load["summary"]["transactions"] == second_load["summary"]["transactions"]
    assert first_load["summary"]["holdings"] == second_load["summary"]["holdings"]

    row = conn.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()
    assert row["n"] == second_load["summary"]["transactions"]


def test_real_to_load_demo_rejected_without_deleting_real_data(client: TestClient, conn: sqlite3.Connection):
    client.post("/api/holdings", json={"ticker": "AAPL", "shares": 5, "avg_cost": 100.0})
    assert AppStateRepository(conn).get_mode() == "REAL"

    response = client.post("/api/demo/load")

    assert response.status_code == 409
    holdings = client.get("/api/holdings").json()
    assert len(holdings) == 1
    assert holdings[0]["ticker"] == "AAPL"
    assert AppStateRepository(conn).get_mode() == "REAL"


# -- clear-real-data (mirror image of clear) --------------------------------


def test_clear_real_data_removes_real_rows_and_returns_to_empty(client: TestClient, conn: sqlite3.Connection):
    client.post("/api/transactions", json={"date": "2026-01-15", "merchant": "TIM HORTONS", "amount": 4.5})
    assert AppStateRepository(conn).get_mode() == "REAL"

    response = client.delete("/api/demo/clear-real-data")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "EMPTY"
    assert body["cleared"] is True
    assert body["summary"]["transactions"] == 1

    status = client.get("/api/demo/status").json()
    assert status == {"mode": "EMPTY", "can_load_demo": True}
    assert client.get("/api/transactions").json()["items"] == []


def test_clear_real_data_is_idempotent_when_no_real_data(client: TestClient):
    response = client.delete("/api/demo/clear-real-data")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "EMPTY"
    assert body["cleared"] is True


def test_full_real_to_demo_sequence(client: TestClient, conn: sqlite3.Connection):
    """The mirror image of test_full_demo_to_real_sequence -- this is the
    exact flow the user asked about: import real data, then clear it
    in-app to unblock Load Demo Data, without shell access."""
    # EMPTY -> import real data -> REAL
    upload_response = _upload(client)
    assert upload_response.status_code == 200
    preview = upload_response.json()
    confirm_response = client.post(f"/api/imports/{preview['batch_id']}/confirm")
    assert confirm_response.status_code == 200
    assert AppStateRepository(conn).get_mode() == "REAL"

    # Attempt Load Demo while REAL -> 409 demo_conflict, real data untouched
    conflict_response = client.post("/api/demo/load")
    assert conflict_response.status_code == 409
    assert conflict_response.json()["error"] == "demo_conflict"
    assert len(client.get("/api/transactions").json()["items"]) > 0

    # User clears real data in-app -> EMPTY
    clear_response = client.delete("/api/demo/clear-real-data")
    assert clear_response.status_code == 200
    assert clear_response.json()["mode"] == "EMPTY"
    assert AppStateRepository(conn).get_mode() == "EMPTY"
    assert client.get("/api/transactions").json()["items"] == []

    # Retry Load Demo -> succeeds
    retry_response = client.post("/api/demo/load")
    assert retry_response.status_code == 200
    assert retry_response.json()["mode"] == "DEMO"
    assert AppStateRepository(conn).get_mode() == "DEMO"
