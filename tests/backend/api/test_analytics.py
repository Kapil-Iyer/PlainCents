"""Analytics API tests (ML-G): route wiring, validation, and mode scoping."""
import pytest
from fastapi.testclient import TestClient

from backend.repositories.app_state_repository import AppStateRepository
from backend.repositories.transaction_repository import TransactionRepository

ENDPOINTS = [
    "/api/analytics/category-trend",
    "/api/analytics/top-merchants",
    "/api/analytics/category-movers",
    "/api/analytics/spend-pace",
    "/api/analytics/forecast-accuracy",
]


def _seed(conn, mode="real", n=3):
    repo = TransactionRepository(conn)
    for i in range(n):
        repo.create({
            "date": f"2026-01-{i + 1:02d}",
            "merchant": f"NORTHSIDE PIZZA {i}",
            "amount": 10.0 + i,
            "bank_source": "RBC",
            "predicted_category": "Food & Dining",
            "data_mode": mode,
            "dedup_key": f"seed{mode}{i}",
        })
    conn.commit()


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_endpoint_returns_200_on_an_empty_database(client: TestClient, endpoint):
    """EMPTY mode must render a valid, well-typed empty result -- never a
    500 and never a chart drawn from nothing."""
    response = client.get(endpoint)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_endpoint_returns_200_with_data(client: TestClient, conn, endpoint):
    _seed(conn)
    AppStateRepository(conn).set_mode("REAL")
    conn.commit()

    response = client.get(endpoint)
    assert response.status_code == 200, response.text


def test_category_trend_rejects_out_of_range_window(client: TestClient):
    assert client.get("/api/analytics/category-trend?months=0").status_code == 422
    assert client.get("/api/analytics/category-trend?months=999").status_code == 422


def test_top_merchants_rejects_out_of_range_limit(client: TestClient):
    assert client.get("/api/analytics/top-merchants?limit=0").status_code == 422
    assert client.get("/api/analytics/top-merchants?limit=999").status_code == 422


def test_empty_mode_reports_no_spend_even_though_rows_exist(client: TestClient, conn):
    """Rows tagged `real` must stay invisible while the app is in EMPTY --
    the same mode-scoping contract every other read endpoint follows."""
    _seed(conn)
    # app_state left at its EMPTY default.

    body = client.get("/api/analytics/category-trend?months=3").json()
    assert body["categories"] == []
    assert all(p["total_spend"] == 0.0 for p in body["points"])


def test_demo_mode_reads_demo_rows_only(client: TestClient, conn):
    _seed(conn, mode="real", n=2)
    _seed(conn, mode="demo", n=1)
    AppStateRepository(conn).set_mode("DEMO")
    conn.commit()

    body = client.get("/api/analytics/top-merchants?months=24").json()
    assert body["distinct_merchants"] == 1


def test_forecast_accuracy_is_honest_about_having_no_snapshots(client: TestClient, conn):
    _seed(conn)
    AppStateRepository(conn).set_mode("REAL")
    conn.commit()

    body = client.get("/api/analytics/forecast-accuracy").json()
    assert body["available"] is False
    assert body["reason"] == "no_snapshots_yet"
    assert body["items"] == []
