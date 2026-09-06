"""Dashboard API tests (Build Plan Phase 6, item 8): response shape and
empty/real-mode behavior via HTTP.

The route always uses the real wall-clock date for "current calendar month"
(no reference_date override exists at the API layer — that's a
DashboardService-only test seam, exercised in test_dashboard_service.py).
So any transaction meant to land in the *current* month here must be dated
against today's actual date, not a hardcoded string.
"""
from datetime import date

from fastapi.testclient import TestClient

_TODAY = date.today().isoformat()


def _sample(**overrides):
    data = {"date": _TODAY, "merchant": "TIM HORTONS", "amount": 4.50}
    data.update(overrides)
    return data


def test_summary_shape_on_empty_database(client: TestClient):
    response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["data_mode"] == "EMPTY"
    assert body["total_spend_current"] == 0
    assert body["total_spend_previous"] == 0
    assert body["category_breakdown"] == []
    assert body["recent_transactions"] == []
    assert isinstance(body["spending_trend"], list) and len(body["spending_trend"]) > 0
    assert body["forecast_summary"] is None
    assert body["portfolio_summary"] is None
    assert set(body["period"].keys()) == {"current", "previous"}


def test_summary_reflects_real_transaction_after_create(client: TestClient):
    client.post("/api/transactions", json=_sample())

    response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["data_mode"] == "REAL"
    assert len(body["recent_transactions"]) == 1
    assert body["recent_transactions"][0]["merchant"] == "TIM HORTONS"


def test_summary_reflects_confirmed_category_correction(client: TestClient):
    created = client.post("/api/transactions", json=_sample()).json()
    client.patch(f"/api/transactions/{created['id']}", json={"confirmed_category": "Healthcare"})

    response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    body = response.json()
    categories = {item["category"] for item in body["category_breakdown"]}
    assert categories == {"Healthcare"}


# -- month query param: the shared analysis-month selector -------------------


def test_summary_month_param_selects_a_completed_historical_month(client: TestClient):
    """A transaction dated in a past month must show up as that month's
    FULL total when explicitly selected via ?month=, even though it would
    never appear in the (day-capped) current-month summary."""
    past_month = "2025-01"
    client.post("/api/transactions", json=_sample(date=f"{past_month}-20", amount=42.0))

    response = client.get(f"/api/dashboard/summary?month={past_month}")

    assert response.status_code == 200
    body = response.json()
    assert body["period"]["current"] == past_month
    assert body["is_current_incomplete"] is False
    assert body["total_spend_current"] == 42.0


def test_summary_no_month_param_defaults_to_current_incomplete_month(client: TestClient):
    response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    assert response.json()["is_current_incomplete"] is True


# -- available-months ----------------------------------------------------------


def test_available_months_empty_database_returns_empty_list(client: TestClient):
    response = client.get("/api/dashboard/available-months")

    assert response.status_code == 200
    assert response.json()["months"] == []


def test_available_months_lists_distinct_months_newest_first(client: TestClient):
    client.post("/api/transactions", json=_sample(date="2025-01-10", amount=1.0))
    client.post("/api/transactions", json=_sample(date="2025-03-10", amount=2.0))
    client.post("/api/transactions", json=_sample(date="2025-01-25", amount=3.0))  # same month as first

    response = client.get("/api/dashboard/available-months")

    assert response.status_code == 200
    assert response.json()["months"] == ["2025-03", "2025-01"]
