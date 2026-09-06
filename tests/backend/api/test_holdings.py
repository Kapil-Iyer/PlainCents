"""Holdings CRUD + refresh API tests (Build Plan Phase 8, item 8): full CRUD
via HTTP, no-network-on-GET, refresh partial failure, and validation."""
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.repositories.price_cache_repository import PriceCacheRepository


def _sample(**overrides):
    data = {"ticker": "AAPL", "shares": 10, "avg_cost": 100.0}
    data.update(overrides)
    return data


def _fake_fetch_price(conn, price_map: dict):
    """A stand-in for pipeline.portfolio.fetch_price that mimics its
    documented side effect (upserts price_cache on success, touches nothing
    on failure) so the read-after-refresh assertions see realistic cache
    state without hitting yfinance."""

    def _fetch(_conn, ticker):
        price = price_map.get(ticker)
        if price is None:
            return None
        PriceCacheRepository(conn).upsert_latest(ticker, price, "2026-01-15T00:00:00")
        return price

    return _fetch


def test_create_holding(client: TestClient):
    response = client.post("/api/holdings", json=_sample())

    assert response.status_code == 201
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["current_price"] is None
    assert body["current_value"] is None
    assert body["pnl"] is None
    assert body["price_is_demo_snapshot"] is False


def test_price_at_demo_sentinel_timestamp_is_flagged_a_snapshot(client: TestClient, conn):
    """PATCH B: a price stamped with the exact demo-seed sentinel timestamp
    must come back flagged so the frontend can say "Demo snapshot" instead
    of implying a real, merely-old cached fetch."""
    from backend.services.demo_seed_data import DEMO_PRICE_FETCHED_AT

    created = client.post("/api/holdings", json=_sample()).json()
    PriceCacheRepository(conn).upsert_latest("AAPL", 178.50, DEMO_PRICE_FETCHED_AT)
    conn.commit()

    body = client.get("/api/holdings").json()

    assert body[0]["id"] == created["id"]
    assert body[0]["price_is_demo_snapshot"] is True
    assert body[0]["price_last_updated"] == DEMO_PRICE_FETCHED_AT


def test_create_holding_rejects_non_positive_shares(client: TestClient):
    response = client.post("/api/holdings", json=_sample(shares=0))

    assert response.status_code == 422


def test_create_holding_rejects_negative_avg_cost(client: TestClient):
    response = client.post("/api/holdings", json=_sample(avg_cost=-1))

    assert response.status_code == 422


def test_create_holding_without_avg_cost_field_succeeds(client: TestClient):
    """Ticker and shares are required; average cost is not."""
    response = client.post("/api/holdings", json={"ticker": "MSFT", "shares": 10})

    assert response.status_code == 201
    body = response.json()
    assert body["avg_cost"] is None
    assert body["pnl"] is None


def test_create_holding_with_explicit_null_avg_cost_succeeds(client: TestClient):
    response = client.post("/api/holdings", json=_sample(avg_cost=None))

    assert response.status_code == 201
    assert response.json()["avg_cost"] is None


def test_market_value_available_without_cost_basis_but_pnl_is_not(client: TestClient, conn):
    client.post("/api/holdings", json={"ticker": "AAPL", "shares": 10})

    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price(conn, {"AAPL": 150.0}),
    ):
        client.post("/api/holdings/refresh-prices")

    holdings = client.get("/api/holdings").json()
    assert holdings[0]["current_value"] == 1500.0
    assert holdings[0]["pnl"] is None


def test_avg_cost_can_be_added_later_and_pnl_then_appears(client: TestClient, conn):
    created = client.post("/api/holdings", json={"ticker": "AAPL", "shares": 10}).json()
    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price(conn, {"AAPL": 150.0}),
    ):
        client.post("/api/holdings/refresh-prices")

    response = client.patch(f"/api/holdings/{created['id']}", json={"avg_cost": 100.0})

    assert response.status_code == 200
    body = response.json()
    assert body["avg_cost"] == 100.0
    assert body["pnl"] == 500.0


def test_avg_cost_can_be_cleared_via_explicit_null(client: TestClient):
    created = client.post("/api/holdings", json=_sample(avg_cost=100.0)).json()
    assert created["avg_cost"] == 100.0

    response = client.patch(f"/api/holdings/{created['id']}", json={"avg_cost": None})

    assert response.status_code == 200
    assert response.json()["avg_cost"] is None


def test_refresh_prices_does_not_mutate_shares_or_avg_cost(client: TestClient, conn):
    created = client.post("/api/holdings", json=_sample(shares=10, avg_cost=100.0)).json()

    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price(conn, {"AAPL": 150.0}),
    ):
        client.post("/api/holdings/refresh-prices")

    holdings = client.get("/api/holdings").json()
    assert holdings[0]["id"] == created["id"]
    assert holdings[0]["shares"] == 10
    assert holdings[0]["avg_cost"] == 100.0


def test_list_holdings_never_calls_fetch_price(client: TestClient):
    client.post("/api/holdings", json=_sample())

    with patch("backend.services.portfolio_service.fetch_price") as mock_fetch:
        response = client.get("/api/holdings")

    mock_fetch.assert_not_called()
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_list_holdings_empty_mode_returns_empty_list(client: TestClient):
    response = client.get("/api/holdings")

    assert response.status_code == 200
    assert response.json() == []


def test_refresh_prices_success(client: TestClient, conn):
    client.post("/api/holdings", json=_sample())

    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price(conn, {"AAPL": 150.0}),
    ):
        response = client.post("/api/holdings/refresh-prices")

    assert response.status_code == 200
    body = response.json()
    assert body["refreshed"] == [{"ticker": "AAPL", "price": 150.0}]
    assert body["failed"] == []

    holdings = client.get("/api/holdings").json()
    assert holdings[0]["current_price"] == 150.0
    assert holdings[0]["current_value"] == 1500.0
    assert holdings[0]["pnl"] == 500.0


def test_refresh_prices_partial_failure_returns_200(client: TestClient, conn):
    client.post("/api/holdings", json=_sample(ticker="AAPL"))
    client.post("/api/holdings", json=_sample(ticker="BADTICKER"))

    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price(conn, {"AAPL": 150.0}),
    ):
        response = client.post("/api/holdings/refresh-prices")

    assert response.status_code == 200
    body = response.json()
    assert {"ticker": "AAPL", "price": 150.0} in body["refreshed"]
    assert {"ticker": "BADTICKER", "error": "price_fetch_failed"} in body["failed"]


def test_update_holding_shares(client: TestClient):
    created = client.post("/api/holdings", json=_sample()).json()

    response = client.patch(f"/api/holdings/{created['id']}", json={"shares": 20})

    assert response.status_code == 200
    assert response.json()["shares"] == 20


def test_update_missing_holding_returns_404_envelope(client: TestClient):
    response = client.patch("/api/holdings/999999", json={"shares": 20})

    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_delete_holding(client: TestClient):
    created = client.post("/api/holdings", json=_sample()).json()

    response = client.delete(f"/api/holdings/{created['id']}")

    assert response.status_code == 200
    assert response.json() == {"id": created["id"], "deleted": True}
    assert client.get("/api/holdings").json() == []


def test_delete_missing_holding_returns_404_envelope(client: TestClient):
    response = client.delete("/api/holdings/999999")

    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_create_holding_transitions_empty_to_real(client: TestClient):
    client.post("/api/holdings", json=_sample())

    health = client.get("/api/health").json()
    assert health["data_mode"] == "REAL"
