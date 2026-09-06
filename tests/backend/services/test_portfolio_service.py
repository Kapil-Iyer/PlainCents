"""PortfolioService unit tests (Build Plan Phase 8, item 8): read/refresh
separation (GET path never calls fetch_price), refresh partial failure with
cache preserved, CRUD, EMPTY -> REAL transition, and financial nullability."""
from unittest.mock import patch

import pytest

from backend.api.errors import BadRequestError, NotFoundError
from backend.repositories.price_cache_repository import PriceCacheRepository
from backend.services.app_state_service import AppStateService
from backend.services.portfolio_service import PortfolioService


@pytest.fixture
def app_state_service(conn):
    return AppStateService(conn)


@pytest.fixture
def service(conn, app_state_service):
    return PortfolioService(conn, app_state_service=app_state_service)


def _sample(**overrides):
    data = {"ticker": "AAPL", "shares": 10, "avg_cost": 100.0}
    data.update(overrides)
    return data


def _fake_fetch_price(price_map: dict):
    """A stand-in for pipeline.portfolio.fetch_price that mimics its
    documented side effect (cache-first, upserts price_cache on a
    successful fetch, returns None without touching the cache on failure)
    so tests exercising the service's read-after-refresh path see the same
    cache state the real function would leave behind."""

    def _fetch(conn, ticker):
        price = price_map.get(ticker)
        if price is None:
            return None
        PriceCacheRepository(conn).upsert_latest(ticker, price, "2026-01-15T00:00:00")
        return price

    return _fetch


def test_create_holding_transitions_empty_to_real(service, app_state_service):
    assert app_state_service.get_mode() == "EMPTY"

    service.create_holding(_sample())

    assert app_state_service.get_mode() == "REAL"


def test_create_holding_uppercases_and_strips_ticker(service):
    row = service.create_holding(_sample(ticker=" aapl "))

    assert row["ticker"] == "AAPL"


def test_get_holdings_with_prices_never_calls_fetch_price(service):
    service.create_holding(_sample())

    with patch("backend.services.portfolio_service.fetch_price") as mock_fetch:
        rows = service.get_holdings_with_prices(data_mode="real")

    mock_fetch.assert_not_called()
    assert len(rows) == 1


def test_never_refreshed_holding_has_null_price_and_pnl(service):
    row = service.create_holding(_sample())

    assert row["current_price"] is None
    assert row["current_value"] is None
    assert row["pnl"] is None
    assert row["price_last_updated"] is None


def test_refresh_prices_populates_cache_and_pnl(service, conn):
    service.create_holding(_sample(ticker="AAPL", shares=10, avg_cost=100.0))

    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price({"AAPL": 150.0}),
    ):
        result = service.refresh_prices(data_mode="real")

    assert result == {"refreshed": [{"ticker": "AAPL", "price": 150.0}], "failed": []}

    rows = service.get_holdings_with_prices(data_mode="real")
    assert rows[0]["current_price"] == 150.0
    assert rows[0]["current_value"] == 1500.0
    assert rows[0]["pnl"] == 500.0
    assert rows[0]["price_last_updated"] is not None
    # A genuine fetch is never mistaken for a demo snapshot.
    assert rows[0]["price_is_demo_snapshot"] is False


# -- demo price-snapshot honesty (PATCH B) ------------------------------------


def test_never_refreshed_holding_is_not_flagged_a_demo_snapshot(service):
    """No cached price at all is a different, already-honest state ("Not yet
    refreshed") -- it must not also claim to be a demo snapshot."""
    row = service.create_holding(_sample())

    assert row["price_is_demo_snapshot"] is False


def test_price_cached_at_the_demo_sentinel_timestamp_is_flagged(service, conn):
    """The exact fixed timestamp DemoService.load_demo() stamps on a
    never-actually-fetched seeded price must be recognized as a demo
    snapshot, never presented as if it were a real (if old) cached fetch."""
    from backend.services.demo_seed_data import DEMO_PRICE_FETCHED_AT

    service.create_holding(_sample(ticker="AAPL"))
    PriceCacheRepository(conn).upsert_latest("AAPL", 178.50, DEMO_PRICE_FETCHED_AT)
    conn.commit()

    row = service.get_holding(service.get_holdings_with_prices(data_mode="real")[0]["id"])

    assert row["price_is_demo_snapshot"] is True
    assert row["price_last_updated"] == DEMO_PRICE_FETCHED_AT


def test_refreshing_a_demo_snapshot_price_clears_the_flag(service, conn):
    """Once a demo holding is refreshed with a genuine fetch, its
    fetched_at is no longer the sentinel -- the flag must flip to False
    with no extra bookkeeping, the same way a real holding's would."""
    from backend.services.demo_seed_data import DEMO_PRICE_FETCHED_AT

    holding = service.create_holding(_sample(ticker="AAPL"))
    PriceCacheRepository(conn).upsert_latest("AAPL", 178.50, DEMO_PRICE_FETCHED_AT)
    conn.commit()
    assert service.get_holding(holding["id"])["price_is_demo_snapshot"] is True

    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price({"AAPL": 185.0}),
    ):
        service.refresh_prices(data_mode="real")

    refreshed = service.get_holding(holding["id"])
    assert refreshed["price_is_demo_snapshot"] is False
    assert refreshed["current_price"] == 185.0


def test_refresh_prices_partial_failure_preserves_other_tickers(service):
    service.create_holding(_sample(ticker="AAPL"))
    service.create_holding(_sample(ticker="BADTICKER"))

    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price({"AAPL": 150.0}),
    ):
        result = service.refresh_prices(data_mode="real")

    assert result["refreshed"] == [{"ticker": "AAPL", "price": 150.0}]
    assert result["failed"] == [{"ticker": "BADTICKER", "error": "price_fetch_failed"}]


def test_refresh_prices_failure_preserves_last_good_cache(service):
    service.create_holding(_sample(ticker="AAPL"))

    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price({"AAPL": 150.0}),
    ):
        service.refresh_prices(data_mode="real")

    with patch("backend.services.portfolio_service.fetch_price", return_value=None):
        result = service.refresh_prices(data_mode="real")

    assert result["failed"] == [{"ticker": "AAPL", "error": "price_fetch_failed"}]

    rows = service.get_holdings_with_prices(data_mode="real")
    assert rows[0]["current_price"] == 150.0  # last-good price preserved, not overwritten


# -- optional cost basis (portfolio + Power BI completion pass) -------------


def test_create_holding_without_avg_cost_succeeds(service):
    """Ticker and shares are required; average cost is not -- a user who
    only knows "I own 10 MSFT shares" must still be able to add it."""
    row = service.create_holding({"ticker": "MSFT", "shares": 10})

    assert row["avg_cost"] is None


def test_create_holding_with_explicit_null_avg_cost_succeeds(service):
    row = service.create_holding(_sample(avg_cost=None))

    assert row["avg_cost"] is None


def test_market_value_works_without_avg_cost(service, conn):
    service.create_holding({"ticker": "AAPL", "shares": 10})
    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price({"AAPL": 150.0}),
    ):
        service.refresh_prices(data_mode="real")

    rows = service.get_holdings_with_prices(data_mode="real")

    assert rows[0]["current_price"] == 150.0
    assert rows[0]["current_value"] == 1500.0  # shares * price, no cost basis needed


def test_pnl_is_null_without_avg_cost_even_with_a_known_price(service):
    service.create_holding({"ticker": "AAPL", "shares": 10})
    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price({"AAPL": 150.0}),
    ):
        service.refresh_prices(data_mode="real")

    rows = service.get_holdings_with_prices(data_mode="real")

    assert rows[0]["pnl"] is None  # never fabricated as 0 or derived from current_price


def test_pnl_correct_once_avg_cost_is_known(service):
    created = service.create_holding({"ticker": "AAPL", "shares": 10})
    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price({"AAPL": 150.0}),
    ):
        service.refresh_prices(data_mode="real")

    service.update_holding(created["id"], {"avg_cost": 100.0})
    row = service.get_holding(created["id"])

    assert row["avg_cost"] == 100.0
    assert row["pnl"] == 500.0  # (150 - 100) * 10


def test_avg_cost_can_be_added_later_via_update(service):
    created = service.create_holding({"ticker": "AAPL", "shares": 10})
    assert created["avg_cost"] is None

    updated = service.update_holding(created["id"], {"avg_cost": 120.0})

    assert updated["avg_cost"] == 120.0


def test_avg_cost_can_be_cleared_via_update(service):
    created = service.create_holding(_sample(avg_cost=100.0))
    assert created["avg_cost"] == 100.0

    cleared = service.update_holding(created["id"], {"avg_cost": None})

    assert cleared["avg_cost"] is None
    assert cleared["pnl"] is None


def test_refresh_prices_never_mutates_shares_or_avg_cost(service):
    created = service.create_holding(_sample(ticker="AAPL", shares=10, avg_cost=100.0))

    with patch(
        "backend.services.portfolio_service.fetch_price",
        side_effect=_fake_fetch_price({"AAPL": 150.0}),
    ):
        service.refresh_prices(data_mode="real")

    row = service.get_holding(created["id"])
    assert row["shares"] == 10
    assert row["avg_cost"] == 100.0


def test_update_holding_changes_shares(service):
    created = service.create_holding(_sample())

    updated = service.update_holding(created["id"], {"shares": 20})

    assert updated["shares"] == 20
    assert updated["avg_cost"] == created["avg_cost"]


def test_update_holding_no_fields_raises_bad_request(service):
    created = service.create_holding(_sample())

    with pytest.raises(BadRequestError):
        service.update_holding(created["id"], {})


def test_update_missing_holding_raises_not_found(service):
    with pytest.raises(NotFoundError):
        service.update_holding(999999, {"shares": 5})


def test_delete_holding_removes_it(service):
    created = service.create_holding(_sample())

    service.delete_holding(created["id"])

    with pytest.raises(NotFoundError):
        service.get_holding(created["id"])


def test_delete_missing_holding_raises_not_found(service):
    with pytest.raises(NotFoundError):
        service.delete_holding(999999)


def test_create_holding_survives_reconnect(conn, db_path, app_state_service):
    # Mirrors TransactionService's Phase 3 closure patch: the insert and the
    # EMPTY -> REAL transition must commit as one durable unit.
    service = PortfolioService(conn, app_state_service=app_state_service)
    service.create_holding(_sample())
    conn.close()

    import sqlite3

    reconnected = sqlite3.connect(str(db_path))
    reconnected.row_factory = sqlite3.Row
    mode = AppStateService(reconnected).get_mode()
    assert mode == "REAL"
    reconnected.close()
