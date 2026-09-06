"""DemoService unit tests (Build Plan Phase 9, item 8): state-machine
transitions, atomicity, REAL-data protection, price_cache ownership
semantics, and deterministic seed reproducibility."""
from unittest.mock import patch

import pytest

from backend.api.errors import DemoConflictError
from backend.repositories.app_state_repository import AppStateRepository
from backend.repositories.holding_repository import HoldingRepository
from backend.repositories.price_cache_repository import PriceCacheRepository
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.demo_seed_data import generate_demo_transactions
from backend.services.demo_service import DemoService


@pytest.fixture
def app_state(conn):
    return AppStateRepository(conn)


@pytest.fixture
def service(conn):
    return DemoService(conn)


# -- load_demo ------------------------------------------------------------


def test_load_demo_from_empty_transitions_to_demo(service, app_state):
    assert app_state.get_mode() == "EMPTY"

    result = service.load_demo()

    assert result["mode"] == "DEMO"
    assert app_state.get_mode() == "DEMO"


def test_load_demo_seeds_transactions_across_at_least_12_months_and_all_categories(service, conn):
    service.load_demo()

    rows = TransactionRepository(conn).list(data_mode="demo")
    assert len(rows) > 0
    months = {r["date"][:7] for r in rows}
    assert len(months) >= 12
    categories = {r["predicted_category"] for r in rows}
    assert categories == {
        "Food & Dining", "Transport", "Rent & Utilities", "Entertainment",
        "Healthcare", "Shopping", "Subscriptions", "Other",
    }
    assert all(r["data_mode"] == "demo" for r in rows)


def test_load_demo_seeds_holdings_with_price_cache(service, conn):
    from backend.services.demo_seed_data import DEMO_PRICE_FETCHED_AT

    service.load_demo()

    holdings = HoldingRepository(conn).list(data_mode="demo")
    assert len(holdings) >= 1
    price_cache = PriceCacheRepository(conn)
    for holding in holdings:
        cached = price_cache.get_last_known(holding["ticker"])
        assert cached is not None
        assert cached["current_price"] > 0
        # PATCH B: this fixed sentinel (never datetime.now()) is what lets
        # PortfolioService tell a genuinely-never-fetched demo price apart
        # from a real cached one, however old -- see PortfolioService.
        # _to_response's `price_is_demo_snapshot`.
        assert cached["fetched_at"] == DEMO_PRICE_FETCHED_AT


def test_load_demo_seeds_prebuilt_forecast_without_training(service, conn):
    with patch("backend.services.forecast_service.train_and_predict") as mock_train:
        service.load_demo()

    mock_train.assert_not_called()

    run = conn.execute("SELECT * FROM forecast_runs WHERE data_mode = 'demo'").fetchone()
    assert run is not None
    assert run["model_impl_version"] == "demo_seed_v1"  # distinct from the real fit's tag
    predictions = conn.execute(
        "SELECT * FROM forecast_predictions WHERE forecast_run_id = ?", (run["id"],)
    ).fetchall()
    assert len(predictions) == 24  # 8 categories x 3 months


def test_load_demo_rejected_when_real_data_exists(service, conn, app_state):
    TransactionRepository(conn).create(
        {
            "date": "2026-01-15", "merchant": "TIM HORTONS", "amount": 4.5,
            "predicted_category": "Food & Dining", "data_mode": "real",
            "dedup_key": "2026-01-15|4.5|TIM HORTONS||0",
        }
    )
    app_state.set_mode("REAL")
    conn.commit()

    with pytest.raises(DemoConflictError):
        service.load_demo()

    assert app_state.get_mode() == "REAL"
    assert len(TransactionRepository(conn).list(data_mode="demo")) == 0
    assert len(TransactionRepository(conn).list(data_mode="real")) == 1


def test_load_demo_rejected_when_already_demo(service, app_state):
    service.load_demo()

    with pytest.raises(DemoConflictError):
        service.load_demo()

    assert app_state.get_mode() == "DEMO"


def test_load_demo_atomicity_on_mid_seed_failure(service, conn, app_state):
    with patch(
        "backend.repositories.holding_repository.HoldingRepository.create",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError):
            service.load_demo()

    # No partial demo seed, no falsely-DEMO mode.
    assert app_state.get_mode() == "EMPTY"
    assert len(TransactionRepository(conn).list(data_mode="demo")) == 0
    row = conn.execute("SELECT COUNT(*) AS n FROM forecast_runs").fetchone()
    assert row["n"] == 0


# -- clear_demo -------------------------------------------------------------


def test_clear_demo_removes_all_and_only_demo_rows(service, conn, app_state):
    service.load_demo()

    result = service.clear_demo()

    assert result["mode"] == "EMPTY"
    assert app_state.get_mode() == "EMPTY"
    assert len(TransactionRepository(conn).list(data_mode="demo")) == 0
    assert len(HoldingRepository(conn).list(data_mode="demo")) == 0
    row = conn.execute("SELECT COUNT(*) AS n FROM forecast_runs").fetchone()
    assert row["n"] == 0
    row = conn.execute("SELECT COUNT(*) AS n FROM price_cache").fetchone()
    assert row["n"] == 0


def test_clear_demo_preserves_real_rows(service, conn):
    # Row-level isolation test (defense in depth per TRD §4.5): clear_demo()
    # deletes strictly WHERE data_mode='demo', so real rows inserted
    # directly via the repository survive regardless of call context.
    TransactionRepository(conn).create(
        {
            "date": "2026-01-15", "merchant": "REAL MERCHANT", "amount": 10.0,
            "predicted_category": "Other", "data_mode": "real",
            "dedup_key": "2026-01-15|10.0|REAL MERCHANT||0",
        }
    )
    HoldingRepository(conn).create({"ticker": "VOO", "shares": 1, "avg_cost": 1.0, "data_mode": "real"})
    conn.commit()

    service.clear_demo()

    real_txns = TransactionRepository(conn).list(data_mode="real")
    assert len(real_txns) == 1
    assert real_txns[0]["merchant"] == "REAL MERCHANT"
    real_holdings = HoldingRepository(conn).list(data_mode="real")
    assert len(real_holdings) == 1
    assert real_holdings[0]["ticker"] == "VOO"


def test_clear_demo_preserves_price_cache_shared_with_real_holding(service, conn):
    service.load_demo()
    demo_holdings = HoldingRepository(conn).list(data_mode="demo")
    shared_ticker = demo_holdings[0]["ticker"]
    # A real holding on the same ticker as a demo one.
    HoldingRepository(conn).create({"ticker": shared_ticker, "shares": 1, "avg_cost": 1.0, "data_mode": "real"})
    conn.commit()

    service.clear_demo()

    cached = PriceCacheRepository(conn).get_last_known(shared_ticker)
    assert cached is not None  # not deleted — still needed by the real holding


def test_clear_demo_is_idempotent_when_already_empty(service, app_state):
    result = service.clear_demo()

    assert result["mode"] == "EMPTY"
    assert result["cleared"] is True
    assert app_state.get_mode() == "EMPTY"


def test_repeated_load_clear_cycle_reaches_identical_clean_state(service, conn):
    service.load_demo()
    service.clear_demo()
    first_state = {
        "transactions": conn.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()["n"],
        "holdings": conn.execute("SELECT COUNT(*) AS n FROM holdings").fetchone()["n"],
        "forecast_runs": conn.execute("SELECT COUNT(*) AS n FROM forecast_runs").fetchone()["n"],
        "price_cache": conn.execute("SELECT COUNT(*) AS n FROM price_cache").fetchone()["n"],
    }

    service.load_demo()
    service.clear_demo()
    second_state = {
        "transactions": conn.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()["n"],
        "holdings": conn.execute("SELECT COUNT(*) AS n FROM holdings").fetchone()["n"],
        "forecast_runs": conn.execute("SELECT COUNT(*) AS n FROM forecast_runs").fetchone()["n"],
        "price_cache": conn.execute("SELECT COUNT(*) AS n FROM price_cache").fetchone()["n"],
    }

    assert first_state == second_state == {"transactions": 0, "holdings": 0, "forecast_runs": 0, "price_cache": 0}


# -- clear_real_data (mirror image of clear_demo) ---------------------------


def _create_real_transaction(conn, merchant="REAL MERCHANT", dedup="dk-real-1"):
    return TransactionRepository(conn).create(
        {
            "date": "2026-01-15", "merchant": merchant, "amount": 10.0,
            "predicted_category": "Other", "data_mode": "real",
            "dedup_key": dedup,
        }
    )


def test_clear_real_data_removes_all_and_only_real_rows(service, conn, app_state):
    _create_real_transaction(conn)
    HoldingRepository(conn).create({"ticker": "VOO", "shares": 1, "avg_cost": 1.0, "data_mode": "real"})
    app_state.set_mode("REAL")
    conn.commit()

    result = service.clear_real_data()

    assert result["mode"] == "EMPTY"
    assert result["cleared"] is True
    assert app_state.get_mode() == "EMPTY"
    assert len(TransactionRepository(conn).list(data_mode="real")) == 0
    assert len(HoldingRepository(conn).list(data_mode="real")) == 0
    row = conn.execute("SELECT COUNT(*) AS n FROM price_cache").fetchone()
    assert row["n"] == 0


def test_clear_real_data_unblocks_load_demo(service, conn, app_state):
    """This is the whole point: DemoService.load_demo() rejects with 409
    while mode == 'REAL' -- clearing real data must actually unblock it."""
    _create_real_transaction(conn)
    app_state.set_mode("REAL")
    conn.commit()

    with pytest.raises(DemoConflictError):
        service.load_demo()

    service.clear_real_data()

    result = service.load_demo()  # must not raise anymore
    assert result["mode"] == "DEMO"


def test_clear_real_data_preserves_demo_rows(service, conn):
    # Row-level isolation, mirroring test_clear_demo_preserves_real_rows:
    # clear_real_data() deletes strictly WHERE data_mode='real'.
    service.load_demo()
    demo_txns_before = len(TransactionRepository(conn).list(data_mode="demo"))
    assert demo_txns_before > 0

    service.clear_real_data()

    assert len(TransactionRepository(conn).list(data_mode="demo")) == demo_txns_before


def test_clear_real_data_preserves_price_cache_shared_with_demo_holding(service, conn):
    service.load_demo()
    demo_holdings = HoldingRepository(conn).list(data_mode="demo")
    shared_ticker = demo_holdings[0]["ticker"]
    HoldingRepository(conn).create({"ticker": shared_ticker, "shares": 1, "avg_cost": 1.0, "data_mode": "real"})
    conn.commit()

    service.clear_real_data()

    cached = PriceCacheRepository(conn).get_last_known(shared_ticker)
    assert cached is not None  # not deleted -- still needed by the demo holding


def test_clear_real_data_is_idempotent_when_no_real_data(service, app_state):
    result = service.clear_real_data()

    assert result["mode"] == "EMPTY"
    assert result["cleared"] is True
    assert app_state.get_mode() == "EMPTY"


# -- deterministic seed -----------------------------------------------------


def test_seed_generation_is_deterministic():
    from datetime import date

    first = generate_demo_transactions(as_of=date(2026, 9, 2))
    second = generate_demo_transactions(as_of=date(2026, 9, 2))

    assert first == second


def test_seed_generation_never_dates_a_current_month_row_after_as_of():
    """BUG FIX regression test: the current calendar month's rows must never
    land after `as_of` -- a future-dated "spent so far" row is never
    honest, and previously this generated up to day 28 unconditionally, even
    early in the month (see date_windows.py / dashboard_service's day-capped
    'total_spend_current')."""
    from datetime import date

    as_of = date(2026, 9, 6)
    rows = generate_demo_transactions(as_of=as_of)
    current_month_days = [
        int(r["date"][8:10]) for r in rows if r["date"].startswith("2026-09")
    ]

    assert current_month_days  # the current month does have rows...
    assert max(current_month_days) <= as_of.day  # ...none of them in the future


def test_seed_generation_handles_day_1_of_the_month():
    """The edge case where `as_of` is the 1st of the month -- every current-
    month row must land on day 1, never later, and generation must not
    crash on a single-day range."""
    from datetime import date

    as_of = date(2026, 9, 1)
    rows = generate_demo_transactions(as_of=as_of)
    current_month_days = {
        int(r["date"][8:10]) for r in rows if r["date"].startswith("2026-09")
    }

    assert current_month_days == {1}
