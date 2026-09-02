"""
ForecastService tests (Build Plan Phase 7, item 8): cold-start detection,
run persistence/retention, staleness decision table (the real hook,
extending Phase 3's stub-based tests), per-category availability, and
effective-category aggregation.

check_status()/get_latest()/run_forecast() take an already-resolved
data_mode (route-resolved, TRD Section 4.5.1 — same convention as
DashboardService), so tests pass data_mode="real" directly rather than
threading app_state.mode through AppStateRepository for every seed.
"""
from unittest.mock import patch

import pytest

from backend.api.errors import ForecastColdStartError
from backend.repositories.forecast_repository import ForecastRepository
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.forecast_service import MODEL_IMPL_VERSION, ForecastService

from tests.backend.services.forecast_fixtures import seed_months, seed_sparse_category


@pytest.fixture
def service(conn):
    return ForecastService(conn)


# -- check_status ------------------------------------------------------------


def test_check_status_cold_start_when_below_12_months(service, conn):
    seed_months(conn, 5, ["Food & Dining"])

    status = service.check_status("real")

    assert status["status"] == "cold_start"
    assert status["months_available"] == 5
    assert status["months_required"] == 12
    assert status["latest_run_id"] is None
    assert status["is_stale"] is None


def test_check_status_no_forecast_yet_when_eligible_but_no_run(service, conn):
    seed_months(conn, 12, ["Food & Dining"])

    status = service.check_status("real")

    assert status["status"] == "no_forecast_yet"
    assert status["months_available"] == 12


def test_check_status_ready_reflects_latest_run_and_staleness(service, conn):
    seed_months(conn, 12, ["Food & Dining"])
    run = service.run_forecast("real")

    status = service.check_status("real")
    assert status["status"] == "ready"
    assert status["latest_run_id"] == run["run_id"]
    assert status["is_stale"] is False

    service.mark_stale("transaction_updated")
    status_after = service.check_status("real")
    assert status_after["is_stale"] is True


def test_check_status_never_fits(service, conn):
    seed_months(conn, 12, ["Food & Dining"])
    with patch("backend.services.forecast_service.train_and_predict") as mock_fit:
        service.check_status("real")
    mock_fit.assert_not_called()


# -- get_latest ----------------------------------------------------------------


def test_get_latest_returns_none_when_no_run_exists(service, conn):
    seed_months(conn, 12, ["Food & Dining"])
    assert service.get_latest("real") is None


def test_get_latest_never_fits(service, conn):
    seed_months(conn, 12, ["Food & Dining"])
    service.run_forecast("real")
    with patch("backend.services.forecast_service.train_and_predict") as mock_fit:
        service.get_latest("real")
    mock_fit.assert_not_called()


# -- run_forecast: cold start --------------------------------------------------


def test_run_forecast_raises_cold_start_error_below_12_months(service, conn):
    seed_months(conn, 3, ["Food & Dining"])

    with pytest.raises(ForecastColdStartError):
        service.run_forecast("real")

    # Nothing persisted on a rejected cold-start attempt.
    assert ForecastRepository(conn).get_latest_run(data_mode="real") is None


# -- run_forecast: success / persistence / retention --------------------------


def test_run_forecast_persists_run_with_model_impl_version(service, conn):
    seed_months(conn, 12, ["Food & Dining", "Transport"])

    run = service.run_forecast("real")

    stored = ForecastRepository(conn).get_run(run["run_id"])
    assert stored["model_impl_version"] == MODEL_IMPL_VERSION
    # ML-D: production forecaster is the ML-C selected Naive baseline,
    # strategy "N/A" — pinned literally so this test fails loudly if the
    # constant drifts, not just self-consistently against its own import.
    assert MODEL_IMPL_VERSION == "naive_v1"
    assert stored["months_available"] == 12
    assert stored["data_mode"] == "real"
    assert stored["is_stale"] == 0


def test_run_forecast_never_fits_a_random_forest(service, conn):
    seed_months(conn, 12, ["Food & Dining", "Transport"])

    with patch("pipeline.forecast.RandomForestRegressor") as mock_rf_cls:
        service.run_forecast("real")

    mock_rf_cls.assert_not_called()


def test_run_forecast_response_shape(service, conn):
    seed_months(conn, 12, ["Food & Dining"])

    run = service.run_forecast("real")

    assert run["is_stale"] is False
    assert run["stale_reason"] is None
    offsets = {p["month_offset"] for p in run["predictions"]}
    assert offsets == {1, 2, 3}
    from config import CATEGORIES

    assert {p["category"] for p in run["predictions"]} == set(CATEGORIES)


def test_run_retention_two_consecutive_runs_create_two_distinct_rows(service, conn):
    seed_months(conn, 12, ["Food & Dining"])

    run_a = service.run_forecast("real")
    run_b = service.run_forecast("real")

    assert run_a["run_id"] != run_b["run_id"]
    repo = ForecastRepository(conn)
    assert repo.get_run(run_a["run_id"]) is not None
    assert repo.get_run(run_b["run_id"]) is not None
    assert service.get_latest("real")["run_id"] == run_b["run_id"]


# -- per-category availability --------------------------------------------------


def test_run_forecast_marks_absent_category_unavailable_not_zero(service, conn):
    # ML-D: the selected Naive recipe only needs ONE historical data point
    # to produce a prediction (unlike the retired RF path's 7-occurrence
    # rolling-window floor) — so a category is unavailable only when it has
    # ZERO recorded transactions at all, never merely "sparse".
    seed_months(conn, 12, ["Food & Dining"])

    run = service.run_forecast("real")

    healthcare = [p for p in run["predictions"] if p["category"] == "Healthcare"]
    assert healthcare  # present in the response, just unavailable
    assert all(not p["is_available"] for p in healthcare)
    assert all(p["unavailable_reason"] == "insufficient_history" for p in healthcare)
    assert all(p["predicted_amount"] is None for p in healthcare)

    food = [p for p in run["predictions"] if p["category"] == "Food & Dining"]
    assert all(p["is_available"] for p in food)
    assert all(p["predicted_amount"] is not None for p in food)


def test_run_forecast_a_single_recorded_month_is_available_under_naive(service, conn):
    seed_months(conn, 12, ["Food & Dining"])
    seed_sparse_category(conn, "Healthcare", ["2025-01"])

    run = service.run_forecast("real")

    healthcare = [p for p in run["predictions"] if p["category"] == "Healthcare"]
    assert all(p["is_available"] for p in healthcare)
    assert all(p["predicted_amount"] is not None for p in healthcare)
    # Naive/strategy N/A: identical predicted value reused at +1/+2/+3.
    assert len({p["predicted_amount"] for p in healthcare}) == 1


# -- effective category --------------------------------------------------------


def test_run_forecast_aggregates_by_effective_category(service, conn):
    txn_repo = TransactionRepository(conn)
    seed_months(conn, 12, ["Food & Dining"])
    # Correct one of the seeded rows to a different confirmed_category —
    # forecasting must use effective_category (confirmed, here), not
    # predicted_category.
    all_rows = txn_repo.list(data_mode="real")
    txn_repo.update(all_rows[0]["id"], {"confirmed_category": "Entertainment"})
    conn.commit()

    run = service.run_forecast("real")

    categories_with_predictions = {p["category"] for p in run["predictions"] if p["is_available"]}
    # Both the original (now 11 months) and corrected (1 month, unavailable)
    # categories appear in the response; Food & Dining still has enough
    # remaining history to be available.
    assert "Food & Dining" in categories_with_predictions


# -- mark_stale ----------------------------------------------------------------


def test_mark_stale_noop_when_no_run_exists(service, conn):
    seed_months(conn, 12, ["Food & Dining"])
    service.mark_stale("transaction_created")  # must not raise
    assert service.get_latest("real") is None


def test_mark_stale_flips_latest_non_stale_run_and_is_idempotent_on_reason(service, conn):
    seed_months(conn, 12, ["Food & Dining"])
    run = service.run_forecast("real")

    service.mark_stale("transaction_updated")
    status = ForecastRepository(conn).get_run(run["run_id"])
    assert status["is_stale"] == 1
    assert status["stale_reason"] == "transaction_updated"

    # Already stale -> calling again with a different reason must not
    # overwrite the original stale_reason (TRD Section 12.4: only the
    # latest NON-stale run is marked stale).
    service.mark_stale("transaction_deleted")
    status_again = ForecastRepository(conn).get_run(run["run_id"])
    assert status_again["stale_reason"] == "transaction_updated"


def test_mark_stale_only_affects_latest_run_not_older_ones(service, conn):
    seed_months(conn, 12, ["Food & Dining"])
    run_a = service.run_forecast("real")
    run_b = service.run_forecast("real")

    service.mark_stale("transaction_updated")

    repo = ForecastRepository(conn)
    assert repo.get_run(run_b["run_id"])["is_stale"] == 1
    assert repo.get_run(run_a["run_id"])["is_stale"] == 0
