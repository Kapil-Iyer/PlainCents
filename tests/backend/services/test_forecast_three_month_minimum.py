"""
Forecast eligibility: three completed months.

Three is the MATHEMATICAL MINIMUM for the selected production method -- a
3-month rolling mean needs exactly three months to fill one window. It is not
a finding that three months forecasts as accurately as six, nine or twelve;
the ML-C/ML-F history-length sensitivity experiments truncated to 6/9/12/18
months and never evaluated a three-month history at all.
"""
import pandas as pd
import pytest

from backend.api.errors import ForecastColdStartError
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.forecast_service import MONTHS_REQUIRED, ForecastService
from pipeline.forecast import MONTHS_REQUIRED as PIPELINE_MONTHS_REQUIRED
from pipeline.forecast import aggregate_monthly, train_and_predict


@pytest.fixture
def service(conn):
    return ForecastService(conn)


def _seed(conn, months: list[str], category="Food & Dining", amount=100.0):
    repo = TransactionRepository(conn)
    for i, month in enumerate(months):
        repo.create({
            "date": f"{month}-05",
            "merchant": f"NORTHSIDE PIZZA {i}",
            "amount": amount,
            "bank_source": "RBC",
            "predicted_category": category,
            "data_mode": "real",
            "dedup_key": f"seed{i}",
        })
    conn.commit()


# -- the boundary -------------------------------------------------------------


@pytest.mark.parametrize("months,expected", [
    ([], "cold_start"),
    (["2026-01"], "cold_start"),
    (["2026-01", "2026-02"], "cold_start"),
    (["2026-01", "2026-02", "2026-03"], "no_forecast_yet"),
])
def test_eligibility_boundary_at_three_months(service, conn, months, expected):
    _seed(conn, months)

    status = service.check_status("real")

    assert status["status"] == expected
    assert status["months_available"] == len(months)
    assert status["months_required"] == 3


def test_generation_is_rejected_below_three_months_and_persists_nothing(service, conn):
    _seed(conn, ["2026-01", "2026-02"])

    with pytest.raises(ForecastColdStartError):
        service.run_forecast("real")

    assert service.get_latest("real") is None


def test_generation_succeeds_at_exactly_three_months(service, conn):
    _seed(conn, ["2026-01", "2026-02", "2026-03"])

    run = service.run_forecast("real")

    assert run["months_available"] == 3
    assert any(p["is_available"] for p in run["predictions"])


def test_both_month_gates_agree():
    """The service checks MONTHS_REQUIRED and then calls aggregate_monthly(),
    which enforces its own threshold independently. If they ever diverge, a
    request the service accepted raises inside the pipeline as a 500."""
    assert MONTHS_REQUIRED == PIPELINE_MONTHS_REQUIRED == 3


# -- the arithmetic -----------------------------------------------------------


def test_three_month_mean_is_exactly_the_mean_of_the_last_three_months():
    monthly = pd.DataFrame([
        {"month": "2026-01", "category": "Food & Dining", "total_spend": 300.0},
        {"month": "2026-02", "category": "Food & Dining", "total_spend": 450.0},
        {"month": "2026-03", "category": "Food & Dining", "total_spend": 600.0},
    ])

    forecast = train_and_predict(monthly)

    food = forecast[forecast["category"] == "Food & Dining"]
    assert set(food["month_offset"]) == {1, 2, 3}
    # (300 + 450 + 600) / 3 = 450, identically across all three horizons: the
    # prediction never depends on a prior prediction, so there is nothing to
    # recurse.
    assert list(food["predicted_amount"]) == [450.0, 450.0, 450.0]


def test_forecast_is_deterministic_across_runs():
    monthly = pd.DataFrame([
        {"month": "2026-01", "category": "Transport", "total_spend": 120.0},
        {"month": "2026-02", "category": "Transport", "total_spend": 90.0},
        {"month": "2026-03", "category": "Transport", "total_spend": 150.0},
    ])

    first = train_and_predict(monthly)
    second = train_and_predict(monthly)

    pd.testing.assert_frame_equal(first, second)


def test_aggregate_monthly_accepts_exactly_three_months(conn):
    df = pd.DataFrame([
        {"date": "2026-01-05", "amount": 10.0, "category": "Food & Dining"},
        {"date": "2026-02-05", "amount": 20.0, "category": "Food & Dining"},
        {"date": "2026-03-05", "amount": 30.0, "category": "Food & Dining"},
    ])

    monthly = aggregate_monthly(df)

    assert monthly["month"].nunique() == 3


def test_aggregate_monthly_rejects_two_months():
    df = pd.DataFrame([
        {"date": "2026-01-05", "amount": 10.0, "category": "Food & Dining"},
        {"date": "2026-02-05", "amount": 20.0, "category": "Food & Dining"},
    ])

    with pytest.raises(ValueError, match="3 months minimum"):
        aggregate_monthly(df)


# -- effective category flows into the forecast -------------------------------


def test_forecast_aggregates_on_effective_category(service, conn):
    """A user correction must move the forecast, not just the transaction
    list -- otherwise the two views of the same money disagree."""
    repo = TransactionRepository(conn)
    for i, month in enumerate(["2026-01", "2026-02", "2026-03"]):
        repo.create({
            "date": f"{month}-05",
            "merchant": "CAREWELL PHARMACY",
            "amount": 100.0,
            "bank_source": "RBC",
            "predicted_category": "Food & Dining",
            "confirmed_category": "Healthcare",   # the user's own decision
            "data_mode": "real",
            "dedup_key": f"c{i}",
        })
    conn.commit()

    run = service.run_forecast("real")
    by_category = {
        p["category"]: p for p in run["predictions"] if p["month_offset"] == 1
    }

    assert by_category["Healthcare"]["predicted_amount"] == 100.0
    assert by_category["Healthcare"]["is_available"] is True
    # Nothing was ever spent in the MODEL's category, so it must report no
    # history rather than a fabricated $0 forecast.
    assert by_category["Food & Dining"]["is_available"] is False
    assert by_category["Food & Dining"]["predicted_amount"] is None


# -- staleness ----------------------------------------------------------------


def test_correcting_a_category_marks_the_forecast_stale_and_refresh_clears_it(service, conn):
    from pathlib import Path as _Path

    from backend.services.categorization_service import CategorizationService
    from backend.services.transaction_service import TransactionService

    fixture = (_Path(__file__).resolve().parent.parent.parent
               / "fixtures" / "categorizer_model_test.pkl")

    _seed(conn, ["2026-01", "2026-02", "2026-03"])
    first = service.run_forecast("real")
    assert first["is_stale"] is False

    txn = TransactionRepository(conn).list(data_mode="real")[0]
    TransactionService(conn, CategorizationService(fixture)).update(
        txn["id"], {"confirmed_category": "Healthcare"})

    assert service.check_status("real")["is_stale"] is True

    refreshed = service.run_forecast("real")
    assert refreshed["is_stale"] is False
    assert refreshed["run_id"] != first["run_id"]
    # And the refreshed run reflects the correction.
    healthcare = next(
        p for p in refreshed["predictions"]
        if p["category"] == "Healthcare" and p["month_offset"] == 1
    )
    assert healthcare["is_available"] is True
