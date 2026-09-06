"""
AnalyticsService tests (ML-G).

The properties that matter here are not "the endpoint returns 200" but:
  * category grouping follows effective_category, so a manual correction
    moves the charts,
  * merchant grouping follows the stable identity, so one merchant is one
    row rather than one row per card suffix,
  * the movers decomposition is additive, so it reads as an explanation,
  * DEMO and REAL never mix,
  * and forecast-vs-actual refuses to show anything until a genuine
    historical snapshot exists.
"""
from datetime import date

import pytest

from backend.repositories.forecast_repository import ForecastRepository
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.analytics_service import AnalyticsService

TODAY = date(2026, 3, 15)


@pytest.fixture
def service(conn):
    return AnalyticsService(conn)


@pytest.fixture
def repo(conn):
    return TransactionRepository(conn)


def _txn(repo, conn, *, date_, merchant, amount, predicted, confirmed=None,
         bank="RBC", mode="real", key=None):
    n = _txn.counter = getattr(_txn, "counter", 0) + 1
    payload = {
        "date": date_,
        "merchant": merchant,
        "amount": amount,
        "bank_source": bank,
        "predicted_category": predicted,
        "confirmed_category": confirmed,
        "data_mode": mode,
        "dedup_key": f"k{n}",
    }
    if key is not None:
        payload["merchant_key"] = key
    tid = repo.create(payload)
    conn.commit()
    return tid


# -- category trend -----------------------------------------------------------


def test_category_trend_zero_fills_and_uses_effective_category(service, repo, conn):
    _txn(repo, conn, date_="2026-01-10", merchant="NORTHSIDE PIZZA", amount=20.0,
         predicted="Food & Dining")
    _txn(repo, conn, date_="2026-03-02", merchant="CAREWELL PHARMACY", amount=50.0,
         predicted="Food & Dining", confirmed="Healthcare")

    result = service.category_trend("real", months=3, reference_date=TODAY)

    assert result["months"] == ["2026-01", "2026-02", "2026-03"]
    # The corrected row is counted under the USER's category, not the model's.
    assert result["categories"] == ["Food & Dining", "Healthcare"]
    by_month = {p["month"]: p for p in result["points"]}
    assert by_month["2026-01"]["by_category"]["Food & Dining"] == 20.0
    # A month with genuinely no spend in a category is a real zero.
    assert by_month["2026-02"]["by_category"]["Food & Dining"] == 0.0
    assert by_month["2026-02"]["total_spend"] == 0.0
    assert by_month["2026-03"]["by_category"]["Healthcare"] == 50.0


def test_category_trend_excludes_months_outside_the_window(service, repo, conn):
    _txn(repo, conn, date_="2025-01-10", merchant="OLD DINER", amount=99.0,
         predicted="Food & Dining")
    _txn(repo, conn, date_="2026-03-10", merchant="NEW DINER", amount=10.0,
         predicted="Food & Dining")

    result = service.category_trend("real", months=2, reference_date=TODAY)

    assert result["months"] == ["2026-02", "2026-03"]
    assert sum(p["total_spend"] for p in result["points"]) == 10.0


def test_category_trend_is_empty_but_valid_with_no_data(service):
    result = service.category_trend("real", months=3, reference_date=TODAY)

    assert result["categories"] == []
    assert len(result["points"]) == 3
    assert all(p["total_spend"] == 0.0 for p in result["points"])


def test_category_trend_returns_nothing_in_empty_mode(service, repo, conn):
    _txn(repo, conn, date_="2026-03-10", merchant="NORTHSIDE PIZZA", amount=10.0,
         predicted="Food & Dining")

    result = service.category_trend(None, months=3, reference_date=TODAY)

    assert result["categories"] == []
    assert all(p["total_spend"] == 0.0 for p in result["points"])


# -- top merchants ------------------------------------------------------------


def test_top_merchants_groups_by_stable_identity_not_raw_text(service, repo, conn):
    """Three card-suffix variants of one pharmacy are ONE merchant.

    Grouping by raw description would split it into three single-transaction
    rows and hide it from the ranking entirely -- the exact reason merchant
    analytics needed the stable key before they were worth building.
    """
    for suffix, amount in (("4821", 30.0), ("9137", 25.0), ("0284", 45.0)):
        _txn(repo, conn, date_="2026-03-02",
             merchant=f"VISA DEBIT PURCHASE - {suffix} CAREWELL PHARMACY",
             amount=amount, predicted="Healthcare")

    result = service.top_merchants("real", limit=5, months=3, reference_date=TODAY)

    assert len(result["items"]) == 1
    item = result["items"][0]
    assert item["transaction_count"] == 3
    assert item["total_spend"] == 100.0
    assert item["average_transaction"] == 33.33
    assert item["category"] == "Healthcare"
    assert item["pct_of_total"] == 100.0
    assert result["distinct_merchants"] == 1


def test_top_merchants_ranks_by_spend_and_reports_concentration(service, repo, conn):
    _txn(repo, conn, date_="2026-03-02", merchant="OAKFIELD RESIDENCES RENT",
         amount=1800.0, predicted="Rent & Utilities")
    _txn(repo, conn, date_="2026-03-03", merchant="GREENLEAF SUPERMARKET",
         amount=200.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-03-04", merchant="SUNRISE CAFE",
         amount=8.0, predicted="Food & Dining")

    result = service.top_merchants("real", limit=2, months=3, reference_date=TODAY)

    assert [i["merchant"] for i in result["items"]] == [
        "OAKFIELD RESIDENCES RENT", "GREENLEAF SUPERMARKET"]
    assert result["distinct_merchants"] == 3
    # Top 2 of 2008.00 total.
    assert result["top_n_share_pct"] == pytest.approx(99.6, abs=0.1)


def test_top_merchants_uses_effective_category_for_its_label(service, repo, conn):
    _txn(repo, conn, date_="2026-03-02", merchant="VALUEMART DEPT STORE", amount=60.0,
         predicted="Shopping", confirmed="Food & Dining")

    result = service.top_merchants("real", limit=5, months=3, reference_date=TODAY)

    assert result["items"][0]["category"] == "Food & Dining"


# -- category movers ----------------------------------------------------------


def test_category_movers_decomposition_is_additive(service, repo, conn):
    """The per-category changes must sum EXACTLY to the total change --
    that additivity is what makes this an explanation of the month's
    movement rather than a second unrelated chart."""
    _txn(repo, conn, date_="2026-02-05", merchant="A DINER", amount=100.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-02-06", merchant="B TRANSIT FARE", amount=50.0, predicted="Transport")
    _txn(repo, conn, date_="2026-03-05", merchant="A DINER", amount=180.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-03-07", merchant="C PHARMACY", amount=25.0, predicted="Healthcare")

    result = service.category_movers("real", reference_date=TODAY)

    assert result["current_month"] == "2026-03"
    assert result["previous_month"] == "2026-02"
    assert result["total_current"] == 205.0
    assert result["total_previous"] == 150.0
    assert result["total_change"] == 55.0
    assert sum(m["change"] for m in result["movers"]) == pytest.approx(result["total_change"])


def test_category_movers_sorted_by_absolute_movement(service, repo, conn):
    _txn(repo, conn, date_="2026-02-05", merchant="A DINER", amount=100.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-02-06", merchant="B TRANSIT FARE", amount=200.0, predicted="Transport")
    _txn(repo, conn, date_="2026-03-05", merchant="A DINER", amount=110.0, predicted="Food & Dining")

    result = service.category_movers("real", reference_date=TODAY)

    # Transport fell 200; Food & Dining rose 10. Biggest movement first,
    # regardless of direction.
    assert [m["category"] for m in result["movers"]] == ["Transport", "Food & Dining"]
    assert result["movers"][0]["change"] == -200.0


def test_category_movers_reports_no_percentage_against_a_zero_baseline(service, repo, conn):
    _txn(repo, conn, date_="2026-03-05", merchant="C PHARMACY", amount=25.0, predicted="Healthcare")

    result = service.category_movers("real", reference_date=TODAY)

    healthcare = next(m for m in result["movers"] if m["category"] == "Healthcare")
    assert healthcare["previous"] == 0.0
    # There is no meaningful percentage of nothing -- None, never a made-up 100%.
    assert healthcare["change_pct"] is None


def test_category_movers_caps_previous_month_at_the_same_elapsed_day(service, repo, conn):
    """Early in a month, the previous month's total (and its per-category
    breakdown) must be capped at the SAME day-of-month as the current
    (partial) month -- a late-previous-month transaction must not inflate
    `total_previous`/`previous` beyond what the current month has had a
    chance to match yet. Reference date here is the 5th of March."""
    early_march = date(2026, 3, 5)
    _txn(repo, conn, date_="2026-03-03", merchant="A DINER", amount=10.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-02-04", merchant="A DINER", amount=8.0, predicted="Food & Dining")   # day 4, comparable
    _txn(repo, conn, date_="2026-02-20", merchant="B TRANSIT FARE", amount=200.0, predicted="Transport")  # day 20, NOT comparable

    result = service.category_movers("real", reference_date=early_march)

    assert result["comparable_day"] == 5
    assert result["total_current"] == 10.0
    assert result["total_previous"] == 8.0  # the day-20 Transport row must be excluded
    assert "Transport" not in {m["category"] for m in result["movers"] if m["previous"] > 0}
    assert sum(m["change"] for m in result["movers"]) == pytest.approx(result["total_change"])


def test_category_movers_historical_month_uses_full_months_uncapped(service, repo, conn):
    """`analysis_month` selecting a fully-completed past month must compare
    FULL calendar months on both sides, regardless of how far "today" is
    into a later month -- and additivity must still hold."""
    _txn(repo, conn, date_="2026-02-05", merchant="A DINER", amount=100.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-03-05", merchant="A DINER", amount=180.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-03-25", merchant="C PHARMACY", amount=25.0, predicted="Healthcare")

    # "Today" is well past March -- April 6 -- so March is a fully-completed
    # historical month, not the still-in-progress current one.
    result = service.category_movers("real", reference_date=date(2026, 4, 6), analysis_month="2026-03")

    assert result["current_month"] == "2026-03"
    assert result["previous_month"] == "2026-02"
    assert result["is_current_incomplete"] is False
    assert result["total_current"] == 205.0  # both March rows, including day 25
    assert result["total_previous"] == 100.0
    assert result["total_change"] == 105.0
    assert sum(m["change"] for m in result["movers"]) == pytest.approx(result["total_change"])


def test_category_movers_explicit_current_month_still_incomplete(service, repo, conn):
    _txn(repo, conn, date_="2026-03-05", merchant="A DINER", amount=10.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-03-20", merchant="A DINER", amount=999.0, predicted="Food & Dining")

    result = service.category_movers("real", reference_date=TODAY, analysis_month="2026-03")

    assert result["is_current_incomplete"] is True
    assert result["total_current"] == 10.0  # day-20 row still excluded, "today" is the 15th


# -- spend pace ---------------------------------------------------------------


def test_spend_pace_accumulates_and_stops_at_today(service, repo, conn):
    _txn(repo, conn, date_="2026-02-05", merchant="A DINER", amount=40.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-02-20", merchant="A DINER", amount=60.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-03-03", merchant="A DINER", amount=30.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-03-10", merchant="A DINER", amount=20.0, predicted="Food & Dining")

    result = service.spend_pace("real", reference_date=TODAY)

    assert result["day_of_month"] == 15
    assert result["current_to_date"] == 50.0
    assert result["previous_same_point"] == 40.0
    assert result["difference"] == 10.0

    by_day = {p["day"]: p for p in result["points"]}
    assert by_day[3]["current_cumulative"] == 30.0
    assert by_day[10]["current_cumulative"] == 50.0
    # Past today the current month has not happened -- a gap, not a flat line.
    assert by_day[16]["current_cumulative"] is None
    assert by_day[20]["previous_cumulative"] == 100.0


def test_spend_pace_handles_no_history_at_all(service):
    result = service.spend_pace("real", reference_date=TODAY)

    assert result["current_to_date"] == 0.0
    assert result["previous_same_point"] == 0.0
    assert result["points"]


def test_spend_pace_comparable_day_caps_when_previous_month_is_shorter(service, repo, conn):
    """comparable_day must differ from day_of_month when the previous month
    is shorter (March 31 vs February) -- a UI label naming the previous
    period's day range needs comparable_day, never day_of_month, or it
    would claim a nonexistent 'Feb 1-31'."""
    _txn(repo, conn, date_="2026-03-31", merchant="A DINER", amount=15.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-02-28", merchant="A DINER", amount=9.0, predicted="Food & Dining")

    result = service.spend_pace("real", reference_date=date(2026, 3, 31))

    assert result["day_of_month"] == 31
    assert result["comparable_day"] == 28
    assert result["previous_same_point"] == 9.0


def test_spend_pace_day_1_compares_a_single_day_on_each_side(service, repo, conn):
    _txn(repo, conn, date_="2026-06-01", merchant="A DINER", amount=12.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-05-01", merchant="A DINER", amount=7.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-05-15", merchant="A DINER", amount=500.0, predicted="Food & Dining")

    result = service.spend_pace("real", reference_date=date(2026, 6, 1))

    assert result["day_of_month"] == 1
    assert result["comparable_day"] == 1
    assert result["current_to_date"] == 12.0
    # The day-15 May transaction must NOT be pulled into "previous_same_point".
    assert result["previous_same_point"] == 7.0


def test_spend_pace_historical_month_runs_both_curves_to_full_length(service, repo, conn):
    """Selecting a fully-completed historical month: both the current and
    previous curves run to their own full real length (no "today" to stop
    at), and the scalar to-date figures are full-month totals on both
    sides -- not the day-aligned "comparable point" semantics used for the
    still-in-progress month."""
    _txn(repo, conn, date_="2026-02-05", merchant="A DINER", amount=40.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-02-28", merchant="A DINER", amount=60.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-03-03", merchant="A DINER", amount=30.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-03-31", merchant="A DINER", amount=20.0, predicted="Food & Dining")

    # "Today" is well past March -- April 6 -- so March is fully complete.
    result = service.spend_pace("real", reference_date=date(2026, 4, 6), analysis_month="2026-03")

    assert result["is_current_incomplete"] is False
    assert result["day_of_month"] == 31  # March's own full length
    assert result["current_to_date"] == 50.0  # full March total
    assert result["previous_same_point"] == 100.0  # full February total
    assert result["difference"] == -50.0

    by_day = {p["day"]: p for p in result["points"]}
    # The current curve reaches all the way to March 31 -- no gap, since the
    # month is fully complete (unlike the still-in-progress regime).
    assert by_day[31]["current_cumulative"] == 50.0
    assert by_day[28]["previous_cumulative"] == 100.0


def test_spend_pace_explicit_current_month_still_uses_today_semantics(service, repo, conn):
    _txn(repo, conn, date_="2026-03-03", merchant="A DINER", amount=30.0, predicted="Food & Dining")
    _txn(repo, conn, date_="2026-03-20", merchant="A DINER", amount=999.0, predicted="Food & Dining")

    result = service.spend_pace("real", reference_date=TODAY, analysis_month="2026-03")

    assert result["is_current_incomplete"] is True
    assert result["day_of_month"] == 15
    assert result["current_to_date"] == 30.0  # day-20 row not yet "today"


# -- demo / real isolation ----------------------------------------------------


def test_demo_and_real_rows_never_mix(service, repo, conn):
    _txn(repo, conn, date_="2026-03-05", merchant="REAL DINER", amount=10.0,
         predicted="Food & Dining", mode="real")
    _txn(repo, conn, date_="2026-03-05", merchant="DEMO DINER", amount=999.0,
         predicted="Food & Dining", mode="demo")

    real = service.category_trend("real", months=1, reference_date=TODAY)
    demo = service.category_trend("demo", months=1, reference_date=TODAY)

    assert real["points"][0]["total_spend"] == 10.0
    assert demo["points"][0]["total_spend"] == 999.0

    real_merchants = service.top_merchants("real", months=1, reference_date=TODAY)
    assert [i["merchant"] for i in real_merchants["items"]] == ["REAL DINER"]


# -- forecast accuracy: genuine snapshots only --------------------------------


def test_forecast_accuracy_unavailable_without_snapshots(service):
    result = service.forecast_accuracy("real", reference_date=TODAY)

    assert result["available"] is False
    assert result["reason"] == "no_snapshots_yet"
    assert result["items"] == []


def test_forecast_accuracy_ignores_hindsight_predictions(service, conn, repo):
    """A run generated DURING or AFTER the month it 'predicted' is not a
    forecast, it is hindsight. It must never be presented as history."""
    frepo = ForecastRepository(conn)
    run_id = frepo.create_run({"months_available": 6, "data_mode": "real",
                               "model_impl_version": "rolling_mean_3_v1"})
    # Generated in March, "predicting" January -- after the fact.
    conn.execute("UPDATE forecast_runs SET generated_at = ? WHERE id = ?",
                 ("2026-03-01 10:00:00", run_id))
    frepo.save_predictions(run_id, [{
        "category": "Food & Dining", "forecast_month": "2026-01",
        "month_offset": 1, "predicted_amount": 100.0, "is_available": True,
        "unavailable_reason": None,
    }])
    conn.commit()

    result = service.forecast_accuracy("real", reference_date=TODAY)

    assert result["available"] is False


def test_forecast_accuracy_uses_genuine_prior_snapshots(service, conn, repo):
    frepo = ForecastRepository(conn)
    run_id = frepo.create_run({"months_available": 6, "data_mode": "real",
                               "model_impl_version": "rolling_mean_3_v1"})
    # Generated in January, predicting February. February has since ended.
    conn.execute("UPDATE forecast_runs SET generated_at = ? WHERE id = ?",
                 ("2026-01-20 09:00:00", run_id))
    frepo.save_predictions(run_id, [{
        "category": "Food & Dining", "forecast_month": "2026-02",
        "month_offset": 1, "predicted_amount": 120.0, "is_available": True,
        "unavailable_reason": None,
    }])
    conn.commit()

    _txn(repo, conn, date_="2026-02-10", merchant="A DINER", amount=100.0,
         predicted="Food & Dining")

    result = service.forecast_accuracy("real", reference_date=TODAY)

    assert result["available"] is True
    assert result["months_covered"] == ["2026-02"]
    item = result["items"][0]
    assert item["predicted"] == 120.0
    assert item["actual"] == 100.0
    assert item["error"] == 20.0
    assert result["wape"] == pytest.approx(0.2)


def test_forecast_accuracy_compares_against_effective_category(service, conn, repo):
    """A user correction changes what actually landed in a category, and the
    accuracy view must reflect that -- otherwise it scores the forecast
    against numbers the rest of the app no longer shows."""
    frepo = ForecastRepository(conn)
    run_id = frepo.create_run({"months_available": 6, "data_mode": "real",
                               "model_impl_version": "rolling_mean_3_v1"})
    conn.execute("UPDATE forecast_runs SET generated_at = ? WHERE id = ?",
                 ("2026-01-20 09:00:00", run_id))
    frepo.save_predictions(run_id, [{
        "category": "Healthcare", "forecast_month": "2026-02",
        "month_offset": 1, "predicted_amount": 50.0, "is_available": True,
        "unavailable_reason": None,
    }])
    conn.commit()

    _txn(repo, conn, date_="2026-02-10", merchant="CAREWELL PHARMACY", amount=60.0,
         predicted="Food & Dining", confirmed="Healthcare")

    result = service.forecast_accuracy("real", reference_date=TODAY)

    item = next(i for i in result["items"] if i["category"] == "Healthcare")
    assert item["actual"] == 60.0
