"""
pipeline.forecast.train_and_predict() tests (Build Plan Phase 7, item 8;
ML-D/ML-F Production Integration): verifies the interactive path implements
the ML-F selected 3-month rolling mean recipe exactly — same value reused at
every horizon, no RandomForestRegressor fit, no walk-forward/GridSearchCV —
and correctly marks a category with zero recorded history unavailable
rather than fabricating a $0 forecast (TRD Section 12.5). Pure pipeline
test — no DB, no services, mirroring
tests/backend/services/test_ingest_bytes.py's placement convention for a
Phase 7 addition to a V1 pipeline module.

fit_and_forecast()/walk_forward_validate()/GridSearchCV usage in
pipeline/forecast.py are not touched or retested here — tests/test_pipeline.py
(empty today, per ML Spec Section 1.1's audit finding) is V1's own file and
this Phase adds no scope there.
"""
from unittest.mock import patch

import pandas as pd

from config import CATEGORIES
from pipeline.forecast import aggregate_monthly, train_and_predict

FULL_YEAR = list(range(12))


def _raw_rows(month_indices: list[int], category: str, amount: float, start_year=2025, start_month=1) -> list[dict]:
    rows = []
    for i in month_indices:
        total = (start_month - 1) + i
        year = start_year + total // 12
        month = total % 12 + 1
        rows.append({"date": f"{year:04d}-{month:02d}-10", "amount": amount, "category": category})
    return rows


def _monthly_df(category_month_indices: dict[str, list[int]]) -> pd.DataFrame:
    rows = []
    for i, (cat, months) in enumerate(category_month_indices.items()):
        rows.extend(_raw_rows(months, cat, amount=100.0 + i))
    return aggregate_monthly(pd.DataFrame(rows))


def test_train_and_predict_never_calls_walk_forward_or_gridsearch():
    monthly_df = _monthly_df({"Food & Dining": FULL_YEAR, "Transport": FULL_YEAR})

    with patch("pipeline.forecast.walk_forward_validate") as mock_wfv, \
         patch("pipeline.forecast.GridSearchCV") as mock_gscv:
        train_and_predict(monthly_df)

    mock_wfv.assert_not_called()
    mock_gscv.assert_not_called()


def test_train_and_predict_never_fits_a_random_forest():
    # ML-F selected 3-month rolling mean has no fitting step at all — the
    # production path must never construct/fit a RandomForestRegressor
    # (ML-D/ML-F Production Integration).
    monthly_df = _monthly_df({"Food & Dining": FULL_YEAR, "Transport": FULL_YEAR})

    with patch("pipeline.forecast.RandomForestRegressor") as mock_rf_cls:
        train_and_predict(monthly_df)

    mock_rf_cls.assert_not_called()


def test_train_and_predict_returns_three_horizons_for_every_category():
    monthly_df = _monthly_df({"Food & Dining": FULL_YEAR, "Transport": FULL_YEAR})

    result = train_and_predict(monthly_df)

    assert set(result["month_offset"].unique()) == {1, 2, 3}
    assert set(result["category"].unique()) == set(CATEGORIES)
    assert len(result) == 3 * len(CATEGORIES)


def test_train_and_predict_forecast_months_are_sequential_after_last_history_month():
    monthly_df = _monthly_df({"Food & Dining": FULL_YEAR})

    result = train_and_predict(monthly_df)

    food = result[result["category"] == "Food & Dining"].sort_values("month_offset")
    # 12 months starting 2025-01 -> last history month is 2025-12.
    assert list(food["forecast_month"]) == ["2026-01", "2026-02", "2026-03"]


# -- ML-F selected 3-month rolling mean: identical across horizons ------------


def test_train_and_predict_predicts_the_last_observed_month_at_every_horizon():
    # This fixture's monthly total_spend is constant across all 12 months
    # (see _raw_rows: one fixed `amount` per category, every month) — the
    # mean of the last 3 identical months equals that same constant, so this
    # case cannot by itself distinguish rolling-mean from the retired
    # lag-1 Naive recipe. See
    # test_train_and_predict_uses_mean_of_last_3_months_not_just_lag_1 below
    # for a fixture that does.
    monthly_df = _monthly_df({"Food & Dining": FULL_YEAR})

    result = train_and_predict(monthly_df)

    last_actual = monthly_df[monthly_df["category"] == "Food & Dining"].sort_values("month")["total_spend"].iloc[-1]
    food = result[result["category"] == "Food & Dining"].sort_values("month_offset")
    assert list(food["predicted_amount"]) == [round(float(last_actual), 2)] * 3
    # Same value reused at every horizon (selected strategy "N/A", no
    # per-horizon recomputation or recursion) -- true of rolling_mean_3 for
    # exactly the same reason it was true of Naive: the prediction never
    # depends on a prior *prediction*.
    assert len(set(food["predicted_amount"])) == 1


def test_train_and_predict_uses_mean_of_last_3_months_not_just_lag_1():
    # ml/forecasting/baselines.py::rolling_mean_predict(window=3): predicted
    # spend = mean of the category's most recent 3 observed months -- NOT
    # simply the single last month (the retired ML-C Naive recipe). Amounts
    # here vary month to month specifically so this distinction is testable.
    months = [f"2025-{m:02d}" for m in range(1, 13)]
    amounts = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 30.0, 60.0, 90.0]
    monthly_df = pd.DataFrame({"month": months, "category": "Food & Dining", "total_spend": amounts})

    result = train_and_predict(monthly_df)

    food = result[result["category"] == "Food & Dining"].sort_values("month_offset")
    expected = round((30.0 + 60.0 + 90.0) / 3, 2)  # mean of the last 3 months, NOT lag-1's 90.0
    assert list(food["predicted_amount"]) == [expected] * 3
    assert expected != 90.0  # would be 90.0 under lag-1 Naive -- proves this is genuinely rolling-mean behavior


def test_train_and_predict_a_single_observed_month_is_available():
    # rolling_mean_3 only needs one historical data point (mean of "the last
    # 3" degrades gracefully to that single value) — unlike the retired RF
    # path's 7-occurrence rolling-window floor. Transport pads the overall
    # grid to aggregate_monthly's own 6-unique-month floor; Food & Dining
    # itself has exactly one recorded month.
    monthly_df = _monthly_df({"Transport": FULL_YEAR, "Food & Dining": [0]})

    result = train_and_predict(monthly_df)

    last_actual = monthly_df[monthly_df["category"] == "Food & Dining"]["total_spend"].iloc[-1]
    food = result[result["category"] == "Food & Dining"]
    assert food["is_available"].all()
    assert (food["predicted_amount"] == round(float(last_actual), 2)).all()


# -- per-category availability -------------------------------------------------


def test_train_and_predict_marks_absent_category_unavailable_not_fabricated_zero():
    # Food & Dining has recorded history; every other CATEGORIES member is
    # entirely absent from monthly_df (never recorded a transaction).
    monthly_df = _monthly_df({"Food & Dining": FULL_YEAR})

    result = train_and_predict(monthly_df)

    food = result[result["category"] == "Food & Dining"]
    assert food["is_available"].all()
    assert food["predicted_amount"].notna().all()
    assert (food["unavailable_reason"].isna()).all()

    absent = result[result["category"] != "Food & Dining"]
    assert not absent["is_available"].any()
    assert (absent["unavailable_reason"] == "insufficient_history").all()
    # Never fabricated as $0 — genuinely absent (None/NaN), not zero.
    assert absent["predicted_amount"].isna().all()
    assert not (absent["predicted_amount"] == 0).any()


def test_train_and_predict_marks_all_categories_unavailable_when_monthly_df_is_empty_of_them():
    # Food & Dining has a full year of history; Transport has only a single
    # recorded month. Both remain available under rolling_mean_3 (>=1
    # occurrence is sufficient) — only categories with ZERO recorded
    # occurrences are unavailable, unlike the retired RF path's 7-occurrence
    # floor.
    monthly_df = _monthly_df({"Food & Dining": FULL_YEAR, "Transport": [0]})

    result = train_and_predict(monthly_df)

    present = result[result["category"].isin(["Food & Dining", "Transport"])]
    assert present["is_available"].all()

    absent = result[~result["category"].isin(["Food & Dining", "Transport"])]
    assert not absent["is_available"].any()
    assert (absent["unavailable_reason"] == "insufficient_history").all()
    assert absent["predicted_amount"].isna().all()
