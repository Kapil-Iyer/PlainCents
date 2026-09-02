"""
pipeline.forecast.train_and_predict() tests (Build Plan Phase 7, item 8):
verifies the interactive path never calls walk_forward_validate/GridSearchCV,
generates 3-horizon predictions for every category, and correctly marks a
too-sparse category unavailable rather than fabricating a $0 forecast (TRD
Section 12.5). Pure pipeline test — no DB, no services, mirroring
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


def test_train_and_predict_marks_sparse_category_unavailable_not_fabricated_zero():
    # Food & Dining: 12 occurrences, comfortably survives the 7-occurrence
    # rolling/lag floor. Healthcare: only 3 occurrences, does not survive.
    # Every other CATEGORIES member is entirely absent from monthly_df.
    monthly_df = _monthly_df({"Food & Dining": FULL_YEAR, "Healthcare": [0, 1, 2]})

    result = train_and_predict(monthly_df)

    food = result[result["category"] == "Food & Dining"]
    assert food["is_available"].all()
    assert food["predicted_amount"].notna().all()
    assert (food["unavailable_reason"].isna()).all()

    unavailable = result[result["category"] != "Food & Dining"]
    assert not unavailable["is_available"].any()
    assert (unavailable["unavailable_reason"] == "insufficient_history").all()
    # Never fabricated as $0 — genuinely absent (None/NaN), not zero.
    assert unavailable["predicted_amount"].isna().all()
    assert not (unavailable["predicted_amount"] == 0).any()


def test_train_and_predict_marks_all_categories_unavailable_when_none_reach_seven():
    # Two categories, each in a disjoint set of 6 months (< 7 occurrences),
    # together covering 12 distinct calendar months overall (satisfying
    # aggregate_monthly's own floor) -- neither survives dropna individually.
    monthly_df = _monthly_df({"Food & Dining": list(range(0, 6)), "Transport": list(range(6, 12))})

    result = train_and_predict(monthly_df)

    assert not result["is_available"].any()
    assert result["predicted_amount"].isna().all()
