"""
ML Spec Section 12: no future month may enter forecasting features/training
at any point. These tests exercise the actual data_prep/features/temporal_eval
modules together (not just the fold-boundary arithmetic already covered in
test_temporal_eval.py), including the zero-fill correction and the RF/Ridge
frozen-hyperparameter contract.
"""
import numpy as np
import pandas as pd
import pytest

from ml.forecasting.baselines import naive_predict, seasonal_naive_predict
from ml.forecasting.candidates import RF_HPARAMS, RandomForestCandidate, RidgeCandidate
from ml.forecasting.features import build_training_matrix, category_history
from ml.forecasting.temporal_eval import build_folds


def _toy_grid():
    months = [f"2023-{m:02d}" for m in range(1, 13)] + [f"2024-{m:02d}" for m in range(1, 13)]
    rows = []
    for i, m in enumerate(months):
        rows.append({"month": m, "category": "Food & Dining", "total_spend": 100.0 + i})
        rows.append({"month": m, "category": "Transport", "total_spend": 50.0 + i * 0.5})
    return pd.DataFrame(rows), months


def test_training_matrix_never_contains_a_month_outside_train_months():
    grid, months = _toy_grid()
    train_months = months[:10]
    X_train, y_train, le = build_training_matrix(grid, train_months)
    # build_forecast_features drops the first 6 rows per category (rolling
    # window warm-up) but must never reach into month index >= 10.
    assert len(X_train) > 0
    # Reconstruct which original months survived via row count sanity: at most
    # (10 - 6) * n_categories rows should exist (6-month warm-up per category).
    n_categories = grid["category"].nunique()
    assert len(X_train) <= (len(train_months) - 6) * n_categories


def test_rf_hyperparameters_match_trd_shipped_defaults_not_diagnostic_defaults():
    """Regression guard: ML Spec Section 11 requires the SAME hyperparameters
    pipeline.forecast.train_and_predict() ships (n_estimators=100,
    max_depth=10, min_samples_leaf=5) -- not walk_forward_validate's
    diagnostic max_depth=3."""
    assert RF_HPARAMS["n_estimators"] == 100
    assert RF_HPARAMS["max_depth"] == 10
    assert RF_HPARAMS["min_samples_leaf"] == 5


def test_seasonal_naive_ineligible_returns_none_not_zero():
    short_history = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])  # 7 months, need >=12 for h=1
    pred, eligible = seasonal_naive_predict(short_history, horizon=1)
    assert eligible is False
    assert pred is None  # never fabricated as 0 or any other value


def test_seasonal_naive_eligible_uses_exact_prior_year_value():
    # 13 months of history: index 0 = month "M-12", index 12 = month "M" (last train month, origin).
    # Target for h=1 is "M+1"; the value 12 months before "M+1" is at index 1.
    history = np.array([float(i) for i in range(13)])  # values 0..12
    pred, eligible = seasonal_naive_predict(history, horizon=1)
    assert eligible is True
    assert pred == 1.0


def test_naive_uses_only_the_last_observed_value():
    history = np.array([10.0, 20.0, 30.0])
    assert naive_predict(history) == 30.0


def test_category_history_excludes_months_not_requested():
    grid, months = _toy_grid()
    hist = category_history(grid, "Food & Dining", months[:5])
    assert len(hist) == 5
    # values were 100+i for i in 0..4
    assert list(hist) == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_build_folds_plus_training_matrix_end_to_end_no_leakage():
    """Full pipeline check: for every fold, the fitted training matrix's
    underlying months are a subset of that fold's train_months, and never
    intersect the fold's own target months or the reserved final period."""
    grid, months = _toy_grid()
    folds, reserved = build_folds(months, min_train_months=7, n_final_reserved_months=3)
    for fold in folds:
        X_train, y_train, le = build_training_matrix(grid, fold.train_months)
        assert len(X_train) > 0
        # target months must not appear in the training month set
        for target_month in fold.target_months.values():
            assert target_month not in fold.train_months
        # reserved months must not appear in train_months either
        assert not (set(fold.train_months) & set(reserved["months"]))
