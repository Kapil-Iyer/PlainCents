"""
Rolling/lag/calendar feature construction for the forecasting bake-off.

Functionally mirrors pipeline/forecast.py's feature definitions (Section
1.2: rolling_3m_avg, rolling_6m_avg, rolling_std ddof=1, lag_1_spend,
is_december, is_summer) but is an independent implementation local to ml/ --
pipeline/forecast.py itself is never imported for the per-fold TRAIN-only
model fitting done here (Production Isolation), except where explicitly
noted (`build_training_matrix` reuses pipeline.forecast.build_forecast_features
read-only, exactly as pipeline/forecast.py's own walk_forward_validate
already does, per ML Spec Section 12's instruction to reuse that
proven-correct logic rather than reinvent it for the *training* matrix).
The single-row-at-inference-time feature construction below (used by both
multi-step strategies, Section 11.1) IS reimplemented locally because it
must support the recursive strategy's "append a prediction as if it were
an observation" extension, which pipeline/forecast.py has no equivalent of.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import CATEGORIES
from pipeline.forecast import build_forecast_features

FEATURE_COLS = [
    "month_num", "category_encoded", "rolling_3m_avg",
    "rolling_6m_avg", "rolling_std", "is_december", "is_summer",
    "lag_1_spend",
]


def make_label_encoder() -> LabelEncoder:
    le = LabelEncoder()
    le.fit(CATEGORIES)
    return le


def build_training_matrix(monthly_grid: pd.DataFrame, train_months: list[str]) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """TRAIN-only fitting input for RF/Ridge: reuses
    pipeline.forecast.build_forecast_features (read-only) on exactly the
    fold's train_months subset -- no month outside train_months is ever
    passed in, so no future information can enter X_train/y_train."""
    train_subset = monthly_grid[monthly_grid["month"].isin(train_months)][["month", "category", "total_spend"]]
    X, y, le = build_forecast_features(train_subset, fit_le=True)
    return X, y, le


def compute_point_features(spend_history: np.ndarray) -> dict:
    """Given a chronological array of >=6 months of a single category's
    spend (real or, for the recursive strategy, partly predicted), returns
    the 4 history-derived features using the same windows pipeline/forecast.py
    uses at inference time (forecast.py:204-207, 462-465): last 3 months'
    mean/std and last 6 months' mean, last value as lag_1."""
    if len(spend_history) < 6:
        raise ValueError(f"compute_point_features needs >=6 months of history, got {len(spend_history)}")
    return {
        "rolling_3m_avg": float(np.mean(spend_history[-3:])),
        "rolling_6m_avg": float(np.mean(spend_history[-6:])),
        "rolling_std": float(np.std(spend_history[-3:], ddof=1)),
        "lag_1_spend": float(spend_history[-1]),
    }


def calendar_features(target_month: str) -> dict:
    ts = pd.Timestamp(target_month + "-01")
    return {
        "month_num": ts.month,
        "is_december": int(ts.month == 12),
        "is_summer": int(ts.month in (6, 7, 8)),
    }


def build_feature_row(spend_history: np.ndarray, target_month: str, category: str, le: LabelEncoder) -> pd.DataFrame:
    hist = compute_point_features(spend_history)
    cal = calendar_features(target_month)
    row = {
        "month_num": cal["month_num"],
        "category_encoded": le.transform([category])[0],
        "rolling_3m_avg": hist["rolling_3m_avg"],
        "rolling_6m_avg": hist["rolling_6m_avg"],
        "rolling_std": hist["rolling_std"],
        "is_december": cal["is_december"],
        "is_summer": cal["is_summer"],
        "lag_1_spend": hist["lag_1_spend"],
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def category_history(monthly_grid: pd.DataFrame, category: str, months: list[str]) -> np.ndarray:
    subset = monthly_grid[(monthly_grid["category"] == category) & (monthly_grid["month"].isin(months))]
    subset = subset.sort_values("month")
    return subset["total_spend"].values.astype(float)
