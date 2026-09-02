"""
ML Spec Section 11.1: multi-step (+1/+2/+3) forecasting strategies, applied
to the RF and Ridge candidates (Naive/Seasonal Naive have no strategy
distinction -- see baselines.py's module docstring).

Strategy A -- "last-known-history" (V1's current approach): the SAME
rolling/lag feature values (computed once, from real TRAIN history) are
reused for all three horizons' feature rows; only calendar-derived features
(month_num/is_december/is_summer) vary by horizon.

Strategy B -- "recursive": the +1 prediction is appended to the spend
history as if it were a real observation before building the +2 feature
row, and similarly the +2 prediction feeds +3 -- propagating the model's
own predictions forward.

Strategy C (direct horizon-specific models) is NOT implemented in this
ML-B pass: Section 11.1 states it is "only pursued if evidence from A/B
shows a genuine horizon-specific pattern... not adopted by default." The
A vs. B comparison in reports/ml determines whether that evidence exists;
this is a scope decision made AFTER seeing A/B results, not a shortcut
taken in advance.
"""
from __future__ import annotations

import numpy as np

from ml.forecasting.features import build_feature_row, category_history

STRATEGY_LAST_KNOWN_HISTORY = "last_known_history"
STRATEGY_RECURSIVE = "recursive"


def predict_last_known_history(model, le, monthly_grid, fold, category: str) -> dict[int, float]:
    """Strategy A. `model` must expose .predict_row(row_df) -> float."""
    spend_history = category_history(monthly_grid, category, fold.train_months)
    preds = {}
    for h, target_month in fold.target_months.items():
        row = build_feature_row(spend_history, target_month, category, le)
        preds[h] = model.predict_row(row)
    return preds


def predict_recursive(model, le, monthly_grid, fold, category: str) -> dict[int, float]:
    """Strategy B. Horizons are processed in increasing order regardless of
    dict insertion order, since h=2 depends on h=1's prediction having
    already been appended to the working history."""
    spend_history = list(category_history(monthly_grid, category, fold.train_months))
    preds = {}
    for h in sorted(fold.target_months.keys()):
        target_month = fold.target_months[h]
        row = build_feature_row(np.array(spend_history, dtype=float), target_month, category, le)
        pred = model.predict_row(row)
        preds[h] = pred
        spend_history.append(pred)
    return preds


STRATEGY_FUNCS = {
    STRATEGY_LAST_KNOWN_HISTORY: predict_last_known_history,
    STRATEGY_RECURSIVE: predict_recursive,
}
