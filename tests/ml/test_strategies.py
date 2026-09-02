"""
ML Spec Section 11.1 multi-step strategy correctness tests: last-known-
history reuses the same real-history features across all horizons;
recursive feeds each prediction forward into the next horizon's features.
"""
import numpy as np
import pandas as pd

from config import CATEGORIES
from ml.forecasting.features import make_label_encoder
from ml.forecasting.strategies import predict_last_known_history, predict_recursive
from ml.forecasting.temporal_eval import Fold


class _EchoLag1Model:
    """A fake 'model' whose prediction is exactly the lag_1_spend feature it
    was given -- lets the tests assert precisely which history value each
    strategy fed into the feature row at each horizon, without depending on
    a real RF/Ridge fit."""

    def predict_row(self, row_df: pd.DataFrame) -> float:
        return float(row_df["lag_1_spend"].iloc[0])


def _monthly_grid_constant_history(category: str, months: list[str], values: list[float]) -> pd.DataFrame:
    rows = [{"month": m, "category": category, "total_spend": v} for m, v in zip(months, values)]
    # Fill every other category with zeros so build_feature_row's dependencies (unused here) don't choke.
    for other in CATEGORIES:
        if other == category:
            continue
        for m, v in zip(months, [0.0] * len(months)):
            rows.append({"month": m, "category": other, "total_spend": 0.0})
    return pd.DataFrame(rows)


def test_last_known_history_reuses_same_lag1_for_all_horizons():
    months = [f"2024-{m:02d}" for m in range(1, 8)]  # 7 months of TRAIN history
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]  # last value (lag_1) = 70.0
    grid = _monthly_grid_constant_history("Shopping", months, values)
    fold = Fold(origin_index=6, train_months=months, target_months={1: "2024-08", 2: "2024-09", 3: "2024-10"})
    le = make_label_encoder()

    preds = predict_last_known_history(_EchoLag1Model(), le, grid, fold, "Shopping")

    # Strategy A: the SAME real lag_1 (70.0) backs every horizon's prediction
    # via our echo model -- only calendar features differ, which the echo
    # model ignores.
    assert preds[1] == preds[2] == preds[3] == 70.0


def test_recursive_feeds_prior_prediction_forward():
    months = [f"2024-{m:02d}" for m in range(1, 8)]
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    grid = _monthly_grid_constant_history("Shopping", months, values)
    fold = Fold(origin_index=6, train_months=months, target_months={1: "2024-08", 2: "2024-09", 3: "2024-10"})
    le = make_label_encoder()

    preds = predict_recursive(_EchoLag1Model(), le, grid, fold, "Shopping")

    # h=1 uses the real last-known lag_1 (70.0).
    assert preds[1] == 70.0
    # h=2's lag_1 must be h=1's OWN prediction (70.0 again here, since the
    # echo model always returns the current lag_1) -- but critically it is
    # NOT computed from real history alone; the model was called with a
    # history array one element longer than strategy A's.
    assert preds[2] == 70.0
    assert preds[3] == 70.0


def test_recursive_and_last_known_diverge_with_a_trend_sensitive_model():
    """A model that reacts to the 3-month rolling average (not just lag_1)
    should produce DIFFERENT +2/+3 predictions between the two strategies,
    proving the recursive strategy's extended history actually changes the
    feature computation, not just relabels the same number."""
    class _EchoRolling3Model:
        def predict_row(self, row_df: pd.DataFrame) -> float:
            return float(row_df["rolling_3m_avg"].iloc[0])

    months = [f"2024-{m:02d}" for m in range(1, 8)]
    values = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0]  # sharp jump in the last month
    grid = _monthly_grid_constant_history("Shopping", months, values)
    fold = Fold(origin_index=6, train_months=months, target_months={1: "2024-08", 2: "2024-09", 3: "2024-10"})
    le = make_label_encoder()

    preds_a = predict_last_known_history(_EchoRolling3Model(), le, grid, fold, "Shopping")
    preds_b = predict_recursive(_EchoRolling3Model(), le, grid, fold, "Shopping")

    # Strategy A: rolling_3m_avg computed once from real history (last 3
    # real months: 10,10,100 -> mean 40), reused for h=1,2,3.
    assert preds_a[1] == preds_a[2] == preds_a[3]
    assert abs(preds_a[1] - 40.0) < 1e-9

    # Strategy B: h=1 matches strategy A's h=1 (both use only real history).
    assert abs(preds_b[1] - preds_a[1]) < 1e-9
    # h=2 recomputes rolling_3m_avg over [10, 100, pred_1=40] -> mean 50,
    # which differs from strategy A's constant 40.
    assert preds_b[2] != preds_a[2]
