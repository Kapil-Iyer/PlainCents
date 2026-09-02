"""
ML Spec Section 15: history-length sensitivity experiment. Truncates the
TRAIN history available at a FIXED set of test origins to 6/9/12/18 months
and re-evaluates WAPE/MAE, holding the target (test) months constant across
truncation lengths where possible -- exactly as Section 15 specifies.

This is an ANALYSIS only. It does not change config.py's frozen 12-unique-
month product eligibility rule (ML Spec Section 15's own explicit
constraint) -- nothing here is imported by pipeline/ or backend/.

SCOPE NOTE (honesty over completeness, per the Failure/Blocker Rule): with
only 24 months of synthetic history and 3 reserved for the sealed FINAL
TEST period, only origins with >=18 months of available prior history can
be tested at all four truncation lengths (6/9/12/18) on a like-for-like
basis. That leaves exactly 3 usable origins (dev-region indices 17, 18, 19)
-- a small sample, reported as such, not padded with fabricated data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import CATEGORIES
from ml.common.metrics import forecast_metric_bundle
from ml.forecasting.baselines import naive_predict
from ml.forecasting.candidates import RandomForestCandidate
from ml.forecasting.data_prep import build_monthly_grid
from ml.forecasting.features import build_training_matrix, category_history, make_label_encoder
from ml.forecasting.strategies import predict_last_known_history
from ml.forecasting.temporal_eval import Fold, build_folds

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = REPO_ROOT / "reports" / "ml" / "results" / "history_length_sensitivity.json"

TRUNCATION_LENGTHS = [6, 9, 12, 18]
MIN_COMMON_HISTORY = max(TRUNCATION_LENGTHS)  # 18: origins need >= this much available history


def run() -> dict:
    grid = build_monthly_grid()
    all_months = sorted(grid["month"].unique())
    folds, reserved_final_period = build_folds(all_months, min_train_months=7, n_final_reserved_months=3)

    usable_origins = [f for f in folds if len(f.train_months) >= MIN_COMMON_HISTORY]

    if not usable_origins:
        result = {
            "status": "UNAVAILABLE",
            "reason": f"No fold has >= {MIN_COMMON_HISTORY} months of available prior history "
                      f"after reserving the final {reserved_final_period['n_months']} months; "
                      f"history-length sensitivity cannot be evaluated at all four truncation "
                      f"lengths on a like-for-like origin set with only {len(all_months)} total months.",
            "truncation_lengths_attempted": TRUNCATION_LENGTHS,
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        return result

    le = make_label_encoder()
    by_truncation = {}

    for trunc_len in TRUNCATION_LENGTHS:
        rf_rows = []
        naive_rows = []
        for fold in usable_origins:
            truncated_train_months = fold.train_months[-trunc_len:]
            truncated_fold = Fold(origin_index=fold.origin_index, train_months=truncated_train_months,
                                   target_months=fold.target_months)
            X_train, y_train, _ = build_training_matrix(grid, truncated_train_months)
            rf = RandomForestCandidate().fit(X_train, y_train) if not X_train.empty else None

            for category in CATEGORIES:
                spend_history = category_history(grid, category, truncated_train_months)
                if len(spend_history) < 6:
                    continue
                naive_pred = naive_predict(spend_history)
                rf_preds = predict_last_known_history(rf, le, grid, truncated_fold, category) if rf is not None else None
                for h, target_month in truncated_fold.target_months.items():
                    actual = category_history(grid, category, [target_month])[0]
                    naive_rows.append((actual, naive_pred))
                    if rf_preds is not None:
                        rf_rows.append((actual, rf_preds[h]))

        rf_actuals, rf_preds_arr = zip(*rf_rows) if rf_rows else ([], [])
        naive_actuals, naive_preds_arr = zip(*naive_rows) if naive_rows else ([], [])

        by_truncation[str(trunc_len)] = {
            "n_origins_used": len(usable_origins),
            "n_predictions_naive": len(naive_rows),
            "n_predictions_rf": len(rf_rows),
            "random_forest": forecast_metric_bundle(np.array(rf_actuals), np.array(rf_preds_arr)) if rf_rows else None,
            "random_forest_note": None if rf_rows else (
                f"RF produced zero valid feature rows at {trunc_len} months of TRAIN history: "
                "pipeline.forecast.build_forecast_features requires 6 PRIOR months per category "
                "before the current one (a 6-row rolling-window warm-up), so an exactly-6-month "
                "TRAIN window yields no row with i>=6 in any category's own series. This is itself "
                "a Section 15 finding: RF's own feature engineering imposes a de facto >=7-month "
                "floor independent of the product's 12-month rule."
            ),
            "naive": forecast_metric_bundle(np.array(naive_actuals), np.array(naive_preds_arr)) if naive_rows else None,
        }

    result = {
        "status": "EVALUATED",
        "origins_used_dev_index": [f.origin_index for f in usable_origins],
        "origins_used_last_train_month": [f.train_months[-1] for f in usable_origins],
        "note": "Small sample (n_origins=%d) -- findings are indicative, not a robust statistical basis for changing the frozen 12-month product rule (ML Spec Section 15)." % len(usable_origins),
        "by_truncation_length": by_truncation,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
    return result


if __name__ == "__main__":
    result = run()
    if result["status"] == "UNAVAILABLE":
        print("UNAVAILABLE:", result["reason"])
    else:
        for trunc_len, entry in result["by_truncation_length"].items():
            rf = entry["random_forest"]
            nv = entry["naive"]
            rf_str = f"RF WAPE={rf['wape']:.4f} MAE={rf['mae']:.2f}" if rf else f"RF n/a ({entry['random_forest_note']})"
            nv_str = f"Naive WAPE={nv['wape']:.4f} MAE={nv['mae']:.2f}" if nv else "Naive n/a"
            print(f"history={trunc_len}mo: {rf_str} | {nv_str}")
