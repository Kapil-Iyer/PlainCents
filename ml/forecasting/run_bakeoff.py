"""
ML-B Part B: forecasting bake-off runner.

Orchestrates: build the zero-filled monthly (month x category) grid ->
build calendar-boundary expanding-window VALIDATION folds with the most
recent 3 calendar months reserved as the sealed FINAL TEST period -> for
every fold, fit RF/Ridge fresh on that fold's TRAIN-only months -> produce
Naive/Seasonal-Naive/RF-strategyA/RF-strategyB/Ridge-strategyA/Ridge-
strategyB predictions for every category at every available horizon ->
score against VALIDATION actuals only -> persist long-format predictions
and aggregated metrics.

FINAL TEST is never scored here: `temporal_eval.build_folds` structurally
excludes its months from every fold, and `reserved_final_period` in the
output records only which calendar months were reserved -- no value or
metric computed from them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import CATEGORIES
from ml.common.experiment_log import log_experiment
from ml.common.metrics import forecast_metric_bundle
from ml.forecasting.baselines import naive_predict, seasonal_naive_predict
from ml.forecasting.candidates import RandomForestCandidate, RidgeCandidate
from ml.forecasting.data_prep import DATASET_ID, EVIDENCE_TIER, build_monthly_grid
from ml.forecasting.features import build_training_matrix, category_history, make_label_encoder
from ml.forecasting.strategies import (
    STRATEGY_LAST_KNOWN_HISTORY,
    STRATEGY_RECURSIVE,
    predict_last_known_history,
    predict_recursive,
)
from ml.forecasting.temporal_eval import assert_no_reserved_month_used, build_folds

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = REPO_ROOT / "reports" / "ml" / "results" / "forecasting_predictions_long.csv"
METRICS_PATH = REPO_ROOT / "reports" / "ml" / "results" / "forecasting_metrics.json"

SEED = 42
MIN_TRAIN_MONTHS = 7
N_FINAL_RESERVED_MONTHS = 3


def run() -> dict:
    grid = build_monthly_grid()
    all_months = sorted(grid["month"].unique())
    folds, reserved_final_period = build_folds(
        all_months, min_train_months=MIN_TRAIN_MONTHS,
        n_final_reserved_months=N_FINAL_RESERVED_MONTHS,
    )
    assert_no_reserved_month_used(folds, reserved_final_period)

    rows = []

    for fold in folds:
        X_train, y_train, le_train = build_training_matrix(grid, fold.train_months)

        rf = RandomForestCandidate().fit(X_train, y_train)
        ridge = RidgeCandidate().fit(X_train, y_train)
        le = make_label_encoder()

        for category in CATEGORIES:
            spend_history = category_history(grid, category, fold.train_months)

            # --- Naive ---
            naive_pred = naive_predict(spend_history)
            for h, target_month in fold.target_months.items():
                actual = category_history(grid, category, [target_month])[0]
                rows.append(dict(origin_index=fold.origin_index, category=category, horizon=h,
                                  target_month=target_month, candidate="naive", strategy="n/a",
                                  actual=actual, predicted=naive_pred, eligible=True))

            # --- Seasonal Naive ---
            for h, target_month in fold.target_months.items():
                pred, eligible = seasonal_naive_predict(spend_history, h)
                actual = category_history(grid, category, [target_month])[0]
                rows.append(dict(origin_index=fold.origin_index, category=category, horizon=h,
                                  target_month=target_month, candidate="seasonal_naive", strategy="n/a",
                                  actual=actual, predicted=pred, eligible=eligible))

            # --- RF, both strategies ---
            rf_a = predict_last_known_history(rf, le, grid, fold, category)
            rf_b = predict_recursive(rf, le, grid, fold, category)
            for h, target_month in fold.target_months.items():
                actual = category_history(grid, category, [target_month])[0]
                rows.append(dict(origin_index=fold.origin_index, category=category, horizon=h,
                                  target_month=target_month, candidate="random_forest",
                                  strategy=STRATEGY_LAST_KNOWN_HISTORY,
                                  actual=actual, predicted=rf_a[h], eligible=True))
                rows.append(dict(origin_index=fold.origin_index, category=category, horizon=h,
                                  target_month=target_month, candidate="random_forest",
                                  strategy=STRATEGY_RECURSIVE,
                                  actual=actual, predicted=rf_b[h], eligible=True))

            # --- Ridge, both strategies ---
            ridge_a = predict_last_known_history(ridge, le, grid, fold, category)
            ridge_b = predict_recursive(ridge, le, grid, fold, category)
            for h, target_month in fold.target_months.items():
                actual = category_history(grid, category, [target_month])[0]
                rows.append(dict(origin_index=fold.origin_index, category=category, horizon=h,
                                  target_month=target_month, candidate="ridge",
                                  strategy=STRATEGY_LAST_KNOWN_HISTORY,
                                  actual=actual, predicted=ridge_a[h], eligible=True))
                rows.append(dict(origin_index=fold.origin_index, category=category, horizon=h,
                                  target_month=target_month, candidate="ridge",
                                  strategy=STRATEGY_RECURSIVE,
                                  actual=actual, predicted=ridge_b[h], eligible=True))

    long_df = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(RESULTS_PATH, index=False)

    metrics = summarize(long_df)
    metrics["dataset_id"] = DATASET_ID
    metrics["evidence_tier"] = EVIDENCE_TIER
    metrics["seed"] = SEED
    metrics["n_folds"] = len(folds)
    metrics["fold_origin_months"] = [fold.train_months[-1] for fold in folds]
    metrics["reserved_final_period"] = reserved_final_period
    metrics["final_test_sealed"] = True

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)

    for (candidate, strategy), _ in long_df.groupby(["candidate", "strategy"]):
        key = f"{candidate}__{strategy}"
        log_experiment(
            experiment_id=f"fcB_{key}",
            dataset_id=DATASET_ID,
            evidence_tier=EVIDENCE_TIER,
            seed=SEED,
            status="SUCCESS",
            model=candidate,
            forecasting_strategy=strategy,
            metrics=metrics["by_candidate_strategy"].get(key, {}).get("combined", {}),
            notes="Forecasting VALIDATION evaluation, calendar-boundary expanding window (ML Spec Section 12/13).",
        )

    return metrics


def summarize(long_df: pd.DataFrame) -> dict:
    result = {"by_candidate_strategy": {}}
    for (candidate, strategy), group in long_df.groupby(["candidate", "strategy"]):
        key = f"{candidate}__{strategy}"
        eligible_group = group[group["eligible"] & group["predicted"].notna()]
        entry = {"n_total_rows": int(len(group)), "n_eligible_rows": int(len(eligible_group))}

        entry["combined"] = forecast_metric_bundle(eligible_group["actual"].values, eligible_group["predicted"].values) if len(eligible_group) else None

        by_horizon = {}
        for h, hgroup in eligible_group.groupby("horizon"):
            by_horizon[str(h)] = forecast_metric_bundle(hgroup["actual"].values, hgroup["predicted"].values)
        entry["by_horizon"] = by_horizon

        by_category = {}
        for cat, cgroup in eligible_group.groupby("category"):
            by_category[cat] = forecast_metric_bundle(cgroup["actual"].values, cgroup["predicted"].values)
        entry["by_category"] = by_category

        result["by_candidate_strategy"][key] = entry

    return result


if __name__ == "__main__":
    metrics = run()
    print(f"Folds: {metrics['n_folds']}, reserved final period: {metrics['reserved_final_period']}")
    for key, entry in metrics["by_candidate_strategy"].items():
        c = entry["combined"]
        if c:
            print(f"{key}: combined WAPE={c['wape']:.4f} MAE={c['mae']:.2f} n={c['n']}")
        else:
            print(f"{key}: no eligible rows")
