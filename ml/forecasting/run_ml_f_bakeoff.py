"""
ML-F: forecast re-evaluation (brief §18-22).

Reuses the EXACT ML-C harness (ml/forecasting/data_prep.py's synthetic
24-month monthly grid, ml/forecasting/temporal_eval.py's calendar-boundary
expanding-window folds with the same 3-month reserved FINAL period) rather
than rebuilding forecasting evaluation from scratch -- the ML-F-A audit and
this phase's brief are both explicit that this dataset "remains synthetic
unless repository evidence shows otherwise," and no new evidence changes
that here. This script only ADDS two small, interpretable baselines (rolling
mean, EWMA) to the existing candidate set and reruns the pre-registered
winner-selection question, once, on the same evidence base.

Candidates compared (brief §19): Naive (current production), Seasonal Naive,
3-month and 6-month rolling mean, EWMA (alpha in {0.3, 0.5, 0.7}), Ridge
(both strategies, continuity), Random Forest (both strategies, continuity).
No LSTM/transformer/Prophet/external API -- explicitly ruled out by the brief
and not justified by any finding here or in the ML-F-A audit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import CATEGORIES
from ml.common.metrics import forecast_metric_bundle
from ml.forecasting.baselines import ewma_predict, naive_predict, rolling_mean_predict, seasonal_naive_predict
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
RESULTS_PATH = REPO_ROOT / "reports" / "ml" / "results" / "ml_f_forecasting_predictions_long.csv"
METRICS_PATH = REPO_ROOT / "reports" / "ml" / "results" / "ml_f_forecasting_metrics.json"
HISTORY_LENGTH_PATH = REPO_ROOT / "reports" / "ml" / "results" / "ml_f_history_length_sensitivity.json"

SEED = 42
MIN_TRAIN_MONTHS = 7
N_FINAL_RESERVED_MONTHS = 3
EWMA_ALPHAS = [0.3, 0.5, 0.7]
TRUNCATION_LENGTHS = [6, 9, 12, 18]


def run_validation_bakeoff() -> dict:
    grid = build_monthly_grid()
    all_months = sorted(grid["month"].unique())
    folds, reserved_final_period = build_folds(
        all_months, min_train_months=MIN_TRAIN_MONTHS, n_final_reserved_months=N_FINAL_RESERVED_MONTHS,
    )
    assert_no_reserved_month_used(folds, reserved_final_period)

    rows = []
    for fold in folds:
        X_train, y_train, _ = build_training_matrix(grid, fold.train_months)
        rf = RandomForestCandidate().fit(X_train, y_train)
        ridge = RidgeCandidate().fit(X_train, y_train)
        le = make_label_encoder()

        for category in CATEGORIES:
            spend_history = category_history(grid, category, fold.train_months)

            def _add(candidate, strategy, pred_by_horizon, eligible_by_horizon=None):
                # eligible_by_horizon: None -> every horizon eligible (the
                # common case); otherwise a {horizon: bool} dict, needed for
                # seasonal_naive, whose eligibility genuinely varies PER
                # HORIZON within the same fold (>=13-month floor measured
                # from each individual target month) -- collapsing that into
                # one shared flag across all three horizons would silently
                # discard/keep rows incorrectly.
                for h, target_month in fold.target_months.items():
                    actual = category_history(grid, category, [target_month])[0]
                    pred = pred_by_horizon[h] if isinstance(pred_by_horizon, dict) else pred_by_horizon
                    eligible = True if eligible_by_horizon is None else eligible_by_horizon[h]
                    rows.append(dict(origin_index=fold.origin_index, category=category, horizon=h,
                                      target_month=target_month, candidate=candidate, strategy=strategy,
                                      actual=actual, predicted=pred, eligible=eligible))

            _add("naive", "n/a", naive_predict(spend_history))
            _add("rolling_mean_3", "n/a", rolling_mean_predict(spend_history, 3))
            _add("rolling_mean_6", "n/a", rolling_mean_predict(spend_history, 6))
            for alpha in EWMA_ALPHAS:
                _add(f"ewma_{alpha}", "n/a", ewma_predict(spend_history, alpha))

            seasonal_by_h, seasonal_eligible_by_h = {}, {}
            for h in fold.target_months:
                pred, eligible = seasonal_naive_predict(spend_history, h)
                seasonal_by_h[h] = pred
                seasonal_eligible_by_h[h] = eligible
            _add("seasonal_naive", "n/a", seasonal_by_h, eligible_by_horizon=seasonal_eligible_by_h)

            _add("random_forest", STRATEGY_LAST_KNOWN_HISTORY, predict_last_known_history(rf, le, grid, fold, category))
            _add("random_forest", STRATEGY_RECURSIVE, predict_recursive(rf, le, grid, fold, category))
            _add("ridge", STRATEGY_LAST_KNOWN_HISTORY, predict_last_known_history(ridge, le, grid, fold, category))
            _add("ridge", STRATEGY_RECURSIVE, predict_recursive(ridge, le, grid, fold, category))

    long_df = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(RESULTS_PATH, index=False)

    metrics = _summarize(long_df)
    metrics.update({
        "dataset_id": DATASET_ID, "evidence_tier": EVIDENCE_TIER, "seed": SEED,
        "n_folds": len(folds), "reserved_final_period": reserved_final_period, "final_test_sealed": True,
    })
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)
    return metrics


def _summarize(long_df: pd.DataFrame) -> dict:
    result = {"by_candidate_strategy": {}}
    for (candidate, strategy), group in long_df.groupby(["candidate", "strategy"]):
        key = f"{candidate}__{strategy}"
        eligible_group = group[group["eligible"] & group["predicted"].notna()]
        entry = {"n_total_rows": int(len(group)), "n_eligible_rows": int(len(eligible_group))}
        entry["combined"] = forecast_metric_bundle(eligible_group["actual"].values, eligible_group["predicted"].values) if len(eligible_group) else None
        entry["by_horizon"] = {
            str(h): forecast_metric_bundle(hg["actual"].values, hg["predicted"].values)
            for h, hg in eligible_group.groupby("horizon")
        }
        result["by_candidate_strategy"][key] = entry
    return result


def select_winner(metrics: dict) -> dict:
    """Pre-registered winner rule (ML-F brief §22), applied BEFORE the sealed
    FINAL_TEST is touched. Lowest pooled VALIDATION WAPE, subject to: stable
    across history lengths (checked separately by run_history_length_
    sensitivity), no materially larger history requirement than the product
    gate, no significant runtime complexity, and a MEANINGFUL improvement
    (not noise) -- tie-break toward simplicity in the stated order."""
    WAPE_MEANINGFUL_DELTA = 0.01  # ML-F brief §22.4: "meaningful rather than trivial noise"
    simplicity_order = [
        "naive__n/a", "rolling_mean_3__n/a", "rolling_mean_6__n/a",
        "ewma_0.3__n/a", "ewma_0.5__n/a", "ewma_0.7__n/a",
        "ridge__last_known_history", "ridge__recursive",
        "random_forest__last_known_history", "random_forest__recursive",
    ]
    by_cs = metrics["by_candidate_strategy"]
    ranked = sorted(
        ((k, v["combined"]["wape"]) for k, v in by_cs.items() if v["combined"] is not None),
        key=lambda kv: kv[1],
    )
    best_key, best_wape = ranked[0]
    naive_wape = by_cs["naive__n/a"]["combined"]["wape"]

    # Only override Naive (current production, simplest) if something beats
    # it by a MEANINGFUL margin -- otherwise keep Naive per the tie-break order.
    if naive_wape - best_wape < WAPE_MEANINGFUL_DELTA:
        winner_key = "naive__n/a"
    else:
        tied = [k for k, w in ranked if w - best_wape <= WAPE_MEANINGFUL_DELTA]
        winner_key = next((k for k in simplicity_order if k in tied), best_key)

    return {
        "ranked_by_validation_pooled_wape": ranked,
        "winner": winner_key,
        "winner_pooled_wape": dict(ranked)[winner_key],
        "naive_pooled_wape": naive_wape,
        "reasoning": (
            f"Lowest pooled VALIDATION WAPE, tie-broken (within {WAPE_MEANINGFUL_DELTA}) toward "
            "simplicity in the order Naive > rolling mean/EWMA > Ridge > RF (ML-F brief Section 22). "
            "A challenger must beat Naive by more than this margin to be selected -- otherwise Naive "
            "(the simplest, current production model) is kept."
        ),
    }


def run_history_length_sensitivity() -> dict:
    """Extends ml/forecasting/history_sensitivity.py's truncation experiment
    (already-run evidence: RF/Ridge have a de facto >=7-month floor from
    their own rolling-window feature engineering, independent of the
    product's history gate) to also cover rolling mean and EWMA -- both of
    which, like Naive, have NO minimum-history floor beyond one observed
    month, and are checked here to confirm their pooled WAPE is likewise
    unaffected by how much history is truncated away."""
    from ml.forecasting.candidates import RandomForestCandidate as RFC
    from ml.forecasting.strategies import predict_last_known_history as _plkh
    from ml.forecasting.temporal_eval import Fold

    grid = build_monthly_grid()
    all_months = sorted(grid["month"].unique())
    folds, reserved_final_period = build_folds(all_months, min_train_months=7, n_final_reserved_months=3)
    min_common_history = max(TRUNCATION_LENGTHS)
    usable_origins = [f for f in folds if len(f.train_months) >= min_common_history]

    if not usable_origins:
        result = {"status": "UNAVAILABLE", "reason": "No fold has enough history for a like-for-like comparison."}
        HISTORY_LENGTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_LENGTH_PATH, "w") as f:
            json.dump(result, f, indent=2)
        return result

    le = make_label_encoder()
    by_truncation = {}
    for trunc_len in TRUNCATION_LENGTHS:
        rows_by_candidate: dict[str, list] = {
            "naive": [], "rolling_mean_3": [], "rolling_mean_6": [],
            "ewma_0.3": [], "ewma_0.5": [], "ewma_0.7": [], "random_forest": [],
        }
        for fold in usable_origins:
            truncated_train_months = fold.train_months[-trunc_len:]
            truncated_fold = Fold(origin_index=fold.origin_index, train_months=truncated_train_months,
                                   target_months=fold.target_months)
            X_train, y_train, _ = build_training_matrix(grid, truncated_train_months)
            rf = RFC().fit(X_train, y_train) if not X_train.empty else None

            for category in CATEGORIES:
                spend_history = category_history(grid, category, truncated_train_months)
                if len(spend_history) < 1:
                    continue
                preds = {
                    "naive": naive_predict(spend_history),
                    "rolling_mean_3": rolling_mean_predict(spend_history, 3),
                    "rolling_mean_6": rolling_mean_predict(spend_history, 6),
                    "ewma_0.3": ewma_predict(spend_history, 0.3),
                    "ewma_0.5": ewma_predict(spend_history, 0.5),
                    "ewma_0.7": ewma_predict(spend_history, 0.7),
                }
                rf_preds = _plkh(rf, le, grid, truncated_fold, category) if rf is not None else None
                for h, target_month in truncated_fold.target_months.items():
                    actual = category_history(grid, category, [target_month])[0]
                    for name, pred in preds.items():
                        rows_by_candidate[name].append((actual, pred))
                    if rf_preds is not None:
                        rows_by_candidate["random_forest"].append((actual, rf_preds[h]))

        entry = {}
        for name, pairs in rows_by_candidate.items():
            if not pairs:
                entry[name] = None
                continue
            actuals, preds_arr = zip(*pairs)
            entry[name] = {**forecast_metric_bundle(np.array(actuals), np.array(preds_arr)), "n_predictions": len(pairs)}
        by_truncation[str(trunc_len)] = entry

    result = {
        "status": "EVALUATED",
        "origins_used_dev_index": [f.origin_index for f in usable_origins],
        "note": f"Small sample (n_origins={len(usable_origins)}) -- indicative, per ML Spec Section 15's own caveat, carried forward for ML-F.",
        "by_truncation_length": by_truncation,
    }
    HISTORY_LENGTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_LENGTH_PATH, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
    return result


if __name__ == "__main__":
    metrics = run_validation_bakeoff()
    print(f"Folds: {metrics['n_folds']}, reserved final period: {metrics['reserved_final_period']}")
    for key, entry in sorted(metrics["by_candidate_strategy"].items()):
        c = entry["combined"]
        print(f"{key}: pooled WAPE={c['wape']:.4f} n={c['n']}" if c else f"{key}: no eligible rows")

    winner_info = select_winner(metrics)
    print(f"\nRanked: {winner_info['ranked_by_validation_pooled_wape']}")
    print(f"Winner: {winner_info['winner']} (WAPE={winner_info['winner_pooled_wape']:.4f} vs Naive {winner_info['naive_pooled_wape']:.4f})")
    print(f"Reasoning: {winner_info['reasoning']}")

    with open(REPO_ROOT / "reports" / "ml" / "results" / "ml_f_forecast_winner_selection.json", "w") as f:
        json.dump(winner_info, f, indent=2, default=str)

    hist = run_history_length_sensitivity()
    print(f"\nHistory-length sensitivity: {hist['status']}")
    if hist["status"] == "EVALUATED":
        for trunc_len, entry in hist["by_truncation_length"].items():
            summary = ", ".join(f"{name}={m['wape']:.4f}" for name, m in entry.items() if m)
            print(f"  history={trunc_len}mo: {summary}")
