"""
ML-F Part: forecasting FINAL evaluation (brief §23).

Runs the single permitted FINAL pass for the ML-F-selected forecast recipe
against the reserved temporal period 2024-10/2024-11/2024-12 (same reserved
period ML-C used -- ml/forecasting/temporal_eval.build_folds's own
reservation logic, not re-derived). Mirrors ml/forecasting/run_final.py's
discipline exactly (refuse unless frozen, chronology preserved, no
real-world accuracy claim), generalized to whichever candidate
reports/ml/ML_F_SELECTION_RECORD.json's forecasting_selection.winner names,
rather than hardcoding "naive".
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import CATEGORIES
from ml.common.metrics import forecast_metric_bundle
from ml.forecasting.baselines import ewma_predict, naive_predict, rolling_mean_predict
from ml.forecasting.data_prep import DATASET_ID, EVIDENCE_TIER, build_monthly_grid
from ml.forecasting.temporal_eval import build_folds

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SELECTION_RECORD_PATH = REPO_ROOT / "reports" / "ml" / "ML_F_SELECTION_RECORD.json"
OUT_PATH = REPO_ROOT / "reports" / "ml" / "results" / "ml_f_final_forecasting.json"

_SIMPLE_PREDICTORS = {
    "naive": lambda h: naive_predict(h),
    "rolling_mean_3": lambda h: rolling_mean_predict(h, 3),
    "rolling_mean_6": lambda h: rolling_mean_predict(h, 6),
    "ewma_0.3": lambda h: ewma_predict(h, 0.3),
    "ewma_0.5": lambda h: ewma_predict(h, 0.5),
    "ewma_0.7": lambda h: ewma_predict(h, 0.7),
}


class SelectionNotFrozenError(RuntimeError):
    pass


def load_and_verify_selection(selection_record_path: Path = SELECTION_RECORD_PATH) -> dict:
    if not selection_record_path.exists():
        raise SelectionNotFrozenError(f"{selection_record_path} does not exist -- ML-F forecasting selection must be frozen first.")
    with open(selection_record_path, encoding="utf-8") as f:
        selection = json.load(f)
    winner_key = selection.get("forecasting_selection", {}).get("winner")
    if not winner_key:
        raise SelectionNotFrozenError(f"{selection_record_path} does not name a forecasting winner -- refusing to run FINAL.")
    candidate_name = winner_key.split("__")[0]
    if candidate_name not in _SIMPLE_PREDICTORS:
        raise SelectionNotFrozenError(
            f"Winner {winner_key!r} is not one of the simple (no-fit, current-month-only) predictors this "
            "module supports -- a learned-model (Ridge/RF) winner would need its own per-fold refit path, "
            "not this simple lookup evaluator."
        )
    return selection, candidate_name


def run(selection_record_path: Path = SELECTION_RECORD_PATH, out_path: Path = OUT_PATH) -> dict:
    selection, candidate_name = load_and_verify_selection(selection_record_path)
    predictor = _SIMPLE_PREDICTORS[candidate_name]

    grid = build_monthly_grid()
    all_months = sorted(grid["month"].unique())
    _, reserved_final_period = build_folds(all_months, min_train_months=7, n_final_reserved_months=3)
    reserved_months = reserved_final_period["months"]
    if reserved_months != ["2024-10", "2024-11", "2024-12"]:
        raise RuntimeError(f"Unexpected reserved period {reserved_months} -- refusing to run FINAL.")

    train_months = [m for m in all_months if m not in reserved_months]

    rows = []
    for category in CATEGORIES:
        cat_grid = grid[grid["category"] == category].sort_values("month")
        history = cat_grid[cat_grid["month"].isin(train_months)]["total_spend"].to_numpy()
        assert len(history) == len(train_months)
        pred = predictor(history)  # same value reused at +1/+2/+3, like naive_v1 today

        for h, target_month in zip([1, 2, 3], reserved_months):
            actual_row = cat_grid[cat_grid["month"] == target_month]["total_spend"]
            assert len(actual_row) == 1
            actual = float(actual_row.iloc[0])
            rows.append({"category": category, "horizon": h, "target_month": target_month,
                         "actual": actual, "predicted": pred})

    long_df = pd.DataFrame(rows)
    combined = forecast_metric_bundle(long_df["actual"].to_numpy(), long_df["predicted"].to_numpy())
    by_horizon = {str(h): forecast_metric_bundle(hg["actual"].to_numpy(), hg["predicted"].to_numpy())
                  for h, hg in long_df.groupby("horizon")}
    by_category = {cat: forecast_metric_bundle(cg["actual"].to_numpy(), cg["predicted"].to_numpy())
                   for cat, cg in long_df.groupby("category")}

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        git_commit = None

    result = {
        "result_label": "Untouched temporal-test performance on reserved synthetic months",
        "not_to_be_described_as": ["Tier B", "real-world", "temporal validation"],
        "selected_candidate": candidate_name,
        "selection_record_ref": "reports/ml/ML_F_SELECTION_RECORD.json",
        "dataset_id": DATASET_ID,
        "evidence_tier": EVIDENCE_TIER,
        "reserved_period": {"months": reserved_months, "n_months": 3},
        "train_months_used": {"start": train_months[0], "end": train_months[-1], "n_months": len(train_months)},
        "chronology_preserved": True,
        "n_predictions": int(len(long_df)),
        "n_categories": len(CATEGORIES),
        "model_impl_version": "rolling_mean_3_v1" if candidate_name == "rolling_mean_3" else f"{candidate_name}_v1",
        "git_commit": git_commit,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "final_metrics": {"combined": combined, "by_horizon": by_horizon, "by_category": by_category},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=False, default=str)
    return result


if __name__ == "__main__":
    result = run()
    c = result["final_metrics"]["combined"]
    print(f"Selected candidate: {result['selected_candidate']}")
    print(f"FINAL reserved period {result['reserved_period']['months']}: combined WAPE={c['wape']:.4f} MAE={c['mae']:.2f} n={c['n']}")
    for h, m in result["final_metrics"]["by_horizon"].items():
        print(f"  +{h}: WAPE={m['wape']:.4f} MAE={m['mae']:.2f}")
