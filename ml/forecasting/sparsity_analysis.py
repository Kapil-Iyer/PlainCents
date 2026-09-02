"""
ML Spec Section 16: per-category sparsity/zero-spend analysis. Evaluates the
FOUR candidate eligibility rules Section 16 lists against every category's
actual sparsity profile in the (zero-filled) monthly grid, WITHOUT adopting
any one rule as final -- Section 16 explicitly reserves that decision for a
future, evidence-informed step, not this document. Also reports how the
already-computed forecasting metrics (run_bakeoff's by_category breakdown)
behave differently for dense vs. sparse vs. always-zero categories, since
that behavioral difference is the actual point of the analysis, not the
sparsity statistics alone.
"""
from __future__ import annotations

import json
from pathlib import Path

from ml.forecasting.data_prep import build_monthly_grid
from ml.forecasting.temporal_eval import build_folds

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = REPO_ROOT / "reports" / "ml" / "results" / "sparsity_analysis.json"
FORECAST_METRICS_PATH = REPO_ROOT / "reports" / "ml" / "results" / "forecasting_metrics.json"

# Section 16's four candidate rules, applied with illustrative-but-disclosed
# thresholds (none are frozen by the ML Spec; chosen for interpretability,
# not tuned to favor any category or candidate).
MIN_TOTAL_MONTHS_THRESHOLD = 12          # rule 1: "minimum total historical months"
MIN_NONZERO_MONTHS_THRESHOLD = 6         # rule 2: "minimum non-zero-spend months"
MIN_RECENT_NONZERO_WINDOW = 6            # rule 3: "at least one non-zero month within the last N"
# rule 4 ("survives build_forecast_features's dropna") is structural, not threshold-based.


def _apply_candidate_rules(total_months: int, nonzero_months: int, last_n_nonzero_count: int, survives_dropna: bool) -> dict:
    return {
        "rule_1_min_total_months": {
            "threshold": MIN_TOTAL_MONTHS_THRESHOLD,
            "actual_total_months": total_months,
            "passes": total_months >= MIN_TOTAL_MONTHS_THRESHOLD,
        },
        "rule_2_min_nonzero_months": {
            "threshold": MIN_NONZERO_MONTHS_THRESHOLD,
            "actual_nonzero_months": nonzero_months,
            "passes": nonzero_months >= MIN_NONZERO_MONTHS_THRESHOLD,
        },
        "rule_3_min_recent_nonzero": {
            "window": MIN_RECENT_NONZERO_WINDOW,
            "actual_recent_nonzero_count": last_n_nonzero_count,
            "passes": last_n_nonzero_count >= 1,
        },
        "rule_4_survives_dropna_floor": {
            "passes": survives_dropna,
        },
    }


def run() -> dict:
    grid = build_monthly_grid()
    all_months = sorted(grid["month"].unique())
    folds, reserved = build_folds(all_months, min_train_months=7, n_final_reserved_months=3)
    dev_months = [m for m in all_months if m not in set(reserved["months"])]

    per_category = {}
    for category, cat_grid in grid[grid["month"].isin(dev_months)].groupby("category"):
        cat_grid = cat_grid.sort_values("month")
        total_months = len(cat_grid)
        nonzero_months = int((cat_grid["total_spend"] > 0).sum())
        last_n = cat_grid.tail(MIN_RECENT_NONZERO_WINDOW)
        last_n_nonzero_count = int((last_n["total_spend"] > 0).sum())
        survives_dropna = total_months >= 7  # matches build_forecast_features's 6-prior-month floor

        rules = _apply_candidate_rules(total_months, nonzero_months, last_n_nonzero_count, survives_dropna)

        if nonzero_months == 0:
            bucket = "always_zero"
        elif nonzero_months < total_months:
            bucket = "intermittent"
        else:
            bucket = "dense"

        per_category[category] = {
            "total_months_in_dev_region": total_months,
            "nonzero_months": nonzero_months,
            "pct_zero": round(1 - nonzero_months / total_months, 4) if total_months else None,
            "sparsity_bucket": bucket,
            "candidate_rule_outcomes": rules,
        }

    # Link to already-computed forecasting metrics (by_category), if present.
    metric_linkage = {}
    if FORECAST_METRICS_PATH.exists():
        with open(FORECAST_METRICS_PATH) as f:
            fmetrics = json.load(f)
        for key, entry in fmetrics.get("by_candidate_strategy", {}).items():
            by_cat = entry.get("by_category", {})
            for category, bundle in by_cat.items():
                metric_linkage.setdefault(category, {})[key] = {
                    "wape": bundle["wape"], "mae": bundle["mae"],
                    "mape_all": bundle["mape_all"], "n_near_zero_actual": bundle["n_near_zero_actual"],
                    "n": bundle["n"],
                }

    result = {
        "per_category_sparsity": per_category,
        "metric_behavior_by_category": metric_linkage,
        "interpretation_note": (
            "WAPE is undefined (NaN, denominator=0) for a category with zero actual spend across "
            "the entire evaluated window ('Other', per this dataset's K-Means-derived labels never "
            "assigning that category -- see reports/ml). MAE remains defined and near-trivially small "
            "for an all-zero category, since any candidate predicting near-zero achieves a small "
            "absolute error almost by construction -- this is a metric-sensitivity finding Section 16 "
            "specifically asks to be surfaced, not treated as evidence of genuine forecasting skill."
        ),
        "rule_adoption_status": "NONE of the four candidate rules is adopted as final -- ML Spec Section 16 explicitly reserves that decision; this analysis only reports how each rule would classify each category.",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
    return result


if __name__ == "__main__":
    result = run()
    for category, info in result["per_category_sparsity"].items():
        print(f"{category}: bucket={info['sparsity_bucket']} nonzero={info['nonzero_months']}/{info['total_months_in_dev_region']}")
