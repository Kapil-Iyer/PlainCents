"""
ML-C Part F2: Forecasting FINAL evaluation.

Runs the single permitted FINAL pass for forecasting (ML Spec Section 20/21):
evaluates ONLY the candidate/configuration frozen in
reports/ml/ML_C_SELECTION_RECORD.json against the reserved temporal period
2024-10 / 2024-11 / 2024-12, training strictly on history before that period
(chronology preserved -- reuses ml.forecasting.temporal_eval.build_folds()'s
own reservation logic rather than re-deriving the boundary independently).

Refuses to run if:
  - the selection record does not exist yet, or
  - it does not name "naive" / strategy "N/A" as selected.

No rejected candidate (seasonal_naive, random_forest, ridge, either
strategy) is evaluated by this module -- there is no code path here that
could run them against the reserved period.

Result is labeled precisely per ML Spec Section 21: "Untouched
temporal-test performance on reserved synthetic months" -- never "Tier B",
"real-world", or "temporal validation" (that label is reserved for a
result where no untouched final period was feasible; one was feasible and
used here).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import CATEGORIES
from ml.common.metrics import forecast_metric_bundle
from ml.forecasting.baselines import naive_predict
from ml.forecasting.data_prep import DATASET_ID, EVIDENCE_TIER, build_monthly_grid
from ml.forecasting.temporal_eval import build_folds

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SELECTION_RECORD_PATH = REPO_ROOT / "reports" / "ml" / "ML_C_SELECTION_RECORD.json"
OUT_PATH = REPO_ROOT / "reports" / "ml" / "results" / "final_forecasting.json"

SELECTED_CANDIDATE_NAME = "naive"
SELECTED_STRATEGY = "N/A"

REJECTED_CANDIDATES = [
    "seasonal_naive", "random_forest__last_known_history", "random_forest__recursive",
    "ridge__last_known_history", "ridge__recursive",
]


class SelectionNotFrozenError(RuntimeError):
    """Raised when FINAL is attempted before the ML-C selection record exists
    and names this module's candidate/strategy as selected."""


def load_and_verify_selection(selection_record_path: Path = SELECTION_RECORD_PATH) -> dict:
    if not selection_record_path.exists():
        raise SelectionNotFrozenError(
            f"{selection_record_path} does not exist -- the ML-C selection must be frozen "
            "before any FINAL evaluation may run."
        )
    with open(selection_record_path, encoding="utf-8") as f:
        selection = json.load(f)
    selected = selection.get("forecasting_selection", {}).get("selected_candidate")
    strategy = selection.get("multi_step_strategy_selection", {}).get("selected_strategy")
    if selected != SELECTED_CANDIDATE_NAME or strategy != SELECTED_STRATEGY:
        raise SelectionNotFrozenError(
            f"Selection record names candidate={selected!r} strategy={strategy!r} as selected, "
            f"not {SELECTED_CANDIDATE_NAME!r}/{SELECTED_STRATEGY!r} -- refusing to run FINAL for "
            "a candidate/strategy that was not frozen as the selection."
        )
    return selection


def run(selection_record_path: Path = SELECTION_RECORD_PATH, out_path: Path = OUT_PATH) -> dict:
    load_and_verify_selection(selection_record_path)

    grid = build_monthly_grid()
    all_months = sorted(grid["month"].unique())

    # Reuse ML-B's own reservation logic rather than re-deriving the boundary.
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
        # Naive has no per-horizon/multi-step variant (selected strategy N/A):
        # the same last-observed value is used at +1/+2/+3.
        pred = naive_predict(history)

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
        "selected_candidate": SELECTED_CANDIDATE_NAME,
        "selected_strategy": SELECTED_STRATEGY,
        "selection_record_ref": "reports/ml/ML_C_SELECTION_RECORD.json",
        "dataset_id": DATASET_ID,
        "evidence_tier": EVIDENCE_TIER,
        "reserved_period": {"months": reserved_months, "n_months": 3},
        "train_months_used": {"start": train_months[0], "end": train_months[-1], "n_months": len(train_months)},
        "chronology_preserved": True,
        "n_predictions": int(len(long_df)),
        "n_categories": len(CATEGORIES),
        "preprocessing_recipe": "none (Naive: predicted = last-observed-month actual, "
                                 "identical value reused for +1/+2/+3)",
        "model_impl_version": "naive_v1",
        "git_commit": git_commit,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "final_metrics": {"combined": combined, "by_horizon": by_horizon, "by_category": by_category},
        "rejected_candidates_not_evaluated_on_reserved_period": REJECTED_CANDIDATES,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=False, default=str)

    return result


if __name__ == "__main__":
    result = run()
    c = result["final_metrics"]["combined"]
    print(f"FINAL reserved period {result['reserved_period']['months']}: "
          f"combined WAPE={c['wape']:.4f} MAE={c['mae']:.2f} n={c['n']}")
    for h, m in result["final_metrics"]["by_horizon"].items():
        print(f"  +{h}: WAPE={m['wape']:.4f} MAE={m['mae']:.2f}")
