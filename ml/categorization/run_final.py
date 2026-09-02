"""
ML-C Part F1: Categorization FINAL evaluation.

Runs the single permitted FINAL pass for categorization (ML Spec Section 20/
21): evaluates ONLY the candidate frozen in reports/ml/ML_C_SELECTION_RECORD.json
against the sealed FINAL_TEST partition (45 rows / 17 merchant groups) of the
frozen ML-B split (data/evaluation/tier_b_split_v1.json).

Refuses to run if:
  - the selection record does not exist yet (selection must be frozen BEFORE
    any FINAL evaluation, per the ML-C brief and ML Spec Section 6/20), or
  - the selection record does not name this module's candidate as selected.

This does not re-fit using FINAL labels: the candidate is fit on TRAIN only
(byte-identical fitting procedure to ML-B's own run_bakeoff.py), then scored
once against FINAL_TEST. No rejected candidate (kmeans, tfidf_linear_svm) is
evaluated by this module -- there is no code path here that could run them
against FINAL_TEST.

Result is labeled precisely per ML Spec Section 21: "Tier B curated
benchmark -- held-out FINAL_TEST", never "real-world" or "temporal
validation" performance.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from config import CATEGORIES
from ml.categorization.candidates import TfidfLogRegCandidate
from ml.categorization.run_bakeoff import get_or_build_split, load_benchmark
from ml.common.metrics import categorization_metric_bundle
from ml.common.splitting import FINAL_TEST, TRAIN, VALIDATION

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SELECTION_RECORD_PATH = REPO_ROOT / "reports" / "ml" / "ML_C_SELECTION_RECORD.json"
OUT_PATH = REPO_ROOT / "reports" / "ml" / "results" / "final_categorization.json"

SELECTED_CANDIDATE_NAME = "tfidf_logreg"


class SelectionNotFrozenError(RuntimeError):
    """Raised when FINAL is attempted before the ML-C selection record exists
    and names this module's candidate as selected."""


def load_and_verify_selection(selection_record_path: Path = SELECTION_RECORD_PATH) -> dict:
    if not selection_record_path.exists():
        raise SelectionNotFrozenError(
            f"{selection_record_path} does not exist -- the ML-C selection must be frozen "
            "before any FINAL evaluation may run."
        )
    with open(selection_record_path, encoding="utf-8") as f:
        selection = json.load(f)
    selected = selection.get("categorization_selection", {}).get("selected_candidate")
    if selected != SELECTED_CANDIDATE_NAME:
        raise SelectionNotFrozenError(
            f"Selection record names {selected!r} as selected, not {SELECTED_CANDIDATE_NAME!r} -- "
            "refusing to run FINAL for a candidate that was not frozen as the selection."
        )
    return selection


def run(selection_record_path: Path = SELECTION_RECORD_PATH, out_path: Path = OUT_PATH) -> dict:
    selection = load_and_verify_selection(selection_record_path)
    cfg = selection["categorization_selection"]["exact_configuration"]

    df = load_benchmark()
    split = get_or_build_split(df)
    df_p = split.apply(df)

    train_df = df_p[df_p["partition"] == TRAIN].reset_index(drop=True)
    val_df = df_p[df_p["partition"] == VALIDATION].reset_index(drop=True)
    final_df = df_p[df_p["partition"] == FINAL_TEST].reset_index(drop=True)

    # Defense in depth -- same isolation check ML-B's own runner performs.
    assert not (set(train_df["merchant_group"]) & set(final_df["merchant_group"])), \
        "TRAIN/FINAL_TEST merchant group overlap detected"
    assert not (set(val_df["merchant_group"]) & set(final_df["merchant_group"])), \
        "VALIDATION/FINAL_TEST merchant group overlap detected"

    candidate = TfidfLogRegCandidate(
        random_state=cfg["random_state"], C=cfg["C"], max_iter=cfg["max_iter"],
    )
    candidate.fit(train_df, label_col="true_category")  # TRAIN only -- no FINAL labels used for fitting

    final_pred = candidate.predict(final_df)
    final_bundle = categorization_metric_bundle(final_df["true_category"].values, final_pred, CATEGORIES)

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        git_commit = None

    result = {
        "result_label": "Tier B curated benchmark — held-out FINAL_TEST",
        "not_to_be_described_as": ["real-world performance", "Tier A performance", "temporal validation performance"],
        "selected_candidate": SELECTED_CANDIDATE_NAME,
        "selected_candidate_label": "TF-IDF + Logistic Regression",
        "selection_record_ref": "reports/ml/ML_C_SELECTION_RECORD.json",
        "dataset_id": "tier_b_benchmark_v1",
        "evidence_tier": "Tier B (independently curated/constructed benchmark; NOT real-world data, ML Spec Section 3.2)",
        "split_definition_ref": "data/evaluation/tier_b_split_v1.json",
        "partition": "FINAL_TEST",
        "partition_n_rows": int(len(final_df)),
        "partition_n_merchant_groups": int(final_df["merchant_group"].nunique()),
        "preprocessing_recipe": candidate.describe(),
        "model_impl_version": "tfidf_logreg_v1",
        "git_commit": git_commit,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "final_metrics": final_bundle,
        "no_refitting_using_final_labels": True,
        "no_mapping_using_final_labels": True,
        "no_error_driven_model_modification_after_seeing_final": True,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=False, default=str)

    return result


if __name__ == "__main__":
    result = run()
    m = result["final_metrics"]
    print(f"FINAL_TEST (n={result['partition_n_rows']}, {result['partition_n_merchant_groups']} merchant groups): "
          f"macro_f1={m['macro_f1']:.4f} accuracy={m['accuracy']:.4f}")
