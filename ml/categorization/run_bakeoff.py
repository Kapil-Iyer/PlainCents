"""
ML-B Part A: categorization bake-off runner.

Orchestrates: load Tier B benchmark -> freeze/verify merchant-grouped
category-stratified split -> fit each of the 3 frozen candidates on TRAIN
only -> evaluate on VALIDATION -> structured error analysis (Section 8) ->
persist machine-readable results. FINAL_TEST rows are loaded (they must
exist somewhere in memory to be excluded correctly) but never passed to any
candidate's .fit()/.predict(), and never scored -- see
`assert_final_test_sealed` below, which is also exercised by tests/ml.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from config import CATEGORIES
from ml.categorization.candidates import (
    KMeansCandidate,
    TfidfLinearSVMCandidate,
    TfidfLogRegCandidate,
)
from ml.common.experiment_log import log_experiment
from ml.common.metrics import categorization_metric_bundle
from ml.common.splitting import (
    FINAL_TEST,
    TRAIN,
    VALIDATION,
    SplitResult,
    merchant_grouped_stratified_split,
    verify_split_isolation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_CSV = REPO_ROOT / "data" / "evaluation" / "tier_b_benchmark.csv"
SPLIT_PATH = REPO_ROOT / "data" / "evaluation" / "tier_b_split_v1.json"
RESULTS_PATH = REPO_ROOT / "reports" / "ml" / "results" / "categorization_results.json"
ERROR_ANALYSIS_PATH = REPO_ROOT / "reports" / "ml" / "results" / "categorization_error_analysis.json"

DATASET_ID = "tier_b_benchmark_v1"
EVIDENCE_TIER = "Tier B (independently curated/constructed benchmark; NOT real-world data, ML Spec Section 3.2)"
SEED = 42


def _clean_merchant(description: str) -> str:
    """Same normalization ingest.py applies to the `merchant` column
    (ingest.py's load_and_clean, steps 5): strip, uppercase, strip illegal
    chars, collapse whitespace. Reused as plain string logic (no fitting
    involved) so candidates see the same text shape CategorizationService
    sees in production. pipeline/ingest.py itself is not imported/modified
    here to keep this module fully decoupled from any future production
    change to that file."""
    m = str(description).strip().upper()
    m = re.sub(r"[^\w\s\-&]", "", m)
    m = re.sub(r"\s+", " ", m).strip()
    return m


def load_benchmark() -> pd.DataFrame:
    df = pd.read_csv(BENCHMARK_CSV, keep_default_na=False)
    df["merchant"] = df["description"].map(_clean_merchant)
    return df


def get_or_build_split(df: pd.DataFrame) -> SplitResult:
    if SPLIT_PATH.exists():
        return SplitResult.load(SPLIT_PATH)
    split = merchant_grouped_stratified_split(
        df, group_col="merchant_group", category_col="true_category", seed=SEED,
        train_frac=0.60, val_frac=0.20, test_frac=0.20,
    )
    split.save(SPLIT_PATH)
    return split


def assert_final_test_sealed(train_df: pd.DataFrame, val_df: pd.DataFrame, final_df: pd.DataFrame) -> None:
    """Defense in depth: raise loudly if FINAL_TEST rows ever end up
    indistinguishable from TRAIN/VALIDATION by merchant group (would mean
    the split itself leaked)."""
    train_groups = set(train_df["merchant_group"])
    val_groups = set(val_df["merchant_group"])
    final_groups = set(final_df["merchant_group"])
    assert not (train_groups & final_groups), "TRAIN/FINAL_TEST merchant group overlap detected"
    assert not (val_groups & final_groups), "VALIDATION/FINAL_TEST merchant group overlap detected"


def run() -> dict:
    df = load_benchmark()
    split = get_or_build_split(df)
    df_p = split.apply(df)

    isolation_report = verify_split_isolation(df_p, "merchant_group")
    if not isolation_report["all_intersections_empty"]:
        raise RuntimeError(f"Merchant group leakage detected: {isolation_report['intersections']}")

    train_df = df_p[df_p["partition"] == TRAIN].reset_index(drop=True)
    val_df = df_p[df_p["partition"] == VALIDATION].reset_index(drop=True)
    final_df = df_p[df_p["partition"] == FINAL_TEST].reset_index(drop=True)
    assert_final_test_sealed(train_df, val_df, final_df)

    candidates = {
        "kmeans": KMeansCandidate(),
        "tfidf_logreg": TfidfLogRegCandidate(),
        "tfidf_linear_svm": TfidfLinearSVMCandidate(),
    }

    results = {
        "dataset_id": DATASET_ID,
        "evidence_tier": EVIDENCE_TIER,
        "seed": SEED,
        "split_definition_ref": str(SPLIT_PATH.relative_to(REPO_ROOT)),
        "partition_counts": isolation_report["row_counts"],
        "partition_merchant_group_counts": isolation_report["unique_merchant_groups"],
        "final_test_sealed": True,
        "candidates": {},
    }

    error_analysis = {}

    for name, candidate in candidates.items():
        candidate.fit(train_df, label_col="true_category")
        train_pred = candidate.predict(train_df)
        val_pred = candidate.predict(val_df)

        train_bundle = categorization_metric_bundle(train_df["true_category"].values, train_pred, CATEGORIES)
        val_bundle = categorization_metric_bundle(val_df["true_category"].values, val_pred, CATEGORIES)

        results["candidates"][name] = {
            "config": candidate.describe(),
            "train_diagnostic": train_bundle,
            "validation": val_bundle,
        }

        log_experiment(
            experiment_id=f"catA_{name}",
            dataset_id=DATASET_ID,
            evidence_tier=EVIDENCE_TIER,
            seed=SEED,
            status="SUCCESS",
            metrics={"validation_macro_f1": val_bundle["macro_f1"], "validation_accuracy": val_bundle["accuracy"]},
            partition_definition_ref=str(SPLIT_PATH.relative_to(REPO_ROOT)),
            model=name,
            hyperparameters=candidate.describe(),
            notes="Categorization VALIDATION evaluation (ML Spec Section 7).",
        )

        # Structured error analysis (Section 8) -- VALIDATION misclassifications only.
        val_df_copy = val_df.copy()
        val_df_copy["predicted_category"] = val_pred
        errors = val_df_copy[val_df_copy["predicted_category"] != val_df_copy["true_category"]]
        error_rows = []
        for _, row in errors.iterrows():
            error_rows.append({
                "merchant_description": row["description"],
                "amount": row["amount"],
                "true_category": row["true_category"],
                "predicted_category": row["predicted_category"],
                "candidate_model": name,
                "error_analysis_tag": row["error_analysis_tag"],
            })
        error_analysis[name] = {
            "n_validation_rows": len(val_df),
            "n_errors": len(errors),
            "error_rate": len(errors) / len(val_df) if len(val_df) else float("nan"),
            "errors": error_rows,
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=str)

    with open(ERROR_ANALYSIS_PATH, "w") as f:
        json.dump(error_analysis, f, indent=2, sort_keys=True, default=str)

    return results


if __name__ == "__main__":
    results = run()
    for name, r in results["candidates"].items():
        v = r["validation"]
        print(f"{name}: VALIDATION macro_f1={v['macro_f1']:.4f} accuracy={v['accuracy']:.4f}")
