"""
ML-F Production Integration: builds the production categorization artifact
(config.LOGREG_MODEL_PATH, models/tfidf_logreg_v2.pkl) — the ML-F selected
recipe (word TF-IDF, max_features=200, ngram_range=(1,2), sublinear_tf, L2 —
same family as ML-C's TfidfLogRegCandidate, just a larger vocabulary), fit on
the deployment-oriented benchmark's TRAIN partition (ml/data/build_deployment_
benchmark.py; data/evaluation/deployment_split_v1.json), using the exact
hyperparameters recorded in reports/ml/ML_F_SELECTION_RECORD.json's
winner.exact_configuration.

Supersedes ML-D's scripts/build_production_logreg_model.py behavior (which
fit on Tier B's TRAIN partition, data/evaluation/tier_b_split_v1.json). Tier B
and the ML-C selection record are untouched historical evidence — this script
no longer reads or writes them; reports/ml/ML_F_SELECTION_RECORD.json's own
`tier_b_continuity_evaluation` field is where Tier B still gets checked (as a
continuity report, never as the fitting data).

Production training-data discipline (ML Spec Section 6 principle, extended by
ML-F): TRAIN only. VALIDATION and FINAL_TEST rows are loaded (get_or_build_split
reconstructs the full partitioned frame) but are never passed to .fit().
FINAL_TEST labels are never used here in any way.

Refuses to run unless reports/ml/ML_F_SELECTION_RECORD.json exists and names a
winner whose config is reconstructible via
ml.categorization.run_deployment_bakeoff._rebuild_candidate_from_config — a
production artifact is never built for a candidate that was not frozen as the
ML-F winner.

Run (from repo root, so `config`/`ml` resolve on sys.path):
    python -m scripts.build_production_logreg_model
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CATEGORIES, LOGREG_MODEL_PATH
from ml.categorization.run_deployment_bakeoff import (
    TRAIN,
    VALIDATION,
    FINAL_TEST,
    _rebuild_candidate_from_config,
    assert_final_test_sealed,
    get_or_build_split,
    load_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTION_RECORD_PATH = REPO_ROOT / "reports" / "ml" / "ML_F_SELECTION_RECORD.json"

MODEL_IMPL_VERSION = "tfidf_logreg_v2"


class SelectionNotFrozenError(RuntimeError):
    """Raised when a production artifact build is attempted before the ML-F
    selection record exists and names a reconstructible winner."""


def load_and_verify_selection(selection_record_path: Path = SELECTION_RECORD_PATH) -> dict:
    if not selection_record_path.exists():
        raise SelectionNotFrozenError(
            f"{selection_record_path} does not exist -- the ML-F selection must be frozen "
            "before a production artifact may be built."
        )
    with open(selection_record_path, encoding="utf-8") as f:
        selection = json.load(f)
    winner = selection.get("winner", {})
    if not winner.get("candidate_name"):
        raise SelectionNotFrozenError(
            f"{selection_record_path} does not name a winner candidate -- refusing to build "
            "a production artifact for an unfrozen selection."
        )
    return selection


def build(out_path: Path = LOGREG_MODEL_PATH, selection_record_path: Path = SELECTION_RECORD_PATH) -> dict:
    selection = load_and_verify_selection(selection_record_path)
    winner_name = selection["winner"]["candidate_name"]
    winner_cfg = selection["winner"]["exact_configuration"]

    df = load_benchmark()
    split = get_or_build_split(df)
    df_p = split.apply(df)
    df_p = df_p[~df_p["is_ambiguous"]].reset_index(drop=True)  # never fit/eval categorical candidates on ambiguous rows

    train_df = df_p[df_p["partition"] == TRAIN].reset_index(drop=True)
    val_df = df_p[df_p["partition"] == VALIDATION]
    final_df = df_p[df_p["partition"] == FINAL_TEST]

    assert not (set(train_df["merchant_group"]) & set(val_df["merchant_group"])), \
        "TRAIN/VALIDATION merchant group overlap detected"
    assert not (set(train_df["merchant_group"]) & set(final_df["merchant_group"])), \
        "TRAIN/FINAL_TEST merchant group overlap detected"
    assert_final_test_sealed(train_df, val_df, final_df)

    # _rebuild_candidate_from_config expects the shape produced by a
    # candidate's own .describe() (tfidf_config/word_tfidf_config/etc, not
    # the flattened exact_configuration this record stores) -- adapt.
    describe_shaped_cfg = {
        "tfidf_config": winner_cfg["tfidf_config"],
        "normalize_fn": winner_cfg.get("normalize_fn"),
    }
    candidate = _rebuild_candidate_from_config(winner_name, describe_shaped_cfg)
    candidate.C = winner_cfg["C"]
    candidate.max_iter = winner_cfg["max_iter"]
    candidate.random_state = winner_cfg["random_state"]
    candidate.fit(train_df, label_col="true_category")  # TRAIN only

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        git_commit = None

    metadata = {
        "model_impl_version": MODEL_IMPL_VERSION,
        "family": "TF-IDF + Logistic Regression",
        "candidate_name": winner_name,
        "selection_record_ref": "reports/ml/ML_F_SELECTION_RECORD.json",
        "dataset_id": "deployment_benchmark_v1",
        "dataset_ref": "ml/data/build_deployment_benchmark.py -> data/evaluation/deployment_benchmark.csv",
        "split_definition_ref": "data/evaluation/deployment_split_v1.json",
        "tier_b_continuity_ref": "reports/ml/ML_F_SELECTION_RECORD.json (tier_b_continuity_evaluation field) — Tier B was NOT used to select this recipe",
        "fit_partition": "TRAIN",
        "fit_partition_n_rows": int(len(train_df)),
        "fit_partition_n_merchant_groups": int(train_df["merchant_group"].nunique()),
        "recipe": candidate.describe(),
        "category_taxonomy": list(CATEGORIES),
        "git_commit": git_commit,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "known_limitations": [
            "Sanitized/curated training data (not real-world transactions) — no real-world accuracy claim.",
            "Small-data regime (n=190 rows / 73 merchant groups across TRAIN+VALIDATION+FINAL_TEST) — held-out macro-F1 is modest and honestly reported, not inflated.",
            "Structurally-ambiguous rows (generic e-transfer, ABM/ATM withdrawal) are routed deterministically to 'Other' upstream of this model (backend/services/ambiguity.py), not solved by the classifier.",
        ],
    }

    payload = {
        "vectorizer": candidate._vectorizer,
        "model": candidate._model,
        "model_impl_version": MODEL_IMPL_VERSION,
        "metadata": metadata,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)
    return metadata


if __name__ == "__main__":
    meta = build()
    print(f"Production LogReg artifact written: {LOGREG_MODEL_PATH}")
    print(
        f"Fit on TRAIN only: {meta['fit_partition_n_rows']} rows, "
        f"{meta['fit_partition_n_merchant_groups']} merchant groups"
    )
    print(f"model_impl_version={meta['model_impl_version']} git_commit={meta['git_commit']}")
