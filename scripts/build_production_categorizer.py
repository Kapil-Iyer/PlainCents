"""
ML-G Production Integration: builds the production categorization artifact
(config.CATEGORIZER_MODEL_PATH, models/categorizer_v3.pkl).

Supersedes scripts/build_production_logreg_model.py, which fit ML-F's
word-only, 200-term recipe on the v1 deployment corpus. That script and the
ML-C/ML-F selection records are left untouched as historical evidence; this
one neither reads nor writes them.

THE ARTIFACT NOW CARRIES ITS OWN DECISION CONTRACT. ML-F's payload held only
{vectorizer, model}, which meant CategorizationService had to *assume* how to
prepare text and what to do with a low-confidence prediction. It assumed "no
normalization, always answer" -- so a recipe that normalized its input could
never have been served correctly, and an evidence-free row was always
answered with argmax(intercept_). This payload instead records:

    normalizer_name   which text normalizer the vectorizer was FIT with,
                      resolved back to the identical function at inference
    min_margin        the abstention threshold fitted on VALIDATION
    categories        the taxonomy the model was trained against

so the served decision is the selected decision, by construction.

TRAINING-DATA DISCIPLINE: TRAIN only. VALIDATION and FINAL_TEST rows are
loaded (the split has to be reconstructed to know which rows are TRAIN) but
never passed to .fit(). FINAL_TEST labels are never read here at all.

Refuses to run unless reports/ml/ML_G_SELECTION_RECORD.json exists and names
a winner whose configuration is reconstructible -- a production artifact is
never built for a recipe that was not frozen as the selected winner.

Run (from the repo root, so `config`/`ml` resolve on sys.path):
    python -m scripts.build_production_categorizer
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CATEGORIES, CATEGORIZER_MODEL_PATH  # noqa: E402
from ml.categorization.candidates_v2 import rebuild_candidate  # noqa: E402
from ml.categorization.run_mlg_bakeoff import (  # noqa: E402
    FINAL_TEST,
    TRAIN,
    VALIDATION,
    assert_final_test_sealed,
    get_or_build_split,
    load_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTION_RECORD_PATH = REPO_ROOT / "reports" / "ml" / "ML_G_SELECTION_RECORD.json"

MODEL_IMPL_VERSION = "tfidf_word_char_logreg_v3"


class SelectionNotFrozenError(RuntimeError):
    """Raised when an artifact build is attempted before the ML-G selection
    record exists and names a reconstructible winner."""


def load_and_verify_selection(path: Path = SELECTION_RECORD_PATH) -> dict:
    if not path.exists():
        raise SelectionNotFrozenError(
            f"{path} does not exist -- the ML-G selection must be frozen before a "
            "production artifact may be built."
        )
    selection = json.loads(path.read_text(encoding="utf-8"))
    winner = selection.get("winner", {})
    if not winner.get("candidate_name") or not winner.get("exact_configuration"):
        raise SelectionNotFrozenError(
            f"{path} does not name a reconstructible winner -- refusing to build a "
            "production artifact for an unfrozen selection."
        )
    if "min_margin" not in (selection.get("abstention_policy") or {}):
        raise SelectionNotFrozenError(
            f"{path} does not record an abstention policy -- refusing to build an "
            "artifact whose served decision would differ from the evaluated one."
        )
    return selection


def build(out_path: Path = CATEGORIZER_MODEL_PATH,
          selection_record_path: Path = SELECTION_RECORD_PATH) -> dict:
    selection = load_and_verify_selection(selection_record_path)
    winner_cfg = selection["winner"]["exact_configuration"]
    policy = selection["abstention_policy"]

    df = load_benchmark()
    df_p = get_or_build_split(df).apply(df)
    df_p = df_p[~df_p["is_ambiguous"]].reset_index(drop=True)

    train_df = df_p[df_p["partition"] == TRAIN].reset_index(drop=True)
    val_df = df_p[df_p["partition"] == VALIDATION]
    final_df = df_p[df_p["partition"] == FINAL_TEST]
    assert_final_test_sealed(train_df, val_df, final_df)

    candidate = rebuild_candidate(winner_cfg).fit(train_df, label_col="true_category")

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        git_commit = None

    metadata = {
        "model_impl_version": MODEL_IMPL_VERSION,
        "family": "Word TF-IDF + character TF-IDF (FeatureUnion) + Logistic Regression",
        "candidate_name": selection["winner"]["candidate_name"],
        "selection_record_ref": "reports/ml/ML_G_SELECTION_RECORD.json",
        "dataset_id": selection["dataset"]["dataset_id"],
        "dataset_ref": selection["dataset"]["source"],
        "split_definition_ref": selection["dataset"]["split_definition_ref"],
        "fit_partition": "TRAIN",
        "fit_partition_n_rows": int(len(train_df)),
        "fit_partition_n_merchant_groups": int(train_df["merchant_group"].nunique()),
        "vocabulary_size": candidate.vocabulary_size,
        "recipe": candidate.describe(),
        "decision_policy": {
            "normalizer_name": winner_cfg.get("normalizer_name"),
            "min_margin": policy["min_margin"],
            "zero_feature_rule": policy["zero_feature_rule"],
            "abstain_category": policy["abstain_category"],
        },
        "category_taxonomy": list(CATEGORIES),
        "validation_macro_f1": selection["candidates_evaluated"][
            selection["winner"]["candidate_name"]]["validation_macro_f1"],
        "sealed_final_test_macro_f1_model_only":
            selection["sealed_final_test_model_only"]["macro_f1"],
        "sealed_final_test_macro_f1_with_policy":
            selection["sealed_final_test_with_policy"]["macro_f1"],
        "git_commit": git_commit,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "known_limitations": selection["what_still_cannot_be_inferred"],
    }

    payload = {
        "vectorizer": candidate._vectorizer,
        "model": candidate._model,
        "model_impl_version": MODEL_IMPL_VERSION,
        # The served decision contract -- see the module docstring.
        "normalizer_name": winner_cfg.get("normalizer_name"),
        "min_margin": policy["min_margin"],
        "abstain_category": policy["abstain_category"],
        "categories": list(CATEGORIES),
        "metadata": metadata,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)
    return metadata


if __name__ == "__main__":
    meta = build()
    print(f"Production categorizer written: {CATEGORIZER_MODEL_PATH}")
    print(f"  model_impl_version = {meta['model_impl_version']}")
    print(f"  fit on TRAIN only  = {meta['fit_partition_n_rows']} rows / "
          f"{meta['fit_partition_n_merchant_groups']} merchant groups")
    print(f"  vocabulary size    = {meta['vocabulary_size']}")
    print(f"  normalizer         = {meta['decision_policy']['normalizer_name']}")
    print(f"  min_margin         = {meta['decision_policy']['min_margin']}")
    print(f"  sealed FINAL_TEST macro-F1 = "
          f"{meta['sealed_final_test_macro_f1_model_only']:.4f} model-only / "
          f"{meta['sealed_final_test_macro_f1_with_policy']:.4f} with policy")
    print(f"  git_commit         = {meta['git_commit']}")
