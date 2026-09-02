"""
ML-D Production Integration: builds the production categorization artifact
(config.LOGREG_MODEL_PATH, models/tfidf_logreg_v1.pkl) — the ML-C selected
TF-IDF + Logistic Regression recipe (ml/categorization/candidates.py::
TfidfLogRegCandidate), fit strictly on the frozen Tier B TRAIN partition
(data/evaluation/tier_b_split_v1.json — 133 rows / 47 merchant groups), using
the exact hyperparameters recorded in reports/ml/ML_C_SELECTION_RECORD.json's
categorization_selection.exact_configuration.

Production training-data discipline (ML Spec Section 6; ML-D Section 4): the
DEFAULT policy applies here because the frozen ML Spec does not explicitly
authorize a TRAIN+VALIDATION production refit — TRAIN only. VALIDATION and
FINAL_TEST rows are loaded (they must exist in memory to reconstruct/verify
the split) but are never passed to .fit(). This mirrors
ml/categorization/run_final.py's own isolation discipline exactly, just
fitting once more to produce a long-lived artifact for the running service
to load, rather than to score against FINAL_TEST. FINAL_TEST labels are
never used here in any way.

Refuses to run unless the ML-C selection record exists and names
"tfidf_logreg" as selected — a production artifact is never built for a
candidate that was not frozen as the selection.

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
from ml.categorization.candidates import TfidfLogRegCandidate
from ml.categorization.run_bakeoff import get_or_build_split, load_benchmark
from ml.common.splitting import FINAL_TEST, TRAIN, VALIDATION

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTION_RECORD_PATH = REPO_ROOT / "reports" / "ml" / "ML_C_SELECTION_RECORD.json"

SELECTED_CANDIDATE_NAME = "tfidf_logreg"
MODEL_IMPL_VERSION = "tfidf_logreg_v1"


class SelectionNotFrozenError(RuntimeError):
    """Raised when a production artifact build is attempted before the ML-C
    selection record exists and names this module's candidate as selected."""


def load_and_verify_selection(selection_record_path: Path = SELECTION_RECORD_PATH) -> dict:
    if not selection_record_path.exists():
        raise SelectionNotFrozenError(
            f"{selection_record_path} does not exist -- the ML-C selection must be frozen "
            "before a production artifact may be built."
        )
    with open(selection_record_path, encoding="utf-8") as f:
        selection = json.load(f)
    selected = selection.get("categorization_selection", {}).get("selected_candidate")
    if selected != SELECTED_CANDIDATE_NAME:
        raise SelectionNotFrozenError(
            f"Selection record names {selected!r} as selected, not {SELECTED_CANDIDATE_NAME!r} -- "
            "refusing to build a production artifact for a candidate that was not frozen as the selection."
        )
    return selection


def build(out_path: Path = LOGREG_MODEL_PATH, selection_record_path: Path = SELECTION_RECORD_PATH) -> dict:
    selection = load_and_verify_selection(selection_record_path)
    cfg = selection["categorization_selection"]["exact_configuration"]

    df = load_benchmark()
    split = get_or_build_split(df)
    df_p = split.apply(df)

    train_df = df_p[df_p["partition"] == TRAIN].reset_index(drop=True)
    val_df = df_p[df_p["partition"] == VALIDATION]
    final_df = df_p[df_p["partition"] == FINAL_TEST]

    # Defense in depth, mirrors ml/categorization/run_final.py: the artifact
    # actually shipped to production must never be fit on a merchant group
    # that also appears in VALIDATION or FINAL_TEST.
    assert not (set(train_df["merchant_group"]) & set(val_df["merchant_group"])), \
        "TRAIN/VALIDATION merchant group overlap detected"
    assert not (set(train_df["merchant_group"]) & set(final_df["merchant_group"])), \
        "TRAIN/FINAL_TEST merchant group overlap detected"

    candidate = TfidfLogRegCandidate(
        random_state=cfg["random_state"], C=cfg["C"], max_iter=cfg["max_iter"],
    )
    candidate.fit(train_df, label_col="true_category")  # TRAIN only -- never VALIDATION/FINAL_TEST labels

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        git_commit = None

    metadata = {
        "model_impl_version": MODEL_IMPL_VERSION,
        "family": "TF-IDF + Logistic Regression",
        "candidate_name": SELECTED_CANDIDATE_NAME,
        "selection_record_ref": "reports/ml/ML_C_SELECTION_RECORD.json",
        "split_definition_ref": "data/evaluation/tier_b_split_v1.json",
        "fit_partition": "TRAIN",
        "fit_partition_n_rows": int(len(train_df)),
        "fit_partition_n_merchant_groups": int(train_df["merchant_group"].nunique()),
        "recipe": candidate.describe(),
        "category_taxonomy": list(CATEGORIES),
        "git_commit": git_commit,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
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
