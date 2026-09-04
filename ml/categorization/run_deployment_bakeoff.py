"""
ML-F: deployment-oriented categorization bake-off.

Orchestrates: load the deployment benchmark (ml/data/build_deployment_
benchmark.py's output) -> freeze/verify merchant-grouped, category-stratified
split -> fit each ML-F candidate (A-F, per the pre-registered plan) on TRAIN
only -> select the winner on VALIDATION using the rule fixed BEFORE looking
at FINAL_TEST -> evaluate the winner exactly once on the sealed FINAL_TEST ->
run the same winning recipe on the existing Tier B benchmark for continuity
(never used to pick the winner).

Ambiguous rows (`is_ambiguous=True`, blank `true_category`) are excluded from
every categorical accuracy/macro-F1 computation and reported separately as a
routing/coverage question (ML-F-A audit §14/§16, ML-F brief §10/§16): does a
deterministic rule -- not the ML classifier -- correctly catch them, since
they carry no learnable spending-purpose label at all.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from config import CATEGORIES
from ml.categorization.candidates import (
    TfidfLinearSVMCandidate,
    TfidfLogRegCandidate,
    TfidfWordCharLogRegCandidate,
)
from ml.categorization.text_normalize import normalize_deployment_text
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
BENCHMARK_CSV = REPO_ROOT / "data" / "evaluation" / "deployment_benchmark.csv"
SPLIT_PATH = REPO_ROOT / "data" / "evaluation" / "deployment_split_v1.json"
RESULTS_PATH = REPO_ROOT / "reports" / "ml" / "results" / "deployment_categorization_results.json"

DATASET_ID = "deployment_benchmark_v1"
EVIDENCE_TIER = (
    "Sanitized deployment-oriented bank-description benchmark "
    "(hand-curated, fabricated merchants; structurally modeled on the "
    "ML-F-A private real-export audit findings -- NOT real-world data)"
)
SEED = 42


def _clean_merchant(description: str) -> str:
    """Identical normalization to pipeline/ingest.py's _clean_merchant_text
    and run_bakeoff.py's own `_clean_merchant` -- same text shape
    CategorizationService sees in production."""
    m = str(description).strip().upper()
    m = re.sub(r"[^\w\s\-&]", "", m)
    m = re.sub(r"\s+", " ", m).strip()
    return m


def load_benchmark() -> pd.DataFrame:
    df = pd.read_csv(BENCHMARK_CSV, keep_default_na=False)
    df["merchant"] = df["description"].map(_clean_merchant)
    df["is_ambiguous"] = df["is_ambiguous"].astype(str).str.lower().isin(["true", "1"])
    return df


def get_or_build_split(df: pd.DataFrame) -> SplitResult:
    if SPLIT_PATH.exists():
        return SplitResult.load(SPLIT_PATH)
    # Ambiguous rows have true_category="" -- merchant_grouped_stratified_split
    # only needs *a* label per group to stratify by, and "" simply becomes one
    # more stratification bucket, so ambiguous merchant groups are isolated
    # across partitions exactly like every other group.
    split = merchant_grouped_stratified_split(
        df, group_col="merchant_group", category_col="true_category", seed=SEED,
        train_frac=0.60, val_frac=0.20, test_frac=0.20,
    )
    split.save(SPLIT_PATH)
    return split


def assert_final_test_sealed(train_df: pd.DataFrame, val_df: pd.DataFrame, final_df: pd.DataFrame) -> None:
    train_groups = set(train_df["merchant_group"])
    val_groups = set(val_df["merchant_group"])
    final_groups = set(final_df["merchant_group"])
    assert not (train_groups & final_groups), "TRAIN/FINAL_TEST merchant group overlap detected"
    assert not (val_groups & final_groups), "VALIDATION/FINAL_TEST merchant group overlap detected"


def _ambiguous_routing_report(df: pd.DataFrame) -> dict:
    """Deterministic ambiguous-row routing check (ML-F brief §16/§10): the
    SAME rule production uses (backend/services/ambiguity.py's
    is_structurally_ambiguous), evaluated here only as a coverage report --
    never as a categorical prediction, never scored into macro-F1."""
    from backend.services.ambiguity import is_structurally_ambiguous

    ambiguous = df[df["is_ambiguous"]]
    if ambiguous.empty:
        return {"n_ambiguous": 0}
    routed = ambiguous["merchant"].map(is_structurally_ambiguous)
    return {
        "n_ambiguous": int(len(ambiguous)),
        "n_correctly_routed": int(routed.sum()),
        "coverage": float(routed.mean()),
    }


def _fit_eval(candidate, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    candidate.fit(train_df, label_col="true_category")
    train_pred = candidate.predict(train_df)
    val_pred = candidate.predict(val_df)
    return {
        "config": candidate.describe(),
        "train_diagnostic": categorization_metric_bundle(train_df["true_category"].values, train_pred, CATEGORIES),
        "validation": categorization_metric_bundle(val_df["true_category"].values, val_pred, CATEGORIES),
    }


def run() -> dict:
    df = load_benchmark()
    split = get_or_build_split(df)
    df_p = split.apply(df)

    isolation_report = verify_split_isolation(df_p, "merchant_group")
    if not isolation_report["all_intersections_empty"]:
        raise RuntimeError(f"Merchant group leakage detected: {isolation_report['intersections']}")

    all_train = df_p[df_p["partition"] == TRAIN].reset_index(drop=True)
    all_val = df_p[df_p["partition"] == VALIDATION].reset_index(drop=True)
    all_final = df_p[df_p["partition"] == FINAL_TEST].reset_index(drop=True)
    assert_final_test_sealed(all_train, all_val, all_final)

    # Categorical candidates are fit/evaluated on non-ambiguous rows only
    # (blank true_category rows have no learnable spending-purpose label).
    train_df = all_train[~all_train["is_ambiguous"]].reset_index(drop=True)
    val_df = all_val[~all_val["is_ambiguous"]].reset_index(drop=True)
    final_df = all_final[~all_final["is_ambiguous"]].reset_index(drop=True)

    results = {
        "dataset_id": DATASET_ID,
        "evidence_tier": EVIDENCE_TIER,
        "seed": SEED,
        "split_definition_ref": str(SPLIT_PATH.relative_to(REPO_ROOT)),
        "partition_counts_all_rows": isolation_report["row_counts"],
        "partition_counts_categorical_only": {
            "TRAIN": len(train_df), "VALIDATION": len(val_df), "FINAL_TEST": len(final_df),
        },
        "partition_merchant_group_counts": isolation_report["unique_merchant_groups"],
        "final_test_sealed": True,
        "ambiguous_routing": {
            "TRAIN": _ambiguous_routing_report(all_train),
            "VALIDATION": _ambiguous_routing_report(all_val),
            "FINAL_TEST": _ambiguous_routing_report(all_final),
        },
        "candidates": {},
    }

    # ---- Candidates A-D (E only if C or D beats A/B; F only on the winner) ----
    candidates = {
        "A_baseline_word_tfidf_50": TfidfLogRegCandidate(candidate_name="A_baseline_word_tfidf_50"),
        "B_word_tfidf_100": TfidfLogRegCandidate(candidate_name="B_word_tfidf_100", tfidf_overrides={"max_features": 100}),
        "B_word_tfidf_200": TfidfLogRegCandidate(candidate_name="B_word_tfidf_200", tfidf_overrides={"max_features": 200}),
        "B_word_tfidf_400": TfidfLogRegCandidate(candidate_name="B_word_tfidf_400", tfidf_overrides={"max_features": 400}),
        "B_word_tfidf_unbounded": TfidfLogRegCandidate(candidate_name="B_word_tfidf_unbounded", tfidf_overrides={"max_features": None}),
        "C_normalized_word_tfidf_200": TfidfLogRegCandidate(
            candidate_name="C_normalized_word_tfidf_200",
            tfidf_overrides={"max_features": 200},
            normalize_fn=normalize_deployment_text,
        ),
        "D_char_tfidf_3_5_300": TfidfLogRegCandidate(
            candidate_name="D_char_tfidf_3_5_300",
            tfidf_overrides={"analyzer": "char_wb", "ngram_range": (3, 5), "max_features": 300},
        ),
    }

    for name, candidate in candidates.items():
        results["candidates"][name] = _fit_eval(candidate, train_df, val_df)
        v = results["candidates"][name]["validation"]
        log_experiment(
            experiment_id=f"mlF_{name}", dataset_id=DATASET_ID, evidence_tier=EVIDENCE_TIER, seed=SEED,
            status="SUCCESS", metrics={"validation_macro_f1": v["macro_f1"], "validation_accuracy": v["accuracy"]},
            partition_definition_ref=str(SPLIT_PATH.relative_to(REPO_ROOT)), model=name,
            hyperparameters=candidate.describe(), notes="ML-F deployment-oriented VALIDATION evaluation.",
        )

    # ---- Pre-registered winner rule among A-D (ML-F brief §11) ----
    ranked = sorted(results["candidates"].items(), key=lambda kv: kv[1]["validation"]["macro_f1"], reverse=True)
    best_name, best_result = ranked[0]
    best_macro_f1 = best_result["validation"]["macro_f1"]
    baseline_macro_f1 = results["candidates"]["A_baseline_word_tfidf_50"]["validation"]["macro_f1"]

    c_or_d_meaningfully_better = any(
        results["candidates"][n]["validation"]["macro_f1"] > baseline_macro_f1 + 0.02
        for n in ("C_normalized_word_tfidf_200", "D_char_tfidf_3_5_300")
        if n in results["candidates"]
    )

    if c_or_d_meaningfully_better:
        # ---- Candidate E: only run because C or D showed a meaningful gain ----
        e_candidate = TfidfWordCharLogRegCandidate(
            word_overrides={"max_features": 200}, normalize_fn=normalize_deployment_text,
        )
        results["candidates"]["E_word_plus_char_tfidf"] = _fit_eval(e_candidate, train_df, val_df)
        ranked = sorted(results["candidates"].items(), key=lambda kv: kv[1]["validation"]["macro_f1"], reverse=True)
        best_name, best_result = ranked[0]
        best_macro_f1 = best_result["validation"]["macro_f1"]

    results["ranked_by_validation_macro_f1"] = [(n, r["validation"]["macro_f1"]) for n, r in ranked]

    # ---- Tie-break rule (ML-F brief §11): simplicity, then LogReg over SVM ----
    TIE_EPSILON = 0.01
    tied = [n for n, r in ranked if best_macro_f1 - r["validation"]["macro_f1"] <= TIE_EPSILON]
    simplicity_order = [
        "A_baseline_word_tfidf_50", "B_word_tfidf_100", "B_word_tfidf_200", "B_word_tfidf_400",
        "B_word_tfidf_unbounded", "C_normalized_word_tfidf_200", "D_char_tfidf_3_5_300",
        "E_word_plus_char_tfidf",
    ]
    for name in simplicity_order:
        if name in tied:
            winner_name = name
            break
    else:
        winner_name = best_name

    # ---- Candidate F: SVM confirmation pass on the winning representation ----
    winner_candidate_cfg = results["candidates"][winner_name]["config"]
    if winner_name.startswith("E_"):
        pass  # F is a LogReg-vs-SVM comparison; skip on the word+char union to keep the search small, per ML-F brief §9
    else:
        svm_candidate = TfidfLinearSVMCandidate(
            tfidf_overrides=dict(winner_candidate_cfg.get("tfidf_config") or {}),
            normalize_fn=normalize_deployment_text if winner_candidate_cfg.get("normalize_fn") else None,
        )
        results["candidates"]["F_linear_svm_on_winner_repr"] = _fit_eval(svm_candidate, train_df, val_df)
        f_macro_f1 = results["candidates"]["F_linear_svm_on_winner_repr"]["validation"]["macro_f1"]
        # Winner rule: prefer LogReg over SVM unless SVM wins outright by more
        # than the tie epsilon (ML-F brief §11).
        if f_macro_f1 > best_macro_f1 + TIE_EPSILON:
            winner_name = "F_linear_svm_on_winner_repr"

    results["winner"] = winner_name
    results["winner_selection_reasoning"] = (
        f"Highest deployment-oriented VALIDATION macro-F1 among A-F, tie-broken "
        f"(within {TIE_EPSILON}) toward simpler representations and LogReg over "
        f"SVM, per the pre-registered rule (ML-F brief Section 11). Selected "
        f"BEFORE viewing FINAL_TEST."
    )

    # ---- Sealed FINAL_TEST: evaluate the frozen winner exactly once ----
    winner_cfg = results["candidates"][winner_name]["config"]
    winner_candidate = _rebuild_candidate_from_config(winner_name, winner_cfg)
    winner_candidate.fit(train_df, label_col="true_category")
    final_pred = winner_candidate.predict(final_df)
    results["final_test"] = categorization_metric_bundle(final_df["true_category"].values, final_pred, CATEGORIES)
    results["final_test"]["n_merchant_groups"] = int(final_df["merchant_group"].nunique())
    results["final_test_ambiguous_routing"] = _ambiguous_routing_report(all_final)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=str)

    return results


def _sanitize_tfidf_overrides(overrides: dict | None) -> dict | None:
    """A config round-tripped through JSON (e.g. reports/ml/
    ML_F_SELECTION_RECORD.json) turns `ngram_range`'s tuple into a list --
    sklearn's TfidfVectorizer requires an actual tuple. Coerce it back."""
    if not overrides:
        return overrides
    out = dict(overrides)
    if "ngram_range" in out and isinstance(out["ngram_range"], list):
        out["ngram_range"] = tuple(out["ngram_range"])
    return out


def _rebuild_candidate_from_config(name: str, cfg: dict):
    """Reconstruct a fresh (unfit) candidate instance from its own
    describe() output (or the equivalent shape loaded back from a frozen
    JSON selection record), so the sealed FINAL_TEST fit is a clean refit on
    TRAIN only -- never reusing an object that has already seen VALIDATION
    through .predict() (defense in depth; predict() never mutates fitted
    state, but a fresh instance removes any doubt)."""
    if name.startswith("E_"):
        return TfidfWordCharLogRegCandidate(
            word_overrides=_sanitize_tfidf_overrides(cfg["word_tfidf_config"]),
            char_overrides=_sanitize_tfidf_overrides(cfg["char_tfidf_config"]),
            normalize_fn=normalize_deployment_text if cfg.get("normalize_fn") else None,
        )
    if name.startswith("F_"):
        return TfidfLinearSVMCandidate(
            tfidf_overrides=_sanitize_tfidf_overrides(cfg["tfidf_config"]),
            normalize_fn=normalize_deployment_text if cfg.get("normalize_fn") else None,
        )
    return TfidfLogRegCandidate(
        candidate_name=name, tfidf_overrides=_sanitize_tfidf_overrides(cfg["tfidf_config"]),
        normalize_fn=normalize_deployment_text if cfg.get("normalize_fn") else None,
    )


def run_tier_b_continuity(winner_name: str, winner_cfg: dict) -> dict:
    """ML-F brief §10/§12: run the FINAL selected production representation
    on the existing Tier B benchmark for continuity -- Tier B is never used
    to pick the winner, only reported alongside it."""
    from ml.categorization.run_bakeoff import get_or_build_split as tier_b_split
    from ml.categorization.run_bakeoff import load_benchmark as load_tier_b

    tb = load_tier_b()
    split = tier_b_split(tb)
    tb_p = split.apply(tb)
    train_df = tb_p[tb_p["partition"] == TRAIN].reset_index(drop=True)
    val_df = tb_p[tb_p["partition"] == VALIDATION].reset_index(drop=True)
    final_df = tb_p[tb_p["partition"] == FINAL_TEST].reset_index(drop=True)

    candidate = _rebuild_candidate_from_config(winner_name, winner_cfg)
    candidate.fit(train_df, label_col="true_category")
    val_pred = candidate.predict(val_df)
    final_pred = candidate.predict(final_df)
    return {
        "dataset_id": "tier_b_benchmark_v1",
        "note": "Continuity evaluation only -- Tier B was NOT used to select the ML-F winner.",
        "validation": categorization_metric_bundle(val_df["true_category"].values, val_pred, CATEGORIES),
        "final_test": categorization_metric_bundle(final_df["true_category"].values, final_pred, CATEGORIES),
    }


if __name__ == "__main__":
    results = run()
    print("=== ML-F deployment-oriented VALIDATION ranking ===")
    for name, f1 in results["ranked_by_validation_macro_f1"]:
        print(f"  {name}: macro_f1={f1:.4f}")
    print(f"\nWinner: {results['winner']}")
    print(f"Reasoning: {results['winner_selection_reasoning']}")
    print(f"\nSealed FINAL_TEST: accuracy={results['final_test']['accuracy']:.4f} macro_f1={results['final_test']['macro_f1']:.4f} n={results['final_test']['n']}")
    print(f"Ambiguous-row routing on FINAL_TEST: {results['final_test_ambiguous_routing']}")

    tb = run_tier_b_continuity(results["winner"], results["candidates"][results["winner"]]["config"])
    print(f"\nTier B continuity: VALIDATION macro_f1={tb['validation']['macro_f1']:.4f} FINAL_TEST macro_f1={tb['final_test']['macro_f1']:.4f}")

    RESULTS_PATH_TB = REPO_ROOT / "reports" / "ml" / "results" / "deployment_winner_tier_b_continuity.json"
    with open(RESULTS_PATH_TB, "w") as f:
        json.dump(tb, f, indent=2, sort_keys=True, default=str)
