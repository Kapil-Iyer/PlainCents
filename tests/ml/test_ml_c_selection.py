"""
ML-C infrastructure tests: fold-stability aggregation correctness, the
selection-record-must-exist-before-FINAL gate, and FINAL evaluators
refusing to run for any candidate/strategy other than the one frozen in
reports/ml/ML_C_SELECTION_RECORD.json.

Does not re-test ML-B's own leakage/isolation guarantees (tests/ml/
test_splitting.py, test_kmeans_isolation.py, test_temporal_eval.py,
test_forecast_leakage.py already cover those) -- this file covers only the
ML-C-specific selection/FINAL-gating logic added on top of them.
"""
import json

import pandas as pd
import pytest

import ml.categorization.run_final as cat_final
import ml.forecasting.run_final as fc_final
from ml.forecasting.fold_stability import (
    build_summary,
    compute_per_origin_records,
    summarize_wape_series,
)

# ---------------------------------------------------------------------------
# Fold-stability aggregation correctness
# ---------------------------------------------------------------------------

def _toy_predictions_df():
    """2 origins, 2 candidates (naive, seasonal_naive), horizons 1-2, one
    category -- small enough to hand-verify every WAPE value."""
    rows = [
        # origin 0: naive perfect (actual==predicted), seasonal_naive off by a fixed amount
        dict(origin_index=0, category="X", horizon=1, target_month="2024-01",
             candidate="naive", strategy="n/a", actual=100.0, predicted=100.0, eligible=True),
        dict(origin_index=0, category="X", horizon=2, target_month="2024-02",
             candidate="naive", strategy="n/a", actual=50.0, predicted=50.0, eligible=True),
        dict(origin_index=0, category="X", horizon=1, target_month="2024-01",
             candidate="seasonal_naive", strategy="n/a", actual=100.0, predicted=80.0, eligible=True),
        dict(origin_index=0, category="X", horizon=2, target_month="2024-02",
             candidate="seasonal_naive", strategy="n/a", actual=50.0, predicted=40.0, eligible=True),
        # origin 1: naive off by 10, seasonal_naive ineligible (eligible=False, predicted=NaN)
        dict(origin_index=1, category="X", horizon=1, target_month="2024-02",
             candidate="naive", strategy="n/a", actual=100.0, predicted=90.0, eligible=True),
        dict(origin_index=1, category="X", horizon=2, target_month="2024-03",
             candidate="naive", strategy="n/a", actual=50.0, predicted=45.0, eligible=True),
        dict(origin_index=1, category="X", horizon=1, target_month="2024-02",
             candidate="seasonal_naive", strategy="n/a", actual=100.0, predicted=None, eligible=False),
        dict(origin_index=1, category="X", horizon=2, target_month="2024-03",
             candidate="seasonal_naive", strategy="n/a", actual=50.0, predicted=None, eligible=False),
    ]
    return pd.DataFrame(rows)


def _toy_origin_to_month():
    return {0: "2023-12", 1: "2024-01"}


def test_per_origin_wape_matches_hand_calculation():
    df = _toy_predictions_df()
    records = compute_per_origin_records(df, [0, 1], _toy_origin_to_month())

    naive_records = {r["origin_index"]: r for r in records[("naive", "n/a")]}
    # origin 0: naive is a perfect prediction -> WAPE == 0
    assert naive_records[0]["wape"] == pytest.approx(0.0)
    # origin 1: |100-90| + |50-45| = 15, actual sum = 150 -> WAPE = 15/150 = 0.1
    assert naive_records[1]["wape"] == pytest.approx(0.1)


def test_ineligible_rows_excluded_from_per_origin_metrics():
    df = _toy_predictions_df()
    records = compute_per_origin_records(df, [0, 1], _toy_origin_to_month())
    seasonal_records = {r["origin_index"]: r for r in records[("seasonal_naive", "n/a")]}
    # origin 1's seasonal_naive rows are all eligible=False -> no data for that origin
    assert seasonal_records[1]["n"] == 0
    assert seasonal_records[1]["wape"] is None
    # origin 0's seasonal_naive rows ARE eligible: |100-80|+|50-40| = 30, actual sum=150 -> 0.2
    assert seasonal_records[0]["wape"] == pytest.approx(0.2)


def test_summarize_wape_series_mean_and_median():
    records = [{"wape": 0.1}, {"wape": 0.3}, {"wape": None}]
    summary = summarize_wape_series(records)
    assert summary["n_origins_with_data"] == 2
    assert summary["mean"] == pytest.approx(0.2)
    assert summary["median"] == pytest.approx(0.2)
    assert summary["min"] == pytest.approx(0.1)
    assert summary["max"] == pytest.approx(0.3)


def test_beats_naive_origin_rate_is_correct_head_to_head_count():
    df = _toy_predictions_df()
    records = compute_per_origin_records(df, [0, 1], _toy_origin_to_month())
    summary = build_summary(records)
    # seasonal_naive: origin 0 WAPE=0.2 vs naive's 0.0 (loses); origin 1 has no
    # data (excluded from the comparison entirely, not counted as a loss).
    sn = summary["seasonal_naive__n/a"]
    assert sn["beats_naive_origin_total"] == 1
    assert sn["beats_naive_origin_count"] == 0
    assert sn["beats_naive_origin_rate"] == pytest.approx(0.0)


def test_committed_fold_stability_report_is_internally_consistent():
    """Regression check against the already-generated, committed report:
    Naive's per-origin mean WAPE (0.191...) should be close to, but not
    required to equal, its own pooled/aggregate WAPE (0.190) -- they are
    different aggregation methods (simple mean-of-origins vs.
    volume-weighted pooled sum) and both are expected to exist."""
    from ml.forecasting.fold_stability import REPO_ROOT
    path = REPO_ROOT / "reports" / "ml" / "ML_C_FOLD_STABILITY.json"
    if not path.exists():
        pytest.skip("ML_C_FOLD_STABILITY.json not yet generated in this environment")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    naive = data["per_candidate_strategy_summary"]["naive__n/a"]
    assert naive["wape_stats_across_origins"]["n_origins_with_data"] == 14
    assert naive["beats_naive_origin_count"] == 0  # naive never "beats itself"
    # Every candidate loses to naive in the majority of individual origins at
    # every horizon (the concrete finding cited in the Section 14 eligibility
    # analysis) -- assert this holds for every non-naive candidate/strategy.
    for key, s in data["per_candidate_strategy_summary"].items():
        if key == "naive__n/a":
            continue
        for h, rec in s["per_horizon_beats_naive"].items():
            if rec["total"]:
                assert rec["rate"] < 0.5, f"{key} horizon +{h} unexpectedly beats naive in >=50% of origins"


# ---------------------------------------------------------------------------
# Selection-record-must-exist-before-FINAL gate
# ---------------------------------------------------------------------------

def test_categorization_final_refuses_when_selection_record_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(cat_final.SelectionNotFrozenError):
        cat_final.load_and_verify_selection(missing_path)


def test_forecasting_final_refuses_when_selection_record_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(fc_final.SelectionNotFrozenError):
        fc_final.load_and_verify_selection(missing_path)


def test_categorization_final_refuses_wrong_candidate(tmp_path):
    bad_record = tmp_path / "selection.json"
    bad_record.write_text(json.dumps({"categorization_selection": {"selected_candidate": "kmeans"}}))
    with pytest.raises(cat_final.SelectionNotFrozenError):
        cat_final.load_and_verify_selection(bad_record)


def test_forecasting_final_refuses_wrong_candidate(tmp_path):
    bad_record = tmp_path / "selection.json"
    bad_record.write_text(json.dumps({
        "forecasting_selection": {"selected_candidate": "random_forest"},
        "multi_step_strategy_selection": {"selected_strategy": "last_known_history"},
    }))
    with pytest.raises(fc_final.SelectionNotFrozenError):
        fc_final.load_and_verify_selection(bad_record)


def test_forecasting_final_refuses_correct_candidate_wrong_strategy(tmp_path):
    """Even naming the right candidate isn't enough if the strategy field
    doesn't match what was actually frozen -- guards against a future
    accidental edit that changes the strategy without re-running FINAL."""
    bad_record = tmp_path / "selection.json"
    bad_record.write_text(json.dumps({
        "forecasting_selection": {"selected_candidate": "naive"},
        "multi_step_strategy_selection": {"selected_strategy": "last_known_history"},
    }))
    with pytest.raises(fc_final.SelectionNotFrozenError):
        fc_final.load_and_verify_selection(bad_record)


def test_real_committed_selection_record_is_accepted_by_both_final_gates():
    """The actual frozen reports/ml/ML_C_SELECTION_RECORD.json this ML-C run
    produced must be accepted by both gates -- this is the same file
    ml.categorization.run_final / ml.forecasting.run_final would use by
    default."""
    if not cat_final.SELECTION_RECORD_PATH.exists():
        pytest.skip("ML_C_SELECTION_RECORD.json not yet written in this environment")
    cat_selection = cat_final.load_and_verify_selection()
    fc_selection = fc_final.load_and_verify_selection()
    assert cat_selection["categorization_selection"]["selected_candidate"] == "tfidf_logreg"
    assert fc_selection["forecasting_selection"]["selected_candidate"] == "naive"
    assert fc_selection["multi_step_strategy_selection"]["selected_strategy"] == "N/A"


# ---------------------------------------------------------------------------
# FINAL result metadata (Section 18) is present and correctly labeled
# ---------------------------------------------------------------------------

def test_final_categorization_result_has_required_section_18_metadata():
    path = cat_final.OUT_PATH
    if not path.exists():
        pytest.skip("final_categorization.json not yet generated in this environment")
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    for field in ("dataset_id", "evidence_tier", "split_definition_ref", "selected_candidate",
                  "preprocessing_recipe", "model_impl_version", "git_commit",
                  "evaluation_timestamp_utc", "final_metrics"):
        assert field in result, f"missing required Section 18 field: {field}"
    assert result["result_label"] == "Tier B curated benchmark — held-out FINAL_TEST"
    assert "real-world performance" in result["not_to_be_described_as"]
    assert result["partition"] == "FINAL_TEST"


def test_final_forecasting_result_has_required_section_18_metadata_and_correct_label():
    path = fc_final.OUT_PATH
    if not path.exists():
        pytest.skip("final_forecasting.json not yet generated in this environment")
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    for field in ("dataset_id", "evidence_tier", "reserved_period", "selected_candidate",
                  "selected_strategy", "model_impl_version", "git_commit",
                  "evaluation_timestamp_utc", "final_metrics"):
        assert field in result, f"missing required Section 18 field: {field}"
    assert result["result_label"] == "Untouched temporal-test performance on reserved synthetic months"
    assert result["reserved_period"]["months"] == ["2024-10", "2024-11", "2024-12"]
    assert "Tier B" in result["not_to_be_described_as"]
    assert "real-world" in result["not_to_be_described_as"]
    assert "temporal validation" in result["not_to_be_described_as"]
    # rejected candidates must be explicitly recorded as not evaluated
    assert "random_forest__last_known_history" in result["rejected_candidates_not_evaluated_on_reserved_period"]
    assert "ridge__last_known_history" in result["rejected_candidates_not_evaluated_on_reserved_period"]
