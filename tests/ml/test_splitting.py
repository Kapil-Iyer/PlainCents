"""
ML Spec Section 6 leakage-safety tests: merchant groups must never cross
categorization partition boundaries.
"""
import pandas as pd
import pytest

from ml.common.splitting import (
    FINAL_TEST,
    TRAIN,
    VALIDATION,
    merchant_grouped_stratified_split,
    verify_split_isolation,
)


def _toy_df():
    rows = []
    for cat_idx, category in enumerate(["A", "B", "C"]):
        for g in range(6):  # 6 merchant groups per category
            group_name = f"{category}_group_{g}"
            for r in range(3):  # 3 transactions per group
                rows.append({"merchant_group": group_name, "category": category, "row_id": f"{group_name}_{r}"})
    return pd.DataFrame(rows)


def test_merchant_groups_never_split_across_partitions():
    df = _toy_df()
    split = merchant_grouped_stratified_split(df, "merchant_group", "category", seed=42)
    df_p = split.apply(df)

    report = verify_split_isolation(df_p, "merchant_group")
    assert report["all_intersections_empty"], report["intersections"]
    assert report["intersections"]["TRAIN_and_VALIDATION"] == []
    assert report["intersections"]["TRAIN_and_FINAL_TEST"] == []
    assert report["intersections"]["VALIDATION_and_FINAL_TEST"] == []


def test_every_row_of_a_group_gets_the_same_partition():
    df = _toy_df()
    split = merchant_grouped_stratified_split(df, "merchant_group", "category", seed=42)
    df_p = split.apply(df)

    partitions_per_group = df_p.groupby("merchant_group")["partition"].nunique()
    assert (partitions_per_group == 1).all()


def test_every_category_present_in_train_and_validation():
    df = _toy_df()
    split = merchant_grouped_stratified_split(df, "merchant_group", "category", seed=42)
    df_p = split.apply(df)

    for category in ["A", "B", "C"]:
        cat_rows = df_p[df_p["category"] == category]
        present_partitions = set(cat_rows["partition"])
        assert TRAIN in present_partitions
        assert VALIDATION in present_partitions, f"category {category} missing from VALIDATION"


def test_deterministic_given_same_seed():
    df = _toy_df()
    split_1 = merchant_grouped_stratified_split(df, "merchant_group", "category", seed=7)
    split_2 = merchant_grouped_stratified_split(df, "merchant_group", "category", seed=7)
    assert split_1.assignment == split_2.assignment


def test_different_seed_can_change_assignment():
    df = _toy_df()
    split_1 = merchant_grouped_stratified_split(df, "merchant_group", "category", seed=1)
    split_2 = merchant_grouped_stratified_split(df, "merchant_group", "category", seed=2)
    assert split_1.assignment != split_2.assignment


def test_save_and_load_round_trip(tmp_path):
    df = _toy_df()
    split = merchant_grouped_stratified_split(df, "merchant_group", "category", seed=42)
    path = tmp_path / "split.json"
    split.save(path)
    from ml.common.splitting import SplitResult
    loaded = SplitResult.load(path)
    assert loaded.assignment == split.assignment
    assert loaded.seed == split.seed


def test_raises_on_missing_group_assignment():
    df = _toy_df()
    split = merchant_grouped_stratified_split(df, "merchant_group", "category", seed=42)
    extra_row = pd.DataFrame([{"merchant_group": "UNSEEN_GROUP", "category": "A", "row_id": "x"}])
    combined = pd.concat([df, extra_row], ignore_index=True)
    with pytest.raises(ValueError):
        split.apply(combined)


def test_tier_b_benchmark_actually_splits_cleanly():
    """Integration check against the real Tier B benchmark file (not a toy)."""
    from pathlib import Path
    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "tier_b_benchmark.csv"
    if not csv_path.exists():
        pytest.skip("tier_b_benchmark.csv not built yet")
    df = pd.read_csv(csv_path, keep_default_na=False)
    split = merchant_grouped_stratified_split(df, "merchant_group", "true_category", seed=42)
    df_p = split.apply(df)
    report = verify_split_isolation(df_p, "merchant_group")
    assert report["all_intersections_empty"]
    # every category must appear in TRAIN and VALIDATION at minimum
    for category in df["true_category"].unique():
        cat_partitions = set(df_p.loc[df_p["true_category"] == category, "partition"])
        assert TRAIN in cat_partitions
        assert VALIDATION in cat_partitions
