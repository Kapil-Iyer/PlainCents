"""
ML Spec Section 6: merchant-grouped, category-stratified TRAIN/VALIDATION/
FINAL-TEST splitting for the categorization bake-off.

Core guarantee this module exists to provide and that tests/ml verify:
all transactions sharing the same `merchant_group` land in exactly one of
TRAIN / VALIDATION / FINAL_TEST. Category balance is preserved as closely as
feasible across partitions by splitting merchant GROUPS within each category
separately (Section 6's "hybrid grouped + stratified" protocol), rather than
splitting rows directly (which would violate grouping) or splitting groups
without regard to category (which could concentrate a whole category into
one partition by chance).

This module deliberately does not consult FINAL_TEST at all beyond
constructing the partition assignment: `verify_split_isolation` reports only
structural information (counts, merchant-group set intersections), never
category-level performance, matching the frozen requirement that ML-B may
"construct/reserve the final partition ... but it must remain sealed."
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TRAIN = "TRAIN"
VALIDATION = "VALIDATION"
FINAL_TEST = "FINAL_TEST"


@dataclass
class SplitResult:
    assignment: dict[str, str]          # merchant_group -> partition
    seed: int
    train_frac: float
    val_frac: float
    test_frac: float
    group_col: str
    category_col: str
    per_category_group_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["partition"] = out[self.group_col].map(self.assignment)
        if out["partition"].isna().any():
            missing = out.loc[out["partition"].isna(), self.group_col].unique().tolist()
            raise ValueError(f"merchant groups with no partition assignment: {missing}")
        return out

    def to_json_dict(self) -> dict:
        return {
            "seed": self.seed,
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "test_frac": self.test_frac,
            "group_col": self.group_col,
            "category_col": self.category_col,
            "per_category_group_counts": self.per_category_group_counts,
            "assignment": self.assignment,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> "SplitResult":
        with open(path) as f:
            data = json.load(f)
        return cls(
            assignment=data["assignment"],
            seed=data["seed"],
            train_frac=data["train_frac"],
            val_frac=data["val_frac"],
            test_frac=data["test_frac"],
            group_col=data["group_col"],
            category_col=data["category_col"],
            per_category_group_counts=data.get("per_category_group_counts", {}),
        )


def merchant_grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str,
    category_col: str,
    seed: int = 42,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
) -> SplitResult:
    """
    Build a merchant-grouped, category-stratified partition assignment.

    Each merchant group is assumed to belong to exactly one category (true
    for the Tier B benchmark, where a merchant group is a single real-world
    business). Groups are shuffled deterministically (seeded) within each
    category, then split into TRAIN/VALIDATION/FINAL_TEST by count.

    Rounding rule (deterministic, documented): for a category with N groups,
    n_val = max(1, round(N * val_frac)) if N >= 3 else (0 if N < 2 else 1),
    n_test = max(1, round(N * test_frac)) if N >= 3 else (1 if N >= 2 and
    n_val == 0 else 0), remainder to TRAIN. This guarantees every category
    with >=3 groups contributes at least one group to VALIDATION and one to
    FINAL_TEST (so no category is silently invisible in either partition),
    while a category with only 1-2 groups degrades gracefully rather than
    raising.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    per_category_group_counts: dict[str, dict[str, int]] = {}

    group_to_category = df.drop_duplicates(subset=[group_col]).set_index(group_col)[category_col].to_dict()
    categories = sorted(set(group_to_category.values()))

    for category in categories:
        groups = sorted([g for g, c in group_to_category.items() if c == category])
        rng.shuffle(groups)
        n = len(groups)

        if n >= 3:
            n_val = max(1, round(n * val_frac))
            n_test = max(1, round(n * test_frac))
            n_val = min(n_val, n - 2)  # leave >=1 for train, >=1 for test
            n_test = min(n_test, n - n_val - 1)
            n_train = n - n_val - n_test
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:  # n == 1
            n_train, n_val, n_test = 1, 0, 0

        train_groups = groups[:n_train]
        val_groups = groups[n_train:n_train + n_val]
        test_groups = groups[n_train + n_val:n_train + n_val + n_test]

        for g in train_groups:
            assignment[g] = TRAIN
        for g in val_groups:
            assignment[g] = VALIDATION
        for g in test_groups:
            assignment[g] = FINAL_TEST

        per_category_group_counts[category] = {
            "total_groups": n, "train": len(train_groups),
            "validation": len(val_groups), "final_test": len(test_groups),
        }

    return SplitResult(
        assignment=assignment, seed=seed,
        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
        group_col=group_col, category_col=category_col,
        per_category_group_counts=per_category_group_counts,
    )


def verify_split_isolation(df_with_partition: pd.DataFrame, group_col: str, partition_col: str = "partition") -> dict:
    """
    Structural-only isolation report (ML Spec Section 6/6.1's leakage guard).

    Returns counts and merchant-group set intersections between every pair
    of partitions. Never touches category labels or model performance —
    this function is safe to call even while FINAL_TEST must remain sealed,
    because it reports only which merchant-group IDENTIFIERS fall in which
    partition, not what those transactions are or how any model does on
    them.
    """
    groups_by_partition = {
        p: set(df_with_partition.loc[df_with_partition[partition_col] == p, group_col])
        for p in [TRAIN, VALIDATION, FINAL_TEST]
    }
    row_counts = {
        p: int((df_with_partition[partition_col] == p).sum())
        for p in [TRAIN, VALIDATION, FINAL_TEST]
    }
    intersections = {
        "TRAIN_and_VALIDATION": sorted(groups_by_partition[TRAIN] & groups_by_partition[VALIDATION]),
        "TRAIN_and_FINAL_TEST": sorted(groups_by_partition[TRAIN] & groups_by_partition[FINAL_TEST]),
        "VALIDATION_and_FINAL_TEST": sorted(groups_by_partition[VALIDATION] & groups_by_partition[FINAL_TEST]),
    }
    all_empty = all(len(v) == 0 for v in intersections.values())
    return {
        "row_counts": row_counts,
        "unique_merchant_groups": {p: len(groups_by_partition[p]) for p in [TRAIN, VALIDATION, FINAL_TEST]},
        "intersections": intersections,
        "all_intersections_empty": all_empty,
    }
