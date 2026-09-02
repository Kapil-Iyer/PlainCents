"""
ML-B Part B forecast data preparation.

DATA SOURCE (documented per ML Spec Section 10/23's data-provenance
requirement): `data/raw/synthetic_24mo.csv` (779 synthetic transactions,
2023-01 through 2024-12, 24 unique calendar months), categorized using the
EXISTING production K-Means artifact (`models/kmeans_model.pkl`) via
`pipeline.cluster.predict_categories` -- read-only use of the production
artifact (never retrained, never overwritten here). This mirrors the ML
Spec Section 2 interpretation of "the relevant current end-to-end synthetic
result" (K-Means-derived categories feeding the forecaster), so the
forecasting bake-off evaluates a production-realistic category input
stream rather than the diagnostic heuristic-label bypass V1's own
`__main__` block additionally prints.

EVIDENCE TIER: synthetic (ML Spec Section 3.1). This is legitimate for
testing forecaster/strategy *mechanism and behavior* (Section 17) but is
NEVER reported as real-user forecasting accuracy (Section 21).

ZERO-FILL CORRECTION (documented, does not modify pipeline/forecast.py):
V1's own `aggregate_monthly` (pipeline/forecast.py:22-50) performs a plain
`groupby(["month","category"]).sum()`, which silently OMITS a
(month, category) combination with zero underlying transactions rather than
recording total_spend=0. ML Spec Section 10 explicitly requires zero-spend
months to be treated as valid data points, not missing ones -- this matters
concretely here: the production K-Means labeling leaves "Other" absent in
18 of 24 months (verified by inspection), which is exactly the sparsity
condition Section 16 must be able to analyze correctly. This module
therefore builds the full (month x category) grid explicitly and fills
absent combinations with 0.0, rather than reusing aggregate_monthly's
groupby-only behavior. This is a correction inside ML-B's OWN evaluation
harness only; pipeline/forecast.py itself is untouched (Production
Isolation).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CATEGORIES
from pipeline.cluster import predict_categories
from pipeline.ingest import load_and_clean

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ID = "synthetic_24mo_kmeans_categorized_v1"
EVIDENCE_TIER = "Synthetic (ML Spec Section 3.1) -- pipeline-behavior/mechanism evaluation only, NEVER a real-world accuracy claim (Section 21)"


def build_monthly_grid() -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (month, category) for EVERY month in
    the observed range x EVERY category in config.CATEGORIES, total_spend
    zero-filled where no transactions existed. Sorted by (category, month)
    to match pipeline/forecast.py's existing row-order convention.
    """
    df = load_and_clean("synthetic_24mo.csv", bank="TD")
    df = predict_categories(df)  # read-only use of models/kmeans_model.pkl
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    raw_grid = df.groupby(["month", "category"], as_index=False)["amount"].sum()
    raw_grid = raw_grid.rename(columns={"amount": "total_spend"})

    all_months = sorted(df["month"].unique())
    full_index = pd.MultiIndex.from_product([all_months, CATEGORIES], names=["month", "category"])
    full_grid = raw_grid.set_index(["month", "category"]).reindex(full_index, fill_value=0.0).reset_index()
    full_grid = full_grid.sort_values(["category", "month"]).reset_index(drop=True)
    return full_grid


def monthly_grid_summary(grid: pd.DataFrame) -> dict:
    all_months = sorted(grid["month"].unique())
    n_zero_by_category = grid[grid["total_spend"] == 0].groupby("category").size().to_dict()
    n_months_by_category_nonzero = (
        grid[grid["total_spend"] > 0].groupby("category")["month"].nunique().to_dict()
    )
    return {
        "dataset_id": DATASET_ID,
        "evidence_tier": EVIDENCE_TIER,
        "calendar_range": {"start": all_months[0], "end": all_months[-1]},
        "n_unique_months": len(all_months),
        "categories": CATEGORIES,
        "n_zero_spend_month_category_rows_by_category": n_zero_by_category,
        "n_nonzero_months_by_category": n_months_by_category_nonzero,
        "total_month_category_cells": len(grid),
    }


if __name__ == "__main__":
    grid = build_monthly_grid()
    summary = monthly_grid_summary(grid)
    import json
    print(json.dumps(summary, indent=2, default=str))
