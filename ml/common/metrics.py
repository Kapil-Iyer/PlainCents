"""
ML Spec Section 7 (categorization) / Section 13 (forecasting) metrics.

Forecast metrics are computed on plain numpy arrays so the exact same
functions serve every candidate/strategy/horizon combination — no
per-candidate metric-calculation drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ZERO_ACTUAL_MAPE_EPSILON = 1e-9  # matches pipeline/forecast.py's own guard (forecast.py:221), documented, not silently different


# ---------------------------------------------------------------------------
# Section 13: forecast metrics
# ---------------------------------------------------------------------------

def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """WAPE = sum(|actual - predicted|) / sum(|actual|). Undefined (NaN) only
    if sum(|actual|) == 0 for the whole evaluated set (all-zero actuals)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denom)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape_safe(actual: np.ndarray, predicted: np.ndarray, epsilon: float = ZERO_ACTUAL_MAPE_EPSILON) -> dict:
    """
    MAPE with EXPLICIT zero-actual handling (ML Spec Section 13's documented
    MAPE failure mode). Never silently averages in a division-by-near-zero
    blowup without flagging it.

    Returns
    -------
    dict with:
      mape_all : MAPE over every row, using max(|actual|, epsilon) as the
                 denominator per row (matches pipeline/forecast.py:221's
                 existing guard, so this is comparable to V1's number).
      mape_nonzero_only : MAPE computed using only rows where |actual| > 1.00
                 (an explicit, disclosed threshold — one dollar of real
                 spend — chosen for interpretability, not tuned to flatter
                 any candidate). NaN if no such rows exist.
      n_rows : total rows evaluated.
      n_near_zero_actual : rows with |actual| <= 1.00, i.e. excluded from
                 mape_nonzero_only and where mape_all is most likely to be
                 distorted (Section 13's "near-zero actual" problem).
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom_all = np.maximum(np.abs(actual), epsilon)
    ape_all = np.abs(actual - predicted) / denom_all * 100

    near_zero_mask = np.abs(actual) <= 1.00
    nonzero_mask = ~near_zero_mask

    return {
        "mape_all": float(np.mean(ape_all)) if len(ape_all) else float("nan"),
        "mape_nonzero_only": float(np.mean(ape_all[nonzero_mask])) if nonzero_mask.any() else float("nan"),
        "n_rows": int(len(actual)),
        "n_near_zero_actual": int(near_zero_mask.sum()),
    }


def forecast_metric_bundle(actual: np.ndarray, predicted: np.ndarray) -> dict:
    m = mape_safe(actual, predicted)
    return {
        "wape": wape(actual, predicted),
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        **m,
        "n": int(len(np.asarray(actual))),
    }


# ---------------------------------------------------------------------------
# Section 7: categorization metrics
# ---------------------------------------------------------------------------

def categorization_metric_bundle(y_true, y_pred, labels: list[str]) -> dict:
    """
    Returns accuracy, macro F1, per-category precision/recall/F1, and a
    confusion matrix (as a nested dict keyed by true/pred label so it is
    JSON-serializable and order-independent of sklearn's array indexing).
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_category = {
        labels[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_dict = {
        true_label: {labels[j]: int(cm[i, j]) for j in range(len(labels))}
        for i, true_label in enumerate(labels)
    }
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_category": per_category,
        "confusion_matrix": cm_dict,
        "n": int(len(y_true)),
    }
