"""
Phase 2: K-Means clustering for expense categorization.
8 clusters, majority-vote label mapping on 160 rows, accuracy on 40 held-out.
Saves KMeans + StandardScaler + TfidfVectorizer to models/kmeans_model.pkl.
"""
import logging
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import CATEGORIES, KMEANS_MODEL_PATH
from pipeline.features import build_feature_matrix

logger = logging.getLogger(__name__)

# Keyword -> category for synthetic/real data labeling (substring match on merchant).
# Order matters: longer/more-specific keywords first (e.g. AMAZON PRIME before AMAZON RETAIL).
MERCHANT_KEYWORDS = {
    "TIM HORTONS":    "Food & Dining",
    "MCDONALDS":      "Food & Dining",
    "SUBWAY":         "Food & Dining",
    "LOBLAWS":        "Food & Dining",
    "METRO GROCER":   "Food & Dining",
    "GROCER":         "Food & Dining",
    "UBER":           "Transport",
    "PRESTO":         "Transport",
    "SHELL":          "Transport",
    "ESSO":           "Transport",
    "GO TRANSIT":     "Transport",
    "GASOLINE":       "Transport",
    "PETRO":          "Transport",
    "RIDESHARE":      "Transport",
    "ROGERS":         "Rent & Utilities",
    "BELL INTERNET":  "Rent & Utilities",
    "HYDRO ONE":      "Rent & Utilities",
    "ENBRIDGE":       "Rent & Utilities",
    "TORONTO HYDRO":  "Rent & Utilities",
    "ELECTRICITY":    "Rent & Utilities",
    "WIRELESS":       "Rent & Utilities",
    "NETFLIX":        "Entertainment",
    "SPOTIFY":        "Entertainment",
    "STEAM":          "Entertainment",
    "CINEPLEX":       "Entertainment",
    "AMAZON PRIME":   "Entertainment",
    "CINEMA":         "Entertainment",
    "STREAMING":      "Entertainment",
    "SHOPPERS":       "Healthcare",
    "REXALL":         "Healthcare",
    "MAPLE":          "Healthcare",
    "TELEHEALTH":     "Healthcare",
    "PHARMACY":       "Healthcare",
    "PRESCRIPTION":   "Healthcare",
    "AMAZON RETAIL":  "Shopping",
    "ZARA":           "Shopping",
    "HM":             "Shopping",
    "IKEA":           "Shopping",
    "BESTBUY":        "Shopping",
    "APPAREL":        "Shopping",
    "FURNITURE":      "Shopping",
    "ELECTRONICS":    "Shopping",
    "ADOBE":          "Subscriptions",
    "MICROSOFT":      "Subscriptions",
    "ICLOUD":         "Subscriptions",
    "YOUTUBE":        "Subscriptions",
    "SUITE":          "Subscriptions",
    "ATM":            "Other",
    "MISCELLANEOUS":  "Other",
    "BANK FEE":       "Other",
}


def _get_true_labels(df: pd.DataFrame) -> pd.Series:
    def label_merchant(merchant: str) -> str:
        m = str(merchant).upper()
        for keyword, category in MERCHANT_KEYWORDS.items():
            if keyword in m:
                return category
        return "Other"
    return df["merchant"].map(label_merchant)


def fit_and_evaluate(df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, float, float]:
    """
    Fit KMeans(8), build cluster->category mapping from 160 labeled rows, evaluate on 40 held-out.
    Shuffle labeled rows once before 160/40 split. Save model to config.KMEANS_MODEL_PATH.

    Parameters
    ----------
    df : DataFrame
        Clean transactions from ingest (columns: date, merchant, amount).
    random_state : int
        Seed for shuffling the 200 labeled rows (and KMeans).

    Returns
    -------
    df_labeled : DataFrame
        Original df with columns cluster_id and category added.
    accuracy_heldout : float
        Mapped label accuracy on the 40-transaction held-out set (target 80%+).
    silhouette : float
        Silhouette score on full X (diagnostic only).
    """
    if df.empty or len(df) < 200:
        raise ValueError("Need at least 200 rows for 160/40 split")

    X, scaler, vectorizer = build_feature_matrix(df, fit=True)
    kmeans = KMeans(n_clusters=12, random_state=random_state, n_init=50)
    cluster_ids = kmeans.fit_predict(X)
    df = df.copy()
    df["cluster_id"] = cluster_ids

    true_labels = _get_true_labels(df)
    df["_true_label"] = true_labels
    labeled = df.assign(true_label=true_labels).reset_index(drop=True)
    indices = labeled.index.tolist()
    random.seed(random_state)
    random.shuffle(indices)
    chosen = indices[:200]
    mapping_indices = chosen[:160]
    eval_indices = chosen[160:200]

    cluster_to_category = {}
    for cid in range(kmeans.n_clusters):
        subset = labeled.loc[mapping_indices]
        subset = subset[subset["cluster_id"] == cid]
        if subset.empty:
            cluster_to_category[cid] = CATEGORIES[0]
            continue
        counts = subset["true_label"].value_counts()
        cluster_to_category[cid] = counts.index[0]

    pred_eval = labeled.loc[eval_indices, "cluster_id"].map(cluster_to_category)
    true_eval = labeled.loc[eval_indices, "true_label"]
    accuracy_heldout = (pred_eval.values == true_eval.values).mean()
    silhouette = float(silhouette_score(X, cluster_ids))

    df_labeled = df.drop(columns=["_true_label"], errors="ignore")
    df_labeled["category"] = df_labeled["cluster_id"].map(cluster_to_category)

    KMEANS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"kmeans": kmeans, "scaler": scaler, "vectorizer": vectorizer, "cluster_to_category": cluster_to_category},
        KMEANS_MODEL_PATH,
    )
    logger.info("Saved KMeans + scaler + vectorizer to %s", KMEANS_MODEL_PATH)

    return df_labeled, float(accuracy_heldout), silhouette


def predict_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load saved model, transform features, assign cluster_id and category.
    DataFrame must have date, merchant, amount.
    """
    if not KMEANS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {KMEANS_MODEL_PATH}. Run fit_and_evaluate first.")
    payload = joblib.load(KMEANS_MODEL_PATH)
    kmeans = payload["kmeans"]
    scaler = payload["scaler"]
    vectorizer = payload["vectorizer"]
    cluster_to_category = payload["cluster_to_category"]

    X, _, _ = build_feature_matrix(df, scaler=scaler, vectorizer=vectorizer, fit=False)
    cluster_ids = kmeans.predict(X)
    df = df.copy()
    df["cluster_id"] = cluster_ids
    df["category"] = df["cluster_id"].map(cluster_to_category)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pipeline.ingest import load_and_clean

    df = load_and_clean("synthetic_12mo.csv", bank="TD")
    df_labeled, accuracy, silhouette = fit_and_evaluate(df)
    cluster_to_category = joblib.load(KMEANS_MODEL_PATH)["cluster_to_category"]
    print("Cluster to category mapping:")
    print(cluster_to_category)
    print(f"Held-out accuracy (40 transactions): {accuracy:.1%}")
    print(f"Silhouette score (diagnostic):       {silhouette:.4f}")
    print(f"Categories assigned: {df_labeled['category'].nunique()}")
    if accuracy < 0.80:
        print("(Target 80%+. If below, try n_clusters=6 or 10, or expand TF-IDF max_features.)")
