"""
Deterministic test-fixture bootstrap for the categorization model.

Builds a small K-Means artifact (same payload shape as production's
models/kmeans_model.pkl: {kmeans, scaler, vectorizer, cluster_to_category})
for use by backend unit/integration tests (CategorizationService), WITHOUT
touching the production artifact.

Deliberately self-contained: does NOT depend on data/raw/synthetic_24mo.csv,
which is gitignored (data/raw/) and therefore not guaranteed to exist on a
fresh clone. Generates its own small, fixed, in-memory synthetic sample
instead, reusing the same building blocks V1 uses (pipeline.features,
pipeline.cluster's heuristic labels) rather than a second, divergent
implementation.

Run: python tests/fixtures/build_test_kmeans_model.py
"""
import random
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import CATEGORIES  # noqa: E402
from pipeline.cluster import _get_true_labels  # noqa: E402
from pipeline.features import build_feature_matrix  # noqa: E402

RANDOM_STATE = 42
N_CLUSTERS = 12
ROWS_PER_CATEGORY = 30  # 8 categories x 30 = 240 rows, comfortably above KMeans's n_clusters=12
OUT_PATH = Path(__file__).resolve().parent / "kmeans_model_test.pkl"

# Small, fixed merchant vocabulary per category — deliberately similar in
# spirit to scripts/generate_synthetic_24mo.py's vocabulary (keyword-rich, so
# pipeline.cluster's MERCHANT_KEYWORDS heuristic labels correctly), but
# defined locally so this fixture has no file dependency on data/raw/.
MERCHANTS = {
    "Food & Dining": ["TIM HORTONS COFFEE", "MCDONALDS BURGER", "LOBLAWS GROCER"],
    "Transport": ["UBER RIDESHARE TRIP", "SHELL GASOLINE FUEL", "PRESTO TRANSIT RELOAD"],
    "Rent & Utilities": ["ROGERS WIRELESS PHONE", "HYDRO ONE ELECTRICITY", "ENBRIDGE GAS HEAT"],
    "Entertainment": ["NETFLIX STREAMING", "SPOTIFY MUSIC STREAMING", "CINEPLEX CINEMA TICKET"],
    "Healthcare": ["SHOPPERS PHARMACY PRESCRIPTION", "REXALL PHARMACY", "MAPLE TELEHEALTH VIRTUAL"],
    "Shopping": ["AMAZON RETAIL APPAREL", "ZARA APPAREL FASHION", "BESTBUY ELECTRONICS RETAIL"],
    "Subscriptions": ["ADOBE CREATIVE SUITE", "MICROSOFT OFFICE SUITE", "ICLOUD BACKUP SUITE"],
    "Other": ["ATM CASH WITHDRAWAL", "BANK FEE PENALTY", "MISCELLANEOUS EXPENSE"],
}

AMOUNT_RANGES = {
    "Food & Dining": (8, 60), "Transport": (5, 50), "Rent & Utilities": (60, 150),
    "Entertainment": (10, 40), "Healthcare": (15, 70), "Shopping": (20, 150),
    "Subscriptions": (5, 20), "Other": (10, 80),
}


def generate_test_transactions() -> pd.DataFrame:
    random.seed(RANDOM_STATE)
    rows = []
    day_counter = 1
    for category in CATEGORIES:
        lo, hi = AMOUNT_RANGES[category]
        for _ in range(ROWS_PER_CATEGORY):
            merchant = random.choice(MERCHANTS[category])
            amount = round(random.uniform(lo, hi), 2)
            month = 1 + (day_counter // 28) % 12
            day = 1 + (day_counter % 28)
            rows.append({
                "date": f"2026-{month:02d}-{day:02d}",
                "merchant": merchant,
                "amount": amount,
            })
            day_counter += 1
    return pd.DataFrame(rows)


def main() -> None:
    df = generate_test_transactions()

    X, scaler, vectorizer = build_feature_matrix(df, fit=True)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster_ids = kmeans.fit_predict(X)

    df = df.copy()
    df["cluster_id"] = cluster_ids
    df["true_label"] = _get_true_labels(df)

    # Simple majority-vote mapping over the full small fixture set. No
    # held-out split needed here — this is a deterministic test artifact,
    # not a scientific evaluation (that discipline lives in ML-B, not here).
    cluster_to_category = {}
    for cid in range(N_CLUSTERS):
        subset = df[df["cluster_id"] == cid]
        if subset.empty:
            cluster_to_category[cid] = CATEGORIES[0]
            continue
        cluster_to_category[cid] = subset["true_label"].value_counts().index[0]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "kmeans": kmeans,
            "scaler": scaler,
            "vectorizer": vectorizer,
            "cluster_to_category": cluster_to_category,
        },
        OUT_PATH,
    )
    print(f"Test K-Means artifact written: {OUT_PATH}")
    print(f"Rows: {len(df)}, clusters: {N_CLUSTERS}, categories mapped: {len(set(cluster_to_category.values()))}")


if __name__ == "__main__":
    main()
