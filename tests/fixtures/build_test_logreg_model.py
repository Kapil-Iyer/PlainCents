"""
Deterministic test-fixture bootstrap for the production categorization
model (ML-D: TF-IDF + Logistic Regression, the ML-C selected recipe).

Builds a small LogReg artifact (same payload shape as production's
models/tfidf_logreg_v1.pkl: {vectorizer, model, model_impl_version,
metadata}) for use by backend unit/integration tests (CategorizationService),
WITHOUT touching the production artifact or the frozen Tier B benchmark/split
under data/evaluation/.

Deliberately self-contained, mirroring build_test_kmeans_model.py's own
style: a small, fixed, in-memory synthetic sample (not the frozen Tier B
benchmark, which is scientific evidence, not a test fixture), fit using the
real TfidfLogRegCandidate class (ml/categorization/candidates.py) so this
fixture tracks the actual selected recipe rather than a second, divergent
implementation. This is a deterministic test artifact, not a scientific
evaluation — no held-out split is needed here (that discipline lives in
ML-B/ML-C, not here).

Run: python tests/fixtures/build_test_logreg_model.py
"""
import random
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import CATEGORIES  # noqa: E402
from ml.categorization.candidates import TfidfLogRegCandidate  # noqa: E402

RANDOM_STATE = 42
ROWS_PER_CATEGORY = 15  # 8 categories x 15 = 120 rows, comfortably enough for a stable fixture
OUT_PATH = Path(__file__).resolve().parent / "logreg_model_test.pkl"
MODEL_IMPL_VERSION = "tfidf_logreg_v1"

# Same small, fixed merchant vocabulary as build_test_kmeans_model.py, so the
# two categorizer fixtures agree on what a "known" merchant/category pairing
# looks like.
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


def generate_test_transactions() -> pd.DataFrame:
    random.seed(RANDOM_STATE)
    rows = []
    for category in CATEGORIES:
        for _ in range(ROWS_PER_CATEGORY):
            merchant = random.choice(MERCHANTS[category])
            rows.append({"merchant": merchant, "true_category": category})
    return pd.DataFrame(rows)


def main() -> None:
    df = generate_test_transactions()

    candidate = TfidfLogRegCandidate(random_state=RANDOM_STATE, C=1.0, max_iter=1000)
    candidate.fit(df, label_col="true_category", text_col="merchant")

    metadata = {
        "model_impl_version": MODEL_IMPL_VERSION,
        "family": "TF-IDF + Logistic Regression",
        "candidate_name": "tfidf_logreg",
        "fit_partition": "TEST_FIXTURE (small in-memory synthetic sample, NOT the frozen Tier B TRAIN partition)",
        "fit_partition_n_rows": len(df),
        "recipe": candidate.describe(),
        "category_taxonomy": list(CATEGORIES),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": candidate._vectorizer,
            "model": candidate._model,
            "model_impl_version": MODEL_IMPL_VERSION,
            "metadata": metadata,
        },
        OUT_PATH,
    )
    print(f"Test LogReg artifact written: {OUT_PATH}")
    print(f"Rows: {len(df)}, categories: {df['true_category'].nunique()}")


if __name__ == "__main__":
    main()
