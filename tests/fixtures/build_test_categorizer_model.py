"""
Deterministic test-fixture bootstrap for the ML-G production categorization
artifact.

Builds a small artifact with the SAME payload shape production uses
(models/categorizer_v3.pkl: vectorizer, model, model_impl_version,
normalizer_name, min_margin, abstain_category, categories, metadata) so
CategorizationService loads it through exactly the production code path --
including resolving the recorded text normalizer and applying the recorded
abstention threshold. A fixture that skipped those fields would leave the two
most important ML-G behaviours untested.

Fit with the real SparseTextCandidate class
(ml/categorization/candidates_v2.py) rather than a second, divergent
implementation, so this fixture tracks the actual selected recipe family.

Deliberately self-contained: a small, fixed, in-memory synthetic sample -- NOT
the frozen deployment benchmark under data/evaluation/, which is scientific
evidence, not a test fixture. This is a deterministic test artifact, not an
evaluation; held-out-split discipline lives in the ML phases, not here.

Supersedes build_test_logreg_model.py, which is kept for the older
ML-D/ML-F-era tests that still assert the previous payload shape.

Run: python tests/fixtures/build_test_categorizer_model.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import CATEGORIES  # noqa: E402
from ml.categorization.candidates_v2 import SparseTextCandidate  # noqa: E402

RANDOM_STATE = 42
OUT_PATH = Path(__file__).resolve().parent / "categorizer_model_test.pkl"
MODEL_IMPL_VERSION = "tfidf_word_char_logreg_v3_test"

# A low but non-zero margin so tests can exercise BOTH branches of the
# decision policy: a well-separated merchant is served by the model, and a
# genuinely ambiguous one abstains. Not the production value (which is fitted
# on VALIDATION); a fixture threshold is a test knob, not evidence.
TEST_MIN_MARGIN = 0.02
NORMALIZER_NAME = "normalize_deployment_text_v2"

# Several distinct merchants per category sharing category-typical head
# nouns -- the same design property the v2 corpus is built on, in miniature,
# so the fixture model can actually generalize to an unseen merchant in a
# test instead of memorizing three strings.
MERCHANTS = {
    "Food & Dining": [
        "TIM HORTONS COFFEE", "MAPLE DINER", "NORTHSIDE PIZZA",
        "SUNRISE CAFE", "LOBLAWS GROCERY", "CEDAR GROCERS",
    ],
    "Transport": [
        "UBER RIDESHARE TRIP", "SHELL FUEL STATION", "PRESTO TRANSIT FARE",
        "METRO TRANSIT PASS", "QUICKPARK PARKING GARAGE", "NORTHSTAR TAXI",
    ],
    "Rent & Utilities": [
        "ROGERS WIRELESS MOBILE", "HYDRO ONE POWER", "ENBRIDGE GAS UTILITY",
        "BRIGHTWAVE INTERNET", "CITYLINE WATER UTILITY", "SUMMIT PROPERTY RENT",
    ],
    "Entertainment": [
        "CINEPLEX CINEMA TICKET", "SILVERSCREEN CINEMAS", "ARCADE ZONE",
        "TICKETVAULT EVENTS", "STARLIGHT BOWL LANES", "QUARRY MUSEUM",
    ],
    "Healthcare": [
        "SHOPPERS PHARMACY", "REXALL PHARMACY", "CAREWELL PHARMACY",
        "BRIGHT SMILE DENTAL", "VITAL PHYSIO CLINIC", "CLEARVIEW OPTICAL",
    ],
    "Shopping": [
        "AMAZON RETAIL APPAREL", "TRENDLINE APPAREL", "HOMEBASE HARDWARE",
        "PAGEBOUND BOOKS", "VALUEMART DEPT STORE", "ACTIVEGEAR SPORTS",
    ],
    "Subscriptions": [
        "ADOBE CREATIVE SUBSCRIPTION", "ACME SUB SERVICE", "CLOUDDESK WORKSPACE PLAN",
        "STREAMBOX PLUS MONTHLY", "FITZONE GYM MEMBERSHIP", "AUDIOWAVE PODCAST PLAN",
    ],
    "Other": [
        "BANK FEE PENALTY", "MONTHLY ACCOUNT SERVICE FEE", "NSF RETURNED ITEM FEE",
        "OVERDRAFT INTEREST CHARGE", "WIRE TRANSFER SERVICE FEE", "CHEQUE ORDER FEE",
    ],
}

# Deployment-shaped decorations, so the fixture model sees boilerplate the
# way the real one does.
TEMPLATES = [
    "{name}",
    "VISA DEBIT PURCHASE - 4821 {name}",
    "pos purchase Opos {name}",
    "{name} #0042",
    "{name} PREAUTH PYMT 774120",
]


def generate_test_transactions() -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        for merchant in MERCHANTS[category]:
            for template in TEMPLATES:
                rows.append({
                    "merchant": template.format(name=merchant).upper(),
                    "true_category": category,
                })
    return pd.DataFrame(rows)


def main() -> None:
    df = generate_test_transactions()

    candidate = SparseTextCandidate(
        name="test_fixture_word_char_union",
        word_config={"max_features": None, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1},
        char_config={"analyzer": "char_wb", "ngram_range": (2, 6), "max_features": 8000,
                     "sublinear_tf": True},
        normalizer_name=NORMALIZER_NAME,
        random_state=RANDOM_STATE,
    ).fit(df, label_col="true_category", text_col="merchant")

    metadata = {
        "model_impl_version": MODEL_IMPL_VERSION,
        "family": "Word TF-IDF + character TF-IDF (FeatureUnion) + Logistic Regression",
        "fit_partition": "TEST_FIXTURE (small in-memory synthetic sample, NOT a frozen benchmark)",
        "fit_partition_n_rows": len(df),
        "recipe": candidate.describe(),
        "category_taxonomy": list(CATEGORIES),
        "decision_policy": {
            "normalizer_name": NORMALIZER_NAME,
            "min_margin": TEST_MIN_MARGIN,
            "zero_feature_rule": "always abstain (unconditional)",
            "abstain_category": "Other",
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": candidate._vectorizer,
            "model": candidate._model,
            "model_impl_version": MODEL_IMPL_VERSION,
            "normalizer_name": NORMALIZER_NAME,
            "min_margin": TEST_MIN_MARGIN,
            "abstain_category": "Other",
            "categories": list(CATEGORIES),
            "metadata": metadata,
        },
        OUT_PATH,
    )
    print(f"Test categorizer artifact written: {OUT_PATH}")
    print(f"Rows: {len(df)}, vocabulary: {candidate.vocabulary_size}")


if __name__ == "__main__":
    main()
