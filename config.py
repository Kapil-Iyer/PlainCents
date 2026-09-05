from pathlib import Path

# -- Paths ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_EXPORTS = BASE_DIR / "data" / "exports"
EXPORTS_DIR = DATA_EXPORTS
DB_PATH = BASE_DIR / "plaincents.db"
KMEANS_MODEL_PATH = BASE_DIR / "models" / "kmeans_model.pkl"
RF_MODEL_PATH = BASE_DIR / "models" / "rf_model.pkl"
# ML-F: production categorization artifact — word TF-IDF (max_features=200)
# + Logistic Regression (ml/categorization/candidates.py::TfidfLogRegCandidate),
# fit on the deployment-oriented TRAIN partition (ml/data/build_deployment_
# benchmark.py; frozen in reports/ml/ML_F_SELECTION_RECORD.json) and built by
# scripts/build_production_logreg_model.py. Supersedes ML-D's tfidf_logreg_v1
# (Tier-B-trained, max_features=50) — the ML-C selection record and Tier-B
# benchmark remain as historical/continuity evidence, not deleted.
# KMEANS_MODEL_PATH/RF_MODEL_PATH above are preserved (V1/ML-B evidence, no
# longer the selected production path) rather than removed.
LOGREG_MODEL_PATH = BASE_DIR / "models" / "tfidf_logreg_v2.pkl"
# ML-G: the production categorization artifact. Word TF-IDF (1-2 grams,
# unbounded vocabulary) UNION character TF-IDF (char_wb 2-6 grams) over
# v2-normalized merchant text, feeding Logistic Regression, plus the
# abstention policy fitted on VALIDATION -- all frozen in
# reports/ml/ML_G_SELECTION_RECORD.json and built by
# scripts/build_production_categorizer.py.
#
# Supersedes LOGREG_MODEL_PATH above (ML-F's word-only, 200-term recipe),
# which is kept as a path constant only so the older build script and its
# tests still resolve; nothing in the running application loads it any more.
CATEGORIZER_MODEL_PATH = BASE_DIR / "models" / "categorizer_v3.pkl"

# -- Category Labels (8) ----------------------------------
CATEGORIES = [
    "Food & Dining",
    "Transport",
    "Rent & Utilities",
    "Entertainment",
    "Healthcare",
    "Shopping",
    "Subscriptions",
    "Other",
]

# -- Bank Date Format Strings -----------------------------
BANK_DATE_FORMATS = {
    "TD":          "%m/%d/%Y",
    "RBC":         "%Y-%m-%d",
    "Scotiabank":  "%d %b %Y",
}

# -- Chart Colors (filled in Phase 7 pre-step) -----------
CHART_COLORS = {
    "Food & Dining":    "#E63946",
    "Transport":        "#457B9D",
    "Rent & Utilities": "#2A9D8F",
    "Entertainment":    "#E9C46A",
    "Healthcare":       "#F4A261",
    "Shopping":         "#A8DADC",
    "Subscriptions":    "#6A4C93",
    "Other":            "#A5A5A5",
    "accent_line":      "#1D3557",
    "accent_rolling":   "#E63946",
    "accent_bar":       "#457B9D",
    "accent_portfolio": "#2A9D8F",
    "accent_good":      "#2A9D8F",
    "accent_bad":       "#E63946",
}
