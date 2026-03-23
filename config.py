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
