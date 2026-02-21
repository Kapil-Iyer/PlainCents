from pathlib import Path

# ── Paths ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_EXPORTS = BASE_DIR / "data" / "exports"
DB_PATH = BASE_DIR / "plaincents.db"
KMEANS_MODEL_PATH = BASE_DIR / "models" / "kmeans_model.pkl"
RF_MODEL_PATH = BASE_DIR / "models" / "rf_model.pkl"

# ── Category Labels (8) ──────────────────────────────────
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

# ── Bank Date Format Strings ─────────────────────────────
BANK_DATE_FORMATS = {
    "TD":          "%m/%d/%Y",
    "RBC":         "%Y-%m-%d",
    "Scotiabank":  "%d %b %Y",
}

# ── Chart Colors (filled in Phase 7 pre-step) ───────────
CHART_COLORS = {
    "Food & Dining":    "#PLACEHOLDER",
    "Transport":        "#PLACEHOLDER",
    "Rent & Utilities": "#PLACEHOLDER",
    "Entertainment":    "#PLACEHOLDER",
    "Healthcare":       "#PLACEHOLDER",
    "Shopping":         "#PLACEHOLDER",
    "Subscriptions":    "#PLACEHOLDER",
    "Other":            "#PLACEHOLDER",
    "accent":           "#PLACEHOLDER",
}