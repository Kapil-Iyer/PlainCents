"""
V2 backend configuration (TRD §16).

Reads environment variables (via .env, loaded with python-dotenv) for values
that differ between environments. Shared constants that already live in the
root config.py (CATEGORIES, BANK_DATE_FORMATS, KMEANS_MODEL_PATH,
LOGREG_MODEL_PATH) are imported from there rather than duplicated (TRD §16,
§18.3) — this module must never redefine them.

CATEGORIZER_MODEL_PATH (ML-G): the production categorization artifact the
backend lifespan hook actually loads at startup (backend/main.py). It carries
not just the fitted vectorizer + classifier but the decision contract they
were selected with — which text normalizer to apply, and the abstention
threshold below which the system answers "Other" instead of guessing.

LOGREG_MODEL_PATH (ML-F) and KMEANS_MODEL_PATH (V1/ML-B) are still exported
for the historical build scripts and their tests, but nothing in the running
application loads either one.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from config import (
    BANK_DATE_FORMATS,
    CATEGORIES,
    CATEGORIZER_MODEL_PATH,
    KMEANS_MODEL_PATH,
    LOGREG_MODEL_PATH,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")

V2_DB_PATH = Path(os.environ.get("V2_DB_PATH", str(ROOT_DIR / "plaincents_v2.db")))
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

__all__ = [
    "V2_DB_PATH",
    "FRONTEND_ORIGIN",
    "LOG_LEVEL",
    "CATEGORIES",
    "BANK_DATE_FORMATS",
    "KMEANS_MODEL_PATH",
    "LOGREG_MODEL_PATH",
    "CATEGORIZER_MODEL_PATH",
]
