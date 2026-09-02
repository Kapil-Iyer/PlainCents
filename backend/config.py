"""
V2 backend configuration (TRD §16).

Reads environment variables (via .env, loaded with python-dotenv) for values
that differ between environments. Shared constants that already live in the
root config.py (CATEGORIES, BANK_DATE_FORMATS, KMEANS_MODEL_PATH) are
imported from there rather than duplicated (TRD §16, §18.3) — this module
must never redefine them.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from config import BANK_DATE_FORMATS, CATEGORIES, KMEANS_MODEL_PATH

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
]
