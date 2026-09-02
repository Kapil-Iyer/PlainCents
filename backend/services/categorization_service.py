"""
CategorizationService (TRD §7.3, §11.1-§11.4; Build Plan Phase 3, Option A).

Loaded once (at FastAPI startup, via the lifespan hook), never per-request.
predict()/predict_batch() reimplement pipeline.cluster.predict_categories()'s
three inner steps (build_feature_matrix -> kmeans.predict -> cluster->category
mapping) directly against the artifact cached in memory, rather than calling
predict_categories() itself — that function reloads the pickle from disk on
every call (pipeline/cluster.py:160-162), which TRD §7.3 explicitly forbids
on the request path.

No confidence score is fabricated (TRD §11.2): K-Means + majority-vote
mapping does not produce a defensible per-prediction confidence.
"""
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import joblib

from backend.api.errors import CategorizationUnavailableError
from pipeline.features import build_feature_matrix

logger = logging.getLogger("backend")

Status = Literal["loaded", "missing", "error"]


class CategorizationService:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.status: Status = "missing"
        self._kmeans = None
        self._scaler = None
        self._vectorizer = None
        self._cluster_to_category: dict | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            self.status = "missing"
            logger.warning("Categorization model not found at %s", self.model_path)
            return
        try:
            payload = joblib.load(self.model_path)
            self._kmeans = payload["kmeans"]
            self._scaler = payload["scaler"]
            self._vectorizer = payload["vectorizer"]
            self._cluster_to_category = payload["cluster_to_category"]
            self.status = "loaded"
            logger.info("Categorization model loaded from %s", self.model_path)
        except Exception:
            self.status = "error"
            logger.exception("Failed to load categorization model from %s", self.model_path)

    def _require_loaded(self) -> None:
        if self.status != "loaded":
            raise CategorizationUnavailableError(
                "The categorization model is unavailable.",
                details={"status": self.status},
            )

    def _predict_frame(self, df: pd.DataFrame) -> list[str]:
        X, _, _ = build_feature_matrix(
            df, scaler=self._scaler, vectorizer=self._vectorizer, fit=False
        )
        cluster_ids = self._kmeans.predict(X)
        return [self._cluster_to_category[cid] for cid in cluster_ids]

    def predict(self, transaction: dict) -> dict:
        """transaction: {merchant, amount, date}. Returns {predicted_category}."""
        self._require_loaded()
        df = pd.DataFrame(
            [
                {
                    "date": transaction["date"],
                    "merchant": transaction["merchant"],
                    "amount": transaction["amount"],
                }
            ]
        )
        categories = self._predict_frame(df)
        return {"predicted_category": categories[0]}

    def predict_batch(self, rows: list[dict]) -> list[dict]:
        """rows: [{merchant, amount, date}, ...]. Returns [{predicted_category}, ...]
        in the same order — used by Phase 4's bulk import, reusing this same
        service instance rather than a second categorization path."""
        self._require_loaded()
        if not rows:
            return []
        df = pd.DataFrame(
            [{"date": r["date"], "merchant": r["merchant"], "amount": r["amount"]} for r in rows]
        )
        categories = self._predict_frame(df)
        return [{"predicted_category": c} for c in categories]
