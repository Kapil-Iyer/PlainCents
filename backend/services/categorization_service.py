"""
CategorizationService (TRD §7.3, §11.1-§11.4; Build Plan Phase 3, Option A;
ML-D Production Integration).

Loaded once (at FastAPI startup, via the lifespan hook), never per-request.

ML-C selected candidate (frozen in reports/ml/ML_C_SELECTION_RECORD.json):
TF-IDF + Logistic Regression (ml/categorization/candidates.py::TfidfLogRegCandidate).
The production artifact this service loads (config.LOGREG_MODEL_PATH,
models/tfidf_logreg_v1.pkl) is built by scripts/build_production_logreg_model.py,
which fits that exact recipe on the frozen Tier B TRAIN partition only
(data/evaluation/tier_b_split_v1.json — 133 rows / 47 merchant groups) —
never VALIDATION, never FINAL_TEST. This service itself never fits anything;
it only loads the already-fitted vectorizer + model and calls .transform()/
.predict() at inference time.

Input is merchant text only (config.CATEGORIES taxonomy) — the selected
recipe does NOT use amount/day-of-week/is_weekend (that was the retired
K-Means path's feature set; adding those features to LogReg here would
create a new, unevaluated model, which ML-D explicitly forbids). amount/date
are still accepted in predict()/predict_batch()'s input dicts for interface
compatibility with existing callers (TransactionService, IngestionService)
but are not used.

No confidence score is exposed, consistent with the prior K-Means behavior,
even though predict_proba is available on the underlying LogisticRegression.

The retired K-Means artifact/path (pipeline/cluster.py, models/kmeans_model.pkl)
is preserved untouched as ML-B evidence — this service simply no longer loads it.
"""
import logging
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from backend.api.errors import CategorizationUnavailableError

logger = logging.getLogger("backend")

Status = Literal["loaded", "missing", "error"]


class CategorizationService:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.status: Status = "missing"
        self._vectorizer = None
        self._model = None
        self.model_impl_version: str | None = None
        self.metadata: dict | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            self.status = "missing"
            logger.warning("Categorization model not found at %s", self.model_path)
            return
        try:
            payload = joblib.load(self.model_path)
            self._vectorizer = payload["vectorizer"]
            self._model = payload["model"]
            self.model_impl_version = payload.get("model_impl_version")
            self.metadata = payload.get("metadata")
            self.status = "loaded"
            logger.info(
                "Categorization model loaded from %s (model_impl_version=%s)",
                self.model_path, self.model_impl_version,
            )
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
        X = self._vectorizer.transform(df["merchant"].fillna("").astype(str))
        return list(self._model.predict(X))

    def predict(self, transaction: dict) -> dict:
        """transaction: {merchant, amount, date}. Returns {predicted_category}.
        amount/date are accepted for interface compatibility with existing
        callers but are not used by the selected merchant-text-only recipe."""
        self._require_loaded()
        df = pd.DataFrame([{"merchant": transaction["merchant"]}])
        categories = self._predict_frame(df)
        return {"predicted_category": categories[0]}

    def predict_batch(self, rows: list[dict]) -> list[dict]:
        """rows: [{merchant, amount, date}, ...]. Returns [{predicted_category}, ...]
        in the same order — used by Phase 4's bulk import, reusing this same
        service instance rather than a second categorization path."""
        self._require_loaded()
        if not rows:
            return []
        df = pd.DataFrame([{"merchant": r["merchant"]} for r in rows])
        categories = self._predict_frame(df)
        return [{"predicted_category": c} for c in categories]
