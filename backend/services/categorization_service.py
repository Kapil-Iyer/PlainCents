"""
CategorizationService -- loads the frozen production categorization artifact
and serves the SELECTED decision, not just the model's argmax.

Loaded once at FastAPI startup (via the lifespan hook), never per request.
This service never fits anything: it loads an already-fitted vectorizer and
classifier and calls .transform()/.predict() at inference time.

WHAT ML-G CHANGED HERE, AND WHY
-------------------------------
The previous implementation did two things that made a well-selected model
serve badly:

  1. It vectorized the raw `merchant` column with no text normalization,
     regardless of what the winning recipe had been FIT with. A recipe that
     normalized its training text could therefore never have been served
     correctly -- classic train/serve skew. The artifact now records its
     normalizer by name and this service resolves that exact function.

  2. It always returned the classifier's argmax. On a row the vectorizer
     produced no features for, sklearn's LogisticRegression returns
     argmax(intercept_) -- one fixed class for every evidence-free input.
     On the shipped ML-F artifact that class was "Food & Dining", which is
     exactly the "everything becomes Food & Dining" production symptom. The
     artifact now records the abstention policy fitted on VALIDATION, and
     this service applies it.

The decision this service returns is therefore the same decision the ML-G
bake-off measured on held-out data, by construction.

Input is merchant text only. amount/date are accepted in the input dicts for
interface compatibility with existing callers but are not used by the
selected recipe -- adding them would create a new, unevaluated model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from backend.api.errors import CategorizationUnavailableError

logger = logging.getLogger("backend")

Status = Literal["loaded", "missing", "error"]

DEFAULT_ABSTAIN_CATEGORY = "Other"


class CategorizationService:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.status: Status = "missing"
        self._vectorizer = None
        self._model = None
        self._normalize_fn = None
        self.normalizer_name: str | None = None
        self.min_margin: float = 0.0
        self.abstain_category: str = DEFAULT_ABSTAIN_CATEGORY
        self.model_impl_version: str | None = None
        self.metadata: dict | None = None
        self._load()

    # -- loading ---------------------------------------------------------

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

            # Resolve the decision contract the artifact was built with.
            # resolve_normalizer raises on an unknown name rather than
            # silently serving un-normalized text -- a mismatch here is
            # precisely the train/serve skew this is meant to prevent, so
            # failing to load is the correct, loud outcome.
            from ml.categorization.text_normalize_v2 import resolve_normalizer

            self.normalizer_name = payload.get("normalizer_name")
            self._normalize_fn = resolve_normalizer(self.normalizer_name)
            self.min_margin = float(payload.get("min_margin", 0.0))
            self.abstain_category = payload.get("abstain_category", DEFAULT_ABSTAIN_CATEGORY)

            self.status = "loaded"
            logger.info(
                "Categorization model loaded from %s (model_impl_version=%s, "
                "normalizer=%s, min_margin=%s)",
                self.model_path, self.model_impl_version,
                self.normalizer_name, self.min_margin,
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

    # -- inference -------------------------------------------------------

    def _prepare(self, merchants: list[str]):
        text = pd.Series(merchants).fillna("").astype(str)
        if self._normalize_fn is not None:
            text = text.map(self._normalize_fn)
        return self._vectorizer.transform(text)

    def _scores(self, X) -> np.ndarray:
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        raw = self._model.decision_function(X)
        if raw.ndim == 1:
            raw = np.column_stack([-raw, raw])
        shifted = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def classify_batch(self, merchants: list[str]) -> list[dict]:
        """The single inference entry point.

        Returns, per input row:
          category          the served decision (may be the abstain category)
          model_category    what the classifier alone said, always preserved
          n_active_features how much evidence the text produced
          margin            top-vs-runner-up probability gap
          abstained         whether the policy overrode the classifier
          abstain_reason    "no_features" | "low_margin" | None

        Keeping `model_category` alongside `category` is what makes the
        decision auditable: the How It Works walkthrough and the diagnostics
        script both need to show what the model said AND what the system
        served, and they must not have to re-run inference to find out.
        """
        self._require_loaded()
        if not merchants:
            return []

        X = self._prepare(merchants)
        model_preds = self._model.predict(X)
        scores = self._scores(X)
        n_active = np.asarray((X != 0).sum(axis=1)).ravel()

        if scores.shape[1] >= 2:
            part = np.partition(scores, -2, axis=1)
            margins = part[:, -1] - part[:, -2]
        else:
            margins = scores.max(axis=1)

        out = []
        for i, model_category in enumerate(model_preds):
            reason = None
            if n_active[i] == 0:
                reason = "no_features"
            elif margins[i] < self.min_margin:
                reason = "low_margin"
            out.append({
                "category": self.abstain_category if reason else str(model_category),
                "model_category": str(model_category),
                "n_active_features": int(n_active[i]),
                "margin": float(margins[i]),
                "abstained": reason is not None,
                "abstain_reason": reason,
            })
        return out

    def classify(self, merchant: str) -> dict:
        return self.classify_batch([merchant])[0]

    # -- backward-compatible surface -------------------------------------
    # TransactionService and the older tests call predict()/predict_batch()
    # with {merchant, amount, date} dicts. Those keep working and now return
    # the SERVED decision (policy applied), not the bare classifier argmax.

    def predict(self, transaction: dict) -> dict:
        return {"predicted_category": self.classify(transaction["merchant"])["category"]}

    def predict_batch(self, rows: list[dict]) -> list[dict]:
        results = self.classify_batch([r["merchant"] for r in rows])
        return [{"predicted_category": r["category"]} for r in results]
