"""
ML-G categorization candidates: classical sparse-text models only.

Separate module from ml/categorization/candidates.py so ML-C's and ML-F's
frozen candidate definitions stay byte-for-byte untouched as historical
evidence. Everything here is classical scikit-learn: TF-IDF (word and/or
character n-grams) feeding a linear classifier or a Naive Bayes variant.
No neural networks, no embeddings, no transformers, no external lookups.

Every candidate exposes the same contract the ML-G bake-off runner needs:

    fit(train_df, label_col, text_col) -> self      # TRAIN rows only
    predict(df, text_col)              -> ndarray   # transform-only
    decision_scores(df, text_col)      -> ndarray   # row-normalized scores
    n_active_features(df, text_col)    -> ndarray   # per-row nnz
    describe()                         -> dict      # for the experiment log
    to_payload()                       -> dict      # serializable production artifact

`decision_scores` is what makes deployment-grade abstention possible: it
returns a per-row score vector over `classes_` that is comparable across
model families (probabilities where available, softmax-normalized decision
margins for LinearSVC). The ML-G decision policy uses two numbers derived
from it -- the top score and the top-minus-second margin -- plus
`n_active_features`, which is the single most diagnostic quantity in this
whole phase: a row with ZERO active features carries no evidence at all, and
any classifier will still return its most-likely-a-priori class for it. That
is exactly how the ML-F production artifact turned every unseen merchant into
"Food & Dining".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from ml.categorization.text_normalize_v2 import resolve_normalizer

RANDOM_STATE = 42

WORD_TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{2,}\b"


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _apply_normalize(series: pd.Series, normalize_fn: Callable[[str], str] | None) -> pd.Series:
    text = series.fillna("").astype(str)
    if normalize_fn is not None:
        text = text.map(normalize_fn)
    return text


def build_vectorizer(word_config: dict | None, char_config: dict | None):
    """Word TF-IDF, character TF-IDF, or a FeatureUnion of both.

    A plain sklearn FeatureUnion -- no custom framework. The word half
    carries merchant-identity terms and the category-typical head nouns the
    v2 corpus makes learnable; the char half is what survives Scotiabank-
    style mid-word truncation ("CAREWELL PHARM+4821") and the glued
    ONLINE PURCHASE ....COM shape, where whole-word tokens simply do not
    exist any more.
    """
    parts = []
    if word_config:
        cfg = dict(word_config)
        cfg.setdefault("analyzer", "word")
        cfg.setdefault("token_pattern", WORD_TOKEN_PATTERN)
        parts.append(("word", TfidfVectorizer(**cfg)))
    if char_config:
        cfg = dict(char_config)
        cfg.setdefault("analyzer", "char_wb")
        cfg.pop("token_pattern", None)
        parts.append(("char", TfidfVectorizer(**cfg)))
    if not parts:
        raise ValueError("at least one of word_config/char_config is required")
    if len(parts) == 1:
        return parts[0][1]
    return FeatureUnion(parts)


@dataclass
class SparseTextCandidate:
    """One classical sparse-text categorization candidate.

    Parameters
    ----------
    name : experiment identifier, recorded in the log and selection record.
    word_config / char_config : TfidfVectorizer kwargs. Passing both builds a
        FeatureUnion; passing one builds that vectorizer alone.
    normalizer_name : key into ml.categorization.text_normalize_v2.NORMALIZERS,
        or None for "vectorize the cleaned merchant text as-is". Stored BY
        NAME (not as a function reference) so the fitted artifact can record
        it and inference can resolve the identical function -- the fix for
        ML-F's train/serve skew.
    model_kind : "logreg" | "linear_svm" | "complement_nb" | "multinomial_nb".
    """

    name: str
    word_config: dict | None = None
    char_config: dict | None = None
    normalizer_name: str | None = None
    model_kind: str = "logreg"
    C: float = 1.0
    max_iter: int = 2000
    class_weight: str | None = None
    alpha: float = 1.0
    random_state: int = RANDOM_STATE
    _fitted_rows: int = field(default=0, init=False, repr=False)

    # -- construction --------------------------------------------------

    @property
    def normalize_fn(self) -> Callable[[str], str] | None:
        return resolve_normalizer(self.normalizer_name)

    def _build_model(self):
        if self.model_kind == "logreg":
            return LogisticRegression(
                C=self.C, max_iter=self.max_iter, class_weight=self.class_weight,
                random_state=self.random_state,
            )
        if self.model_kind == "linear_svm":
            return LinearSVC(
                C=self.C, max_iter=max(self.max_iter, 5000),
                class_weight=self.class_weight, random_state=self.random_state,
            )
        if self.model_kind == "complement_nb":
            return ComplementNB(alpha=self.alpha)
        if self.model_kind == "multinomial_nb":
            return MultinomialNB(alpha=self.alpha)
        raise ValueError(f"unknown model_kind {self.model_kind!r}")

    # -- fit / predict --------------------------------------------------

    def fit(self, train_df: pd.DataFrame, label_col: str = "true_category",
            text_col: str = "merchant") -> "SparseTextCandidate":
        self._vectorizer = build_vectorizer(self.word_config, self.char_config)
        text = _apply_normalize(train_df[text_col], self.normalize_fn)
        X = self._vectorizer.fit_transform(text)
        self._model = self._build_model()
        self._model.fit(X, train_df[label_col].values)
        self._fitted_rows = len(train_df)
        return self

    def _transform(self, df: pd.DataFrame, text_col: str = "merchant"):
        return self._vectorizer.transform(_apply_normalize(df[text_col], self.normalize_fn))

    def predict(self, df: pd.DataFrame, text_col: str = "merchant") -> np.ndarray:
        return self._model.predict(self._transform(df, text_col))

    def decision_scores(self, df: pd.DataFrame, text_col: str = "merchant") -> np.ndarray:
        X = self._transform(df, text_col)
        return scores_from_model(self._model, X)

    def n_active_features(self, df: pd.DataFrame, text_col: str = "merchant") -> np.ndarray:
        X = self._transform(df, text_col)
        return np.asarray((X != 0).sum(axis=1)).ravel()

    @property
    def classes_(self) -> np.ndarray:
        return self._model.classes_

    @property
    def vocabulary_size(self) -> int:
        v = self._vectorizer
        if isinstance(v, FeatureUnion):
            return sum(len(t.vocabulary_) for _, t in v.transformer_list)
        return len(v.vocabulary_)

    # -- reporting ------------------------------------------------------

    def describe(self) -> dict:
        return {
            "name": self.name,
            "word_config": self.word_config,
            "char_config": self.char_config,
            "normalizer_name": self.normalizer_name,
            "model_kind": self.model_kind,
            "C": self.C,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "alpha": self.alpha,
            "random_state": self.random_state,
            "vocabulary_size": self.vocabulary_size if hasattr(self, "_vectorizer") else None,
            "isolation": f"vectorizer/model fit on TRAIN only (n={self._fitted_rows})",
        }

    def config(self) -> dict:
        """The subset of describe() that is sufficient to rebuild an
        identical, unfitted candidate (see rebuild_candidate)."""
        return {
            "name": self.name,
            "word_config": self.word_config,
            "char_config": self.char_config,
            "normalizer_name": self.normalizer_name,
            "model_kind": self.model_kind,
            "C": self.C,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "alpha": self.alpha,
            "random_state": self.random_state,
        }


def scores_from_model(model, X) -> np.ndarray:
    """Per-row score vector over model.classes_, comparable across families.

    LogisticRegression / Naive Bayes expose predict_proba directly.
    LinearSVC exposes only signed distances to each hyperplane, so those are
    softmax-normalized -- NOT presented anywhere as calibrated probabilities,
    only used to rank classes and measure a top-vs-second margin, which is
    all the abstention policy needs.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    raw = model.decision_function(X)
    if raw.ndim == 1:  # binary edge case
        raw = np.column_stack([-raw, raw])
    return _softmax(raw)


def rebuild_candidate(cfg: dict) -> SparseTextCandidate:
    """Reconstruct a fresh, UNFITTED candidate from a config dict that has
    round-tripped through JSON. JSON turns ngram_range tuples into lists;
    sklearn requires actual tuples, so coerce them back."""
    def fix(c: dict | None) -> dict | None:
        if not c:
            return c
        out = dict(c)
        if isinstance(out.get("ngram_range"), list):
            out["ngram_range"] = tuple(out["ngram_range"])
        return out

    return SparseTextCandidate(
        name=cfg["name"],
        word_config=fix(cfg.get("word_config")),
        char_config=fix(cfg.get("char_config")),
        normalizer_name=cfg.get("normalizer_name"),
        model_kind=cfg.get("model_kind", "logreg"),
        C=cfg.get("C", 1.0),
        max_iter=cfg.get("max_iter", 2000),
        class_weight=cfg.get("class_weight"),
        alpha=cfg.get("alpha", 1.0),
        random_state=cfg.get("random_state", RANDOM_STATE),
    )
