"""
ML Spec Section 5 frozen categorization bake-off: exactly three candidates.

  1. K-Means (current production baseline) - re-evaluated here under STRICT
     TRAIN-only fitting/mapping isolation (Section 6.1), which V1's own
     evaluation did not do (V1's 160/40 split drew both from the same
     synthetic/heuristic pool, Section 2).
  2. TF-IDF + Logistic Regression
  3. TF-IDF + Linear SVM

PRODUCTION ISOLATION: this module imports pipeline.features.build_feature_matrix
(read-only reuse of already-verified feature code, Section 5's own reasoning
for candidate 2/3 -- "reuses features.py's TF-IDF vectorization conceptually")
but never imports or calls pipeline.cluster.fit_and_evaluate / predict_categories,
and never writes to models/kmeans_model.pkl. This file trains its own,
separate, evaluation-only K-Means instance every run; it never touches the
production artifact.

Every candidate here exposes the same three-step contract so the bake-off
runner can treat them uniformly:
  fit(train_df)          -- fit using TRAIN rows only
  predict(df)             -- transform + predict, never refits
  describe()               -- hyperparameters/config for the experiment log
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from pipeline.features import build_feature_matrix

RANDOM_STATE = 42

# Same TF-IDF configuration V1's features.py already uses (Section 5's own
# framing: "reuses features.py's TF-IDF vectorization conceptually"). Reused
# verbatim here rather than re-tuned, per the instruction to avoid broad
# hyperparameter searches without evidence (ML Spec Section A4).
TFIDF_CONFIG = dict(
    max_features=50,
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    ngram_range=(1, 2),
    sublinear_tf=True,
)


# ---------------------------------------------------------------------------
# Candidate 1: K-Means, TRAIN-only isolation (Section 6.1)
# ---------------------------------------------------------------------------

@dataclass
class KMeansCandidate:
    n_clusters: int = 12  # matches pipeline/cluster.py:112 exactly -- not re-tuned
    random_state: int = RANDOM_STATE

    def fit(self, train_df: pd.DataFrame, label_col: str = "true_category") -> "KMeansCandidate":
        """
        TRAIN-only fitting, per Section 6.1:
          - StandardScaler + TfidfVectorizer fit on TRAIN only
          - KMeans fit on TRAIN only
          - cluster -> category mapping built from TRAIN labels only (majority
            vote per cluster, over the FULL TRAIN set -- not an internal
            160/40 re-split of TRAIN; VALIDATION already plays that held-out
            role at the outer level, per Section 6.1's explicit instruction
            not to build an ad hoc internal split).
        """
        X, scaler, vectorizer = build_feature_matrix(train_df, fit=True)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=50)
        cluster_ids = kmeans.fit_predict(X)

        labels = train_df[label_col].reset_index(drop=True)
        cluster_to_category: dict[int, str] = {}
        fallback_category = sorted(labels.unique())[0]  # deterministic fallback, documented (mirrors V1's arbitrary-but-fixed fallback)
        for cid in range(self.n_clusters):
            mask = cluster_ids == cid
            if not mask.any():
                cluster_to_category[cid] = fallback_category
                continue
            counts = labels[mask].value_counts()
            cluster_to_category[cid] = counts.index[0]

        self._scaler = scaler
        self._vectorizer = vectorizer
        self._kmeans = kmeans
        self._cluster_to_category = cluster_to_category
        self._fitted_on_n_rows = len(train_df)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Transform-and-predict only. Never calls .fit on scaler/vectorizer/kmeans."""
        X, _, _ = build_feature_matrix(df, scaler=self._scaler, vectorizer=self._vectorizer, fit=False)
        cluster_ids = self._kmeans.predict(X)
        return np.array([self._cluster_to_category[c] for c in cluster_ids])

    def describe(self) -> dict:
        return {
            "name": "kmeans",
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "feature_config": "pipeline.features.build_feature_matrix (amount x0.2 StandardScaler, "
                               "TF-IDF merchant top-50 L2-normalized, day_of_week x0.1, is_weekend x0.1)",
            "cluster_to_category_mapping": {int(k): v for k, v in self._cluster_to_category.items()},
            "isolation": "scaler/vectorizer/kmeans/mapping fit on TRAIN only (n=%d)" % self._fitted_on_n_rows,
        }


# ---------------------------------------------------------------------------
# Candidates 2 & 3: TF-IDF + supervised linear classifiers
# ---------------------------------------------------------------------------

def _apply_normalize(series: pd.Series, normalize_fn: Callable[[str], str] | None) -> pd.Series:
    text = series.fillna("").astype(str)
    if normalize_fn is not None:
        text = text.map(normalize_fn)
    return text


@dataclass
class TfidfLogRegCandidate:
    """ML-C's original word-TF-IDF + LogReg recipe, generalized (ML-F,
    additive/backward-compatible) to also serve as:
      - Candidate B (larger/unbounded word vocabulary) via `tfidf_overrides`
      - Candidate C (deterministic boilerplate normalization) via
        `normalize_fn` (ml/categorization/text_normalize.py)
      - Candidate D (character n-grams) via `tfidf_overrides={"analyzer":
        "char_wb", "ngram_range": (3, 5), ...}`
    Every default below is unchanged from the frozen ML-C recipe, so
    `TfidfLogRegCandidate()` with no arguments is byte-for-byte the same
    candidate `build_production_logreg_model.py` and `run_bakeoff.py`
    already use -- nothing about the ML-C-selected baseline changes."""
    random_state: int = RANDOM_STATE
    C: float = 1.0
    max_iter: int = 1000
    tfidf_overrides: dict | None = None
    normalize_fn: Callable[[str], str] | None = None
    candidate_name: str = "tfidf_logreg"

    def _tfidf_config(self) -> dict:
        cfg = dict(TFIDF_CONFIG)
        if self.tfidf_overrides:
            cfg.update(self.tfidf_overrides)
            # token_pattern only applies to analyzer="word" (sklearn's
            # default); dropping it avoids a spurious UserWarning when a
            # candidate overrides analyzer to "char"/"char_wb".
            if cfg.get("analyzer", "word") != "word":
                cfg.pop("token_pattern", None)
        return cfg

    def fit(self, train_df: pd.DataFrame, label_col: str = "true_category", text_col: str = "merchant") -> "TfidfLogRegCandidate":
        self._vectorizer = TfidfVectorizer(**self._tfidf_config())
        X = self._vectorizer.fit_transform(_apply_normalize(train_df[text_col], self.normalize_fn))
        y = train_df[label_col].values
        self._model = LogisticRegression(
            C=self.C, max_iter=self.max_iter, random_state=self.random_state
        )
        self._model.fit(X, y)
        self._fitted_on_n_rows = len(train_df)
        return self

    def predict(self, df: pd.DataFrame, text_col: str = "merchant") -> np.ndarray:
        X = self._vectorizer.transform(_apply_normalize(df[text_col], self.normalize_fn))
        return self._model.predict(X)

    def predict_proba(self, df: pd.DataFrame, text_col: str = "merchant") -> np.ndarray:
        X = self._vectorizer.transform(_apply_normalize(df[text_col], self.normalize_fn))
        return self._model.predict_proba(X)

    def describe(self) -> dict:
        return {
            "name": self.candidate_name,
            "tfidf_config": {k: v for k, v in self._tfidf_config().items()},
            "normalize_fn": self.normalize_fn.__name__ if self.normalize_fn else None,
            "C": self.C,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "isolation": "vectorizer/model fit on TRAIN only (n=%d)" % self._fitted_on_n_rows,
        }


@dataclass
class TfidfWordCharLogRegCandidate:
    """ML-F Candidate E: word TF-IDF + character TF-IDF combined via a plain
    sklearn FeatureUnion (ML-F brief §8: 'a proper sklearn FeatureUnion or
    equivalent clean implementation', 'do not create a complicated custom ML
    framework'). Only meant to be run if Candidate C or D individually beats
    A/B on VALIDATION (ML-F brief §8/§11)."""
    random_state: int = RANDOM_STATE
    C: float = 1.0
    max_iter: int = 1000
    word_overrides: dict | None = None
    char_overrides: dict | None = None
    normalize_fn: Callable[[str], str] | None = None

    def _word_config(self) -> dict:
        cfg = dict(TFIDF_CONFIG)
        if self.word_overrides:
            cfg.update(self.word_overrides)
        return cfg

    def _char_config(self) -> dict:
        cfg = {"analyzer": "char_wb", "ngram_range": (3, 5), "max_features": 300, "sublinear_tf": True}
        if self.char_overrides:
            cfg.update(self.char_overrides)
        return cfg

    def fit(self, train_df: pd.DataFrame, label_col: str = "true_category", text_col: str = "merchant") -> "TfidfWordCharLogRegCandidate":
        word_cfg, char_cfg = self._word_config(), self._char_config()
        self._union = FeatureUnion([
            ("word", TfidfVectorizer(**word_cfg)),
            ("char", TfidfVectorizer(**char_cfg)),
        ])
        text = _apply_normalize(train_df[text_col], self.normalize_fn)
        X = self._union.fit_transform(text)
        y = train_df[label_col].values
        self._model = LogisticRegression(
            C=self.C, max_iter=self.max_iter, random_state=self.random_state
        )
        self._model.fit(X, y)
        self._fitted_on_n_rows = len(train_df)
        self._word_cfg, self._char_cfg = word_cfg, char_cfg
        return self

    def _transform(self, df: pd.DataFrame, text_col: str = "merchant"):
        return self._union.transform(_apply_normalize(df[text_col], self.normalize_fn))

    def predict(self, df: pd.DataFrame, text_col: str = "merchant") -> np.ndarray:
        return self._model.predict(self._transform(df, text_col))

    def predict_proba(self, df: pd.DataFrame, text_col: str = "merchant") -> np.ndarray:
        return self._model.predict_proba(self._transform(df, text_col))

    def describe(self) -> dict:
        return {
            "name": "tfidf_word_char_logreg",
            "word_tfidf_config": self._word_cfg,
            "char_tfidf_config": self._char_cfg,
            "normalize_fn": self.normalize_fn.__name__ if self.normalize_fn else None,
            "C": self.C,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "isolation": "vectorizer/model fit on TRAIN only (n=%d)" % self._fitted_on_n_rows,
        }


@dataclass
class TfidfLinearSVMCandidate:
    """Generalized (ML-F, additive/backward-compatible) the same way as
    TfidfLogRegCandidate above, so ML-F Candidate F (SVM confirmation pass on
    whichever representation wins) can reuse the winning `tfidf_overrides`/
    `normalize_fn` without a new class. Defaults unchanged from ML-C."""
    random_state: int = RANDOM_STATE
    C: float = 1.0
    max_iter: int = 5000
    tfidf_overrides: dict | None = None
    normalize_fn: Callable[[str], str] | None = None

    def _tfidf_config(self) -> dict:
        cfg = dict(TFIDF_CONFIG)
        if self.tfidf_overrides:
            cfg.update(self.tfidf_overrides)
            if cfg.get("analyzer", "word") != "word":
                cfg.pop("token_pattern", None)
        return cfg

    def fit(self, train_df: pd.DataFrame, label_col: str = "true_category", text_col: str = "merchant") -> "TfidfLinearSVMCandidate":
        self._vectorizer = TfidfVectorizer(**self._tfidf_config())
        X = self._vectorizer.fit_transform(_apply_normalize(train_df[text_col], self.normalize_fn))
        y = train_df[label_col].values
        self._model = LinearSVC(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
        self._model.fit(X, y)
        self._fitted_on_n_rows = len(train_df)
        return self

    def predict(self, df: pd.DataFrame, text_col: str = "merchant") -> np.ndarray:
        X = self._vectorizer.transform(_apply_normalize(df[text_col], self.normalize_fn))
        return self._model.predict(X)

    def describe(self) -> dict:
        return {
            "name": "tfidf_linear_svm",
            "tfidf_config": self._tfidf_config(),
            "normalize_fn": self.normalize_fn.__name__ if self.normalize_fn else None,
            "C": self.C,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "isolation": "vectorizer/model fit on TRAIN only (n=%d)" % self._fitted_on_n_rows,
            "note": "LinearSVC does not produce calibrated probabilities; none fabricated (ML Spec Section 5/7).",
        }
