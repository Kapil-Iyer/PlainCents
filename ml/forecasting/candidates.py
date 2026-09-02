"""
ML Spec Section 11, candidates 3-4: Random Forest and Ridge.

RF uses the EXACT hyperparameters the TRD's shipped interactive
`pipeline.forecast.train_and_predict()` path uses (n_estimators=100,
max_depth=10, min_samples_leaf=5) -- NOT V1 walk_forward_validate's
diagnostic default (max_depth=3) and NOT a GridSearchCV-tuned
configuration, per Section 11's explicit instruction ("evaluating a
configuration the product doesn't actually use would make the acceptance
gate meaningless").

Ridge's hyperparameters are not frozen by the ML Spec. alpha=1.0 (sklearn's
own default) is used, fit on TRAIN-only-scaled features (StandardScaler
fit inside the same per-fold TRAIN-only boundary as Ridge itself) -- the
smallest defensible choice, not tuned against VALIDATION, per Section 11's
"choose the smallest defensible TRAIN/VALIDATION-safe approach" instruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

RF_HPARAMS = {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 5, "random_state": 42}
RIDGE_ALPHA = 1.0


@dataclass
class RandomForestCandidate:
    hparams: dict = field(default_factory=lambda: dict(RF_HPARAMS))

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestCandidate":
        self.model = RandomForestRegressor(**self.hparams)
        self.model.fit(X_train, y_train)
        return self

    def predict_row(self, row_df: pd.DataFrame) -> float:
        return float(self.model.predict(row_df)[0])

    def describe(self) -> dict:
        return {"name": "random_forest", **self.hparams, "source": "TRD-shipped train_and_predict() hyperparameters (ML Spec Section 11)"}


@dataclass
class RidgeCandidate:
    alpha: float = RIDGE_ALPHA

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RidgeCandidate":
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_scaled, y_train)
        return self

    def predict_row(self, row_df: pd.DataFrame) -> float:
        X_scaled = self.scaler.transform(row_df)
        return float(self.model.predict(X_scaled)[0])

    def describe(self) -> dict:
        return {"name": "ridge", "alpha": self.alpha, "scaler": "StandardScaler fit on TRAIN only per fold",
                "source": "sklearn default alpha, not tuned against VALIDATION (ML Spec Section 11)"}
