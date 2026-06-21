from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SklearnMultiLabelBaseline:
    """Small, reproducible one-vs-rest baseline for tabular or late-fusion experiments."""

    random_state: int = 42
    max_iter: int = 1000
    label_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    _model: OneVsRestClassifier | None = None

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> "SklearnMultiLabelBaseline":
        self.feature_columns = list(features.columns)
        self.label_columns = list(labels.columns)
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=self.max_iter,
                solver="liblinear",
                random_state=self.random_state,
            ),
        )
        self._model = OneVsRestClassifier(estimator)
        self._model.fit(features[self.feature_columns], labels[self.label_columns])
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("model must be fit before predict_proba")
        probabilities = self._model.predict_proba(features[self.feature_columns])
        return pd.DataFrame(probabilities, columns=self.label_columns, index=features.index)
