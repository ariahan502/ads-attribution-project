from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class BaselineSpec:
    label: str
    numeric_features: list[str]
    categorical_features: list[str]
    max_iter: int = 100

    @property
    def all_features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


def build_logistic_regression_pipeline(spec: BaselineSpec) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", spec.numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), spec.categorical_features),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=spec.max_iter)),
        ]
    )


def fit_baseline_model(
    train_df: pd.DataFrame,
    *,
    spec: BaselineSpec,
) -> Pipeline:
    model = build_logistic_regression_pipeline(spec)
    model.fit(train_df[spec.all_features], train_df[spec.label])
    return model


def predict_scores(
    model: Pipeline,
    df: pd.DataFrame,
    *,
    spec: BaselineSpec,
) -> list[float]:
    return model.predict_proba(df[spec.all_features])[:, 1]
