from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


@dataclass(frozen=True)
class BaselineSpec:
    label: str
    numeric_features: list[str]
    categorical_features: list[str]
    model_type: str = "logistic_regression"
    model_params: dict[str, Any] | None = None

    @property
    def all_features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]

    @property
    def resolved_model_params(self) -> dict[str, Any]:
        return dict(self.model_params or {})


def build_logistic_regression_pipeline(spec: BaselineSpec) -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, spec.numeric_features),
            ("cat", categorical_pipeline, spec.categorical_features),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("clf", LogisticRegression(**spec.resolved_model_params)),
        ]
    )


def build_xgboost_pipeline(spec: BaselineSpec) -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required for model_type='xgboost'. Install it with `pip install xgboost`."
        ) from exc

    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, spec.numeric_features),
            ("cat", categorical_pipeline, spec.categorical_features),
        ]
    )

    xgboost_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 1,
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    xgboost_params.update(spec.resolved_model_params)

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("clf", XGBClassifier(**xgboost_params)),
        ]
    )


def build_model_pipeline(spec: BaselineSpec) -> Pipeline:
    if spec.model_type == "logistic_regression":
        return build_logistic_regression_pipeline(spec)
    if spec.model_type == "xgboost":
        return build_xgboost_pipeline(spec)
    raise ValueError(f"Unsupported model_type: {spec.model_type}")


def fit_model(
    train_df: pd.DataFrame,
    *,
    spec: BaselineSpec,
) -> Pipeline:
    model = build_model_pipeline(spec)
    model.fit(train_df[spec.all_features], train_df[spec.label])
    return model


def predict_scores(
    model: Pipeline,
    df: pd.DataFrame,
    *,
    spec: BaselineSpec,
) -> list[float]:
    return model.predict_proba(df[spec.all_features])[:, 1]
