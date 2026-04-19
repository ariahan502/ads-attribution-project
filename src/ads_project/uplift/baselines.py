from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ads_project.evaluation.metrics import binary_classification_metrics


class ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X) -> np.ndarray:
        rows = len(X)
        positive = np.full(rows, self.probability, dtype=float)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])


@dataclass(frozen=True)
class UpliftSpec:
    treatment_col: str
    outcome_col: str
    numeric_features: list[str]
    categorical_features: list[str]
    propensity_clip: float = 0.05
    max_iter: int = 100
    ridge_alpha: float = 1.0

    @property
    def all_features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


def _build_preprocess(spec: UpliftSpec) -> ColumnTransformer:
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
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, spec.numeric_features),
            ("cat", categorical_pipeline, spec.categorical_features),
        ]
    )


def _build_classifier(spec: UpliftSpec) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _build_preprocess(spec)),
            ("clf", LogisticRegression(max_iter=spec.max_iter)),
        ]
    )


def _build_regressor(spec: UpliftSpec) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _build_preprocess(spec)),
            ("reg", Ridge(alpha=spec.ridge_alpha)),
        ]
    )


def _fit_outcome_model(df: pd.DataFrame, *, spec: UpliftSpec) -> Pipeline | ConstantProbabilityModel:
    outcome_rate = float(df[spec.outcome_col].mean())
    if df[spec.outcome_col].nunique() < 2:
        return ConstantProbabilityModel(outcome_rate)

    outcome_model = _build_classifier(spec)
    outcome_model.fit(df[spec.all_features], df[spec.outcome_col])
    return outcome_model


def fit_doubly_robust_baseline(
    train_df: pd.DataFrame,
    *,
    spec: UpliftSpec,
) -> dict[str, Any]:
    if train_df[spec.treatment_col].nunique() < 2:
        raise ValueError("uplift training data must contain both treated and control rows")

    propensity_model = _build_classifier(spec)
    propensity_model.fit(train_df[spec.all_features], train_df[spec.treatment_col])

    treated_train = train_df[train_df[spec.treatment_col] == 1].copy()
    control_train = train_df[train_df[spec.treatment_col] == 0].copy()
    if treated_train.empty or control_train.empty:
        raise ValueError("uplift training split must have both treated and control subsets")

    treated_outcome_model = _fit_outcome_model(treated_train, spec=spec)
    control_outcome_model = _fit_outcome_model(control_train, spec=spec)

    e_hat = propensity_model.predict_proba(train_df[spec.all_features])[:, 1]
    e_hat = np.clip(e_hat, spec.propensity_clip, 1.0 - spec.propensity_clip)
    m1_hat = treated_outcome_model.predict_proba(train_df[spec.all_features])[:, 1]
    m0_hat = control_outcome_model.predict_proba(train_df[spec.all_features])[:, 1]

    treatment = train_df[spec.treatment_col].to_numpy(dtype=float)
    outcome = train_df[spec.outcome_col].to_numpy(dtype=float)
    m_t = np.where(treatment == 1.0, m1_hat, m0_hat)
    pseudo_outcome = ((treatment - e_hat) / (e_hat * (1.0 - e_hat))) * (outcome - m_t) + (m1_hat - m0_hat)

    tau_model = _build_regressor(spec)
    tau_model.fit(train_df[spec.all_features], pseudo_outcome)

    return {
        "propensity_model": propensity_model,
        "treated_outcome_model": treated_outcome_model,
        "control_outcome_model": control_outcome_model,
        "tau_model": tau_model,
    }


def predict_doubly_robust_scores(
    df: pd.DataFrame,
    *,
    models: dict[str, Any],
    spec: UpliftSpec,
) -> pd.DataFrame:
    propensity_score = models["propensity_model"].predict_proba(df[spec.all_features])[:, 1]
    propensity_score = np.clip(propensity_score, spec.propensity_clip, 1.0 - spec.propensity_clip)
    treated_outcome_score = models["treated_outcome_model"].predict_proba(df[spec.all_features])[:, 1]
    control_outcome_score = models["control_outcome_model"].predict_proba(df[spec.all_features])[:, 1]
    observational_score = treated_outcome_score - control_outcome_score
    doubly_robust_score = models["tau_model"].predict(df[spec.all_features])

    return pd.DataFrame(
        {
            "propensity_score": propensity_score,
            "treated_outcome_score": treated_outcome_score,
            "control_outcome_score": control_outcome_score,
            "observational_score": observational_score,
            "doubly_robust_score": doubly_robust_score,
        },
        index=df.index,
    )


def ranking_diagnostics(
    df: pd.DataFrame,
    *,
    score_col: str,
    treatment_col: str,
    outcome_col: str,
    top_fraction: float = 0.1,
) -> dict[str, float]:
    if not 0 < top_fraction < 0.5:
        raise ValueError("top_fraction must be between 0 and 0.5")

    ordered = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    k = max(1, int(len(ordered) * top_fraction))
    top = ordered.head(k)
    bottom = ordered.tail(k)

    return {
        "rows": float(len(ordered)),
        "top_rows": float(len(top)),
        "bottom_rows": float(len(bottom)),
        "top_mean_score": float(top[score_col].mean()),
        "bottom_mean_score": float(bottom[score_col].mean()),
        "top_observed_conversion_rate": float(top[outcome_col].mean()),
        "bottom_observed_conversion_rate": float(bottom[outcome_col].mean()),
        "top_treatment_rate": float(top[treatment_col].mean()),
        "bottom_treatment_rate": float(bottom[treatment_col].mean()),
    }


def policy_curve_diagnostics(
    df: pd.DataFrame,
    *,
    score_col: str,
    treatment_col: str,
    outcome_col: str,
    top_fractions: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.3),
) -> list[dict[str, float]]:
    ordered = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    baseline_outcome_rate = float(ordered[outcome_col].mean())
    baseline_treatment_rate = float(ordered[treatment_col].mean())

    curve: list[dict[str, float]] = []
    for top_fraction in top_fractions:
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fractions must be between 0 and 1")

        k = max(1, int(len(ordered) * top_fraction))
        selected = ordered.head(k)
        treated = selected[selected[treatment_col] == 1]
        control = selected[selected[treatment_col] == 0]
        treated_outcome_rate = _mean_or_none(treated[outcome_col])
        control_outcome_rate = _mean_or_none(control[outcome_col])

        curve.append(
            {
                "top_fraction": float(top_fraction),
                "rows": float(len(selected)),
                "mean_score": float(selected[score_col].mean()),
                "observed_outcome_rate": float(selected[outcome_col].mean()),
                "baseline_outcome_rate": baseline_outcome_rate,
                "outcome_rate_lift": float(selected[outcome_col].mean() / baseline_outcome_rate)
                if baseline_outcome_rate > 0
                else float("nan"),
                "treatment_rate": float(selected[treatment_col].mean()),
                "baseline_treatment_rate": baseline_treatment_rate,
                "treated_outcome_rate": treated_outcome_rate,
                "control_outcome_rate": control_outcome_rate,
                "observed_treated_control_gap": None
                if treated_outcome_rate is None or control_outcome_rate is None
                else treated_outcome_rate - control_outcome_rate,
            }
        )

    return curve


def _mean_or_none(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.mean())


def propensity_metrics(
    y_true,
    y_score,
) -> dict[str, float | None]:
    return binary_classification_metrics(y_true, y_score)
