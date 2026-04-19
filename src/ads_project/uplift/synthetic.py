from __future__ import annotations

import numpy as np
import pandas as pd


def add_semisynthetic_uplift_columns(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    true_effect_col: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    synthetic = df.copy()

    cost_rank = _rank_01(synthetic["log_cost"] if "log_cost" in synthetic else synthetic["cost"])
    cpo_rank = _rank_01(synthetic["log_cpo"] if "log_cpo" in synthetic else synthetic["cpo"])
    recency_rank = _rank_01(
        synthetic["log_time_since_last_click"]
        if "log_time_since_last_click" in synthetic
        else synthetic["time_since_last_click"].clip(lower=0)
    )
    has_prev_click = synthetic.get("has_prev_click", synthetic["time_since_last_click"].ge(0).astype(int))

    true_effect = 0.02 + 0.16 * cpo_rank + 0.06 * has_prev_click - 0.04 * recency_rank
    true_effect = np.clip(true_effect.to_numpy(dtype=float), 0.01, 0.25)

    propensity = 0.15 + 0.55 * cost_rank + 0.15 * has_prev_click
    propensity = np.clip(propensity.to_numpy(dtype=float), 0.05, 0.95)
    treatment = rng.binomial(1, propensity)

    baseline_probability = 0.06 + 0.20 * cost_rank + 0.10 * (1.0 - recency_rank)
    baseline_probability = np.clip(baseline_probability.to_numpy(dtype=float), 0.02, 0.70)
    outcome_probability = np.clip(baseline_probability + treatment * true_effect, 0.0, 0.95)
    outcome = rng.binomial(1, outcome_probability)

    synthetic[treatment_col] = treatment
    synthetic[outcome_col] = outcome
    synthetic[true_effect_col] = true_effect
    synthetic[f"{treatment_col}_propensity"] = propensity
    synthetic[f"{outcome_col}_probability"] = outcome_probability
    synthetic["baseline_outcome_probability"] = baseline_probability
    return synthetic


def known_effect_ranking_report(
    df: pd.DataFrame,
    *,
    score_cols: list[str],
    true_effect_col: str,
    top_fractions: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.3),
) -> dict[str, dict[str, float | list[dict[str, float]]]]:
    report: dict[str, dict[str, float | list[dict[str, float]]]] = {}
    baseline_true_effect = float(df[true_effect_col].mean())

    for score_col in score_cols:
        ordered = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        curve = []
        for top_fraction in top_fractions:
            if not 0 < top_fraction <= 1:
                raise ValueError("top_fractions must be between 0 and 1")
            k = max(1, int(len(ordered) * top_fraction))
            selected = ordered.head(k)
            selected_true_effect = float(selected[true_effect_col].mean())
            curve.append(
                {
                    "top_fraction": float(top_fraction),
                    "rows": float(len(selected)),
                    "mean_score": float(selected[score_col].mean()),
                    "mean_true_effect": selected_true_effect,
                    "baseline_true_effect": baseline_true_effect,
                    "true_effect_lift": selected_true_effect / baseline_true_effect
                    if baseline_true_effect > 0
                    else float("nan"),
                }
            )

        report[score_col] = {
            "spearman_corr_with_true_effect": float(
                df[[score_col, true_effect_col]].corr(method="spearman").iloc[0, 1]
            ),
            "pearson_corr_with_true_effect": float(df[[score_col, true_effect_col]].corr().iloc[0, 1]),
            "policy_curve": curve,
        }

    oracle = df.sort_values(true_effect_col, ascending=False).reset_index(drop=True)
    oracle_curve = []
    for top_fraction in top_fractions:
        k = max(1, int(len(oracle) * top_fraction))
        selected = oracle.head(k)
        selected_true_effect = float(selected[true_effect_col].mean())
        oracle_curve.append(
            {
                "top_fraction": float(top_fraction),
                "rows": float(len(selected)),
                "mean_true_effect": selected_true_effect,
                "baseline_true_effect": baseline_true_effect,
                "true_effect_lift": selected_true_effect / baseline_true_effect
                if baseline_true_effect > 0
                else float("nan"),
            }
        )
    report["oracle_true_effect"] = {
        "spearman_corr_with_true_effect": 1.0,
        "pearson_corr_with_true_effect": 1.0,
        "policy_curve": oracle_curve,
    }
    return report


def _rank_01(series: pd.Series) -> pd.Series:
    ranked = series.rank(method="average", pct=True)
    return ranked.fillna(0.5).clip(0.0, 1.0)
