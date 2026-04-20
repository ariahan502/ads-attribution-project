from __future__ import annotations

from typing import Any

import math

import pandas as pd


def policy_simulation_report(
    df: pd.DataFrame,
    *,
    score_cols: list[str],
    top_fractions: list[float],
    outcome_col: str,
    treatment_col: str,
    true_effect_col: str | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    _validate_inputs(df, score_cols=score_cols, top_fractions=top_fractions)

    baseline = _summarize_selection(
        df,
        policy_name="all_rows",
        score_col=None,
        selected_df=df,
        full_df=df,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        true_effect_col=true_effect_col,
        top_fraction=1.0,
    )
    policies = [baseline]

    for top_fraction in top_fractions:
        rows_to_select = max(1, int(len(df) * top_fraction))
        random_expected = df.sample(n=rows_to_select, random_state=random_seed)
        policies.append(
            _summarize_selection(
                df,
                policy_name="random_policy",
                score_col=None,
                selected_df=random_expected,
                full_df=df,
                outcome_col=outcome_col,
                treatment_col=treatment_col,
                true_effect_col=true_effect_col,
                top_fraction=top_fraction,
            )
        )

        if true_effect_col is not None:
            oracle = df.sort_values(true_effect_col, ascending=False).head(rows_to_select)
            policies.append(
                _summarize_selection(
                    df,
                    policy_name="oracle_true_effect",
                    score_col=true_effect_col,
                    selected_df=oracle,
                    full_df=df,
                    outcome_col=outcome_col,
                    treatment_col=treatment_col,
                    true_effect_col=true_effect_col,
                    top_fraction=top_fraction,
                )
            )

        for score_col in score_cols:
            selected = df.sort_values(score_col, ascending=False).head(rows_to_select)
            policies.append(
                _summarize_selection(
                    df,
                    policy_name=f"top_{score_col}",
                    score_col=score_col,
                    selected_df=selected,
                    full_df=df,
                    outcome_col=outcome_col,
                    treatment_col=treatment_col,
                    true_effect_col=true_effect_col,
                    top_fraction=top_fraction,
                )
            )

    policy_frame = pd.DataFrame(policies)
    if true_effect_col is not None:
        policy_frame = _add_oracle_regret(policy_frame)

    return {
        "rows": int(len(df)),
        "score_cols": score_cols,
        "top_fractions": top_fractions,
        "outcome_col": outcome_col,
        "treatment_col": treatment_col,
        "true_effect_col": true_effect_col,
        "random_seed": random_seed,
        "baseline": baseline,
        "policies": _clean_records(policy_frame),
    }


def policy_decision_report(
    policy_report: dict[str, Any],
    *,
    preferred_score_col: str,
    recommended_top_fraction: float | None = None,
) -> dict[str, Any]:
    policy_frame = pd.DataFrame(policy_report["policies"])
    preferred_policy_name = f"top_{preferred_score_col}"
    preferred_rows = policy_frame[policy_frame["policy_name"] == preferred_policy_name].copy()
    if preferred_rows.empty:
        raise ValueError(f"preferred policy is missing from simulation report: {preferred_policy_name}")

    comparison_rows = []
    for _, preferred_row in preferred_rows.sort_values("top_fraction").iterrows():
        top_fraction = float(preferred_row["top_fraction"])
        random_row = _matching_policy(policy_frame, policy_name="random_policy", top_fraction=top_fraction)
        oracle_row = _matching_policy(policy_frame, policy_name="oracle_true_effect", top_fraction=top_fraction)

        comparison = {
            "policy_name": preferred_policy_name,
            "score_col": preferred_score_col,
            "top_fraction": top_fraction,
            "selected_rows": int(preferred_row["selected_rows"]),
            "mean_true_effect": _optional_float(preferred_row.get("mean_true_effect")),
            "true_effect_lift": _optional_float(preferred_row.get("true_effect_lift")),
            "expected_incremental_conversions": _optional_float(
                preferred_row.get("expected_incremental_conversions")
            ),
            "expected_incremental_conversions_per_1k_selected": _optional_float(
                preferred_row.get("expected_incremental_conversions_per_1k_selected")
            ),
            "random_expected_incremental_conversions": _optional_float(
                None if random_row is None else random_row.get("expected_incremental_conversions")
            ),
            "oracle_expected_incremental_conversions": _optional_float(
                None if oracle_row is None else oracle_row.get("expected_incremental_conversions")
            ),
            "incremental_conversions_vs_random": _difference(
                preferred_row.get("expected_incremental_conversions"),
                None if random_row is None else random_row.get("expected_incremental_conversions"),
            ),
            "oracle_regret_incremental_conversions": _optional_float(
                preferred_row.get("oracle_regret_incremental_conversions")
            ),
            "oracle_capture_rate": _safe_ratio(
                _optional_float(preferred_row.get("expected_incremental_conversions")),
                _optional_float(None if oracle_row is None else oracle_row.get("expected_incremental_conversions")),
            ),
        }
        comparison_rows.append(comparison)

    comparison_frame = pd.DataFrame(comparison_rows)
    recommendation_fraction = _choose_recommendation_fraction(
        comparison_frame["top_fraction"].tolist(),
        requested_fraction=recommended_top_fraction,
    )
    recommended_row = comparison_frame[
        comparison_frame["top_fraction"] == recommendation_fraction
    ].iloc[0]
    efficiency_row = comparison_frame.sort_values(
        ["expected_incremental_conversions_per_1k_selected", "top_fraction"],
        ascending=[False, True],
    ).iloc[0]
    oracle_capture_row = comparison_frame.sort_values(
        ["oracle_capture_rate", "top_fraction"],
        ascending=[False, True],
    ).iloc[0]

    summary = {
        "recommended_policy_name": preferred_policy_name,
        "recommended_score_col": preferred_score_col,
        "recommended_top_fraction": float(recommended_row["top_fraction"]),
        "recommended_selected_rows": int(recommended_row["selected_rows"]),
        "recommended_expected_incremental_conversions": _optional_float(
            recommended_row["expected_incremental_conversions"]
        ),
        "recommended_incremental_conversions_vs_random": _optional_float(
            recommended_row["incremental_conversions_vs_random"]
        ),
        "recommended_oracle_capture_rate": _optional_float(recommended_row["oracle_capture_rate"]),
        "recommended_oracle_regret_incremental_conversions": _optional_float(
            recommended_row["oracle_regret_incremental_conversions"]
        ),
        "highest_efficiency_top_fraction": float(efficiency_row["top_fraction"]),
        "highest_efficiency_incremental_conversions_per_1k_selected": _optional_float(
            efficiency_row["expected_incremental_conversions_per_1k_selected"]
        ),
        "highest_oracle_capture_top_fraction": float(oracle_capture_row["top_fraction"]),
        "highest_oracle_capture_rate": _optional_float(oracle_capture_row["oracle_capture_rate"]),
    }

    return {
        "report_type": "policy_decision_report",
        "preferred_score_col": preferred_score_col,
        "summary": summary,
        "policy_comparisons": _clean_records(comparison_frame),
        "caveats": [
            "Semi-synthetic policy results validate ranking and budget-comparison mechanics.",
            "Observational production policy decisions still require causal assumptions or experiment validation.",
            "Rank-style benchmark features are not serving-time-safe without a separate point-in-time design.",
        ],
    }


def _validate_inputs(df: pd.DataFrame, *, score_cols: list[str], top_fractions: list[float]) -> None:
    if df.empty:
        raise ValueError("policy simulation requires at least one row")
    missing_scores = [score_col for score_col in score_cols if score_col not in df.columns]
    if missing_scores:
        raise ValueError(f"policy score columns are missing from input: {missing_scores}")
    for top_fraction in top_fractions:
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fractions must be between 0 and 1")


def _summarize_selection(
    df: pd.DataFrame,
    *,
    policy_name: str,
    score_col: str | None,
    selected_df: pd.DataFrame,
    full_df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    true_effect_col: str | None,
    top_fraction: float,
) -> dict[str, float | int | str | None]:
    selected_rows = len(selected_df)
    summary: dict[str, float | int | str | None] = {
        "policy_name": policy_name,
        "score_col": score_col,
        "top_fraction": float(top_fraction),
        "selected_rows": int(selected_rows),
        "selected_share": float(selected_rows / len(full_df)),
        "mean_score": None if score_col is None else float(selected_df[score_col].mean()),
        "observed_outcome_rate": float(selected_df[outcome_col].mean()),
        "baseline_outcome_rate": float(full_df[outcome_col].mean()),
        "observed_outcome_lift": _safe_ratio(
            float(selected_df[outcome_col].mean()),
            float(full_df[outcome_col].mean()),
        ),
        "treatment_rate": float(selected_df[treatment_col].mean()),
        "baseline_treatment_rate": float(full_df[treatment_col].mean()),
    }

    if true_effect_col is not None:
        selected_effect_sum = float(selected_df[true_effect_col].sum())
        baseline_effect_mean = float(full_df[true_effect_col].mean())
        selected_effect_mean = float(selected_df[true_effect_col].mean())
        summary.update(
            {
                "mean_true_effect": selected_effect_mean,
                "baseline_true_effect": baseline_effect_mean,
                "true_effect_lift": _safe_ratio(selected_effect_mean, baseline_effect_mean),
                "expected_incremental_conversions": selected_effect_sum,
                "expected_incremental_conversions_per_1k_selected": selected_effect_sum / selected_rows * 1000.0,
            }
        )

    return summary


def _add_oracle_regret(policy_frame: pd.DataFrame) -> pd.DataFrame:
    enriched = policy_frame.copy()
    oracle_by_fraction = (
        enriched[enriched["policy_name"] == "oracle_true_effect"]
        .set_index("top_fraction")["expected_incremental_conversions"]
        .to_dict()
    )
    enriched["oracle_expected_incremental_conversions"] = enriched["top_fraction"].map(oracle_by_fraction)
    enriched["oracle_regret_incremental_conversions"] = (
        enriched["oracle_expected_incremental_conversions"] - enriched["expected_incremental_conversions"]
    )
    enriched.loc[
        enriched["oracle_expected_incremental_conversions"].isna(),
        "oracle_regret_incremental_conversions",
    ] = None
    return enriched


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _matching_policy(policy_frame: pd.DataFrame, *, policy_name: str, top_fraction: float) -> dict[str, Any] | None:
    matched = policy_frame[
        (policy_frame["policy_name"] == policy_name)
        & (policy_frame["top_fraction"].astype(float) == float(top_fraction))
    ]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def _choose_recommendation_fraction(
    available_fractions: list[float],
    *,
    requested_fraction: float | None,
) -> float:
    if not available_fractions:
        raise ValueError("at least one policy fraction is required for a decision report")
    target = 0.1 if requested_fraction is None else float(requested_fraction)
    return min(available_fractions, key=lambda fraction: abs(float(fraction) - target))


def _difference(value: Any, baseline: Any) -> float | None:
    clean_value = _optional_float(value)
    clean_baseline = _optional_float(baseline)
    if clean_value is None or clean_baseline is None:
        return None
    return clean_value - clean_baseline


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(as_float):
        return None
    return as_float


def _clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    return [{key: _clean_value(value) for key, value in record.items()} for record in records]


def _clean_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
