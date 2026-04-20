from __future__ import annotations

import pandas as pd


def build_batch_score_output(
    df: pd.DataFrame,
    *,
    score_df: pd.DataFrame,
    id_columns: list[str],
    score_cols: list[str],
    preferred_score_col: str,
    recommended_top_fraction: float,
    optional_columns: list[str] | None = None,
) -> pd.DataFrame:
    if not 0 < recommended_top_fraction <= 1:
        raise ValueError("recommended_top_fraction must be between 0 and 1")

    missing_id_columns = [column for column in id_columns if column not in df.columns]
    if missing_id_columns:
        raise ValueError(f"batch score input is missing id columns: {missing_id_columns}")

    missing_score_columns = [column for column in score_cols if column not in score_df.columns]
    if missing_score_columns:
        raise ValueError(f"batch score input is missing score columns: {missing_score_columns}")
    if preferred_score_col not in score_df.columns:
        raise ValueError(f"preferred_score_col is missing from score columns: {preferred_score_col}")

    output_columns = [*id_columns]
    for column in optional_columns or []:
        if column in df.columns and column not in output_columns:
            output_columns.append(column)

    output = df[output_columns].copy().reset_index(drop=True)
    scores = score_df[score_cols].copy().reset_index(drop=True)
    for column in score_cols:
        output[column] = scores[column]

    score_rank = output[preferred_score_col].rank(method="first", ascending=False)
    output["policy_score_col"] = preferred_score_col
    output["policy_score_rank"] = score_rank.astype(int)
    output["policy_score_percentile"] = 1.0 - (score_rank - 1.0) / max(len(output), 1)

    selected_rows = max(1, int(len(output) * recommended_top_fraction))
    output["recommended_top_fraction"] = float(recommended_top_fraction)
    output["recommended_policy"] = (output["policy_score_rank"] <= selected_rows).astype(int)

    ordered_columns = [
        *output_columns,
        *score_cols,
        "policy_score_col",
        "policy_score_rank",
        "policy_score_percentile",
        "recommended_top_fraction",
        "recommended_policy",
    ]
    return output[ordered_columns].sort_values("policy_score_rank").reset_index(drop=True)


def batch_score_summary(
    scored_output: pd.DataFrame,
    *,
    preferred_score_col: str,
    recommended_top_fraction: float,
    true_effect_col: str | None = None,
) -> dict[str, float | int | str | None]:
    selected = scored_output[scored_output["recommended_policy"] == 1]
    summary: dict[str, float | int | str | None] = {
        "rows": int(len(scored_output)),
        "recommended_rows": int(len(selected)),
        "recommended_share": float(len(selected) / len(scored_output)) if len(scored_output) else 0.0,
        "preferred_score_col": preferred_score_col,
        "recommended_top_fraction": float(recommended_top_fraction),
        "mean_preferred_score": float(scored_output[preferred_score_col].mean()),
        "recommended_mean_preferred_score": float(selected[preferred_score_col].mean()),
    }
    if true_effect_col is not None and true_effect_col in scored_output.columns:
        expected_incremental = float(selected[true_effect_col].sum())
        summary.update(
            {
                "mean_true_effect": float(scored_output[true_effect_col].mean()),
                "recommended_mean_true_effect": float(selected[true_effect_col].mean()),
                "recommended_expected_incremental_conversions": expected_incremental,
                "recommended_expected_incremental_conversions_per_1k": expected_incremental
                / len(selected)
                * 1000.0
                if len(selected)
                else None,
            }
        )
    return summary
