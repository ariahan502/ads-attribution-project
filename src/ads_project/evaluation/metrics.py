from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def _safe_roc_auc(y_true_series: pd.Series, y_score_series: pd.Series) -> float | None:
    if y_true_series.nunique() < 2:
        return None
    return float(roc_auc_score(y_true_series, y_score_series))


def _safe_pr_auc(y_true_series: pd.Series, y_score_series: pd.Series) -> float:
    return float(average_precision_score(y_true_series, y_score_series))


def _safe_log_loss(y_true_series: pd.Series, y_score_series: pd.Series) -> float:
    return float(log_loss(y_true_series, y_score_series, labels=[0, 1]))


def binary_classification_metrics(y_true, y_score) -> dict[str, float]:
    y_true_series = pd.Series(np.asarray(y_true), copy=False).astype(float)
    y_score_series = pd.Series(np.asarray(y_score), copy=False).astype(float)

    return {
        "roc_auc": _safe_roc_auc(y_true_series, y_score_series),
        "pr_auc": _safe_pr_auc(y_true_series, y_score_series),
        "log_loss": _safe_log_loss(y_true_series, y_score_series),
        "brier_score": float(np.mean((y_score_series - y_true_series) ** 2)),
    }


def calibration_and_lift_summary(
    y_true,
    y_score,
    *,
    bins: int = 10,
) -> dict[str, float | list[dict[str, float | int]]]:
    evaluation_df = pd.DataFrame(
        {
            "y_true": pd.Series(np.asarray(y_true), copy=False).astype(float),
            "y_score": pd.Series(np.asarray(y_score), copy=False).astype(float),
        }
    ).sort_values("y_score", ascending=False)

    baseline_positive_rate = float(evaluation_df["y_true"].mean())
    evaluation_df["bin"] = pd.qcut(
        evaluation_df["y_score"],
        q=min(bins, len(evaluation_df)),
        labels=False,
        duplicates="drop",
    )

    grouped = (
        evaluation_df.groupby("bin", observed=True)
        .agg(
            rows=("y_true", "size"),
            positives=("y_true", "sum"),
            avg_score=("y_score", "mean"),
            actual_rate=("y_true", "mean"),
        )
        .reset_index()
        .sort_values("avg_score", ascending=False)
        .reset_index(drop=True)
    )
    grouped["lift_vs_baseline"] = grouped["actual_rate"] / baseline_positive_rate

    calibration_mae = float(np.mean(np.abs(grouped["avg_score"] - grouped["actual_rate"])))

    return {
        "baseline_positive_rate": baseline_positive_rate,
        "calibration_bin_mae": calibration_mae,
        "top_bin_lift": float(grouped["lift_vs_baseline"].iloc[0]),
        "bottom_bin_lift": float(grouped["lift_vs_baseline"].iloc[-1]),
        "decile_summary": [
            {
                "bin": int(row["bin"]),
                "rank_by_score": int(idx),
                "rows": int(row["rows"]),
                "positives": float(row["positives"]),
                "avg_score": float(row["avg_score"]),
                "actual_rate": float(row["actual_rate"]),
                "lift_vs_baseline": float(row["lift_vs_baseline"]),
            }
            for idx, (_, row) in enumerate(grouped.iterrows(), start=1)
        ],
    }


def slice_level_report(
    df: pd.DataFrame,
    *,
    label_col: str,
    score_col: str,
    campaign_col: str,
    timestamp_col: str,
    top_campaigns: int = 10,
    time_slices: int = 5,
) -> dict[str, list[dict[str, float | int | str | None]]]:
    evaluation_df = df.copy()
    evaluation_df[label_col] = np.asarray(evaluation_df[label_col], dtype=float)
    evaluation_df[score_col] = np.asarray(evaluation_df[score_col], dtype=float)

    campaign_counts = evaluation_df[campaign_col].value_counts().head(top_campaigns)
    campaign_rows = evaluation_df[evaluation_df[campaign_col].isin(campaign_counts.index)].copy()

    campaign_summary = [
        _group_metrics(group, label_col=label_col, score_col=score_col, group_name=str(campaign))
        for campaign, group in campaign_rows.groupby(campaign_col, sort=False)
    ]
    campaign_summary.sort(key=lambda row: row["rows"], reverse=True)

    ordered = evaluation_df.sort_values(timestamp_col).reset_index(drop=True)
    ordered["time_slice"] = pd.qcut(
        ordered.index,
        q=min(time_slices, len(ordered)),
        labels=False,
        duplicates="drop",
    )
    time_slice_summary = [
        {
            **_group_metrics(group, label_col=label_col, score_col=score_col, group_name=f"slice_{int(time_slice) + 1}"),
            "time_slice": int(time_slice) + 1,
            "timestamp_min": float(group[timestamp_col].min()),
            "timestamp_max": float(group[timestamp_col].max()),
        }
        for time_slice, group in ordered.groupby("time_slice", sort=True)
    ]

    return {
        "campaign_summary": campaign_summary,
        "time_slice_summary": time_slice_summary,
    }


def _group_metrics(
    group: pd.DataFrame,
    *,
    label_col: str,
    score_col: str,
    group_name: str,
) -> dict[str, float | int | str | None]:
    y_true_series = group[label_col].astype(float)
    y_score_series = group[score_col].astype(float)

    return {
        "group": group_name,
        "rows": int(len(group)),
        "positive_rate": float(y_true_series.mean()),
        "avg_score": float(y_score_series.mean()),
        "roc_auc": _safe_roc_auc(y_true_series, y_score_series),
        "pr_auc": _safe_pr_auc(y_true_series, y_score_series),
        "log_loss": _safe_log_loss(y_true_series, y_score_series),
        "brier_score": float(np.mean((y_score_series - y_true_series) ** 2)),
    }
