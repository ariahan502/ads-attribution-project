from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def binary_classification_metrics(y_true, y_score) -> dict[str, float]:
    y_true_series = pd.Series(np.asarray(y_true), copy=False).astype(float)
    y_score_series = pd.Series(np.asarray(y_score), copy=False).astype(float)

    return {
        "roc_auc": float(roc_auc_score(y_true_series, y_score_series)),
        "pr_auc": float(average_precision_score(y_true_series, y_score_series)),
        "log_loss": float(log_loss(y_true_series, y_score_series, labels=[0, 1])),
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
