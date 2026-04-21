from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def calibration_drift_report(
    reference_y_true,
    reference_y_score,
    current_y_true,
    current_y_score,
    *,
    bins: int = 10,
) -> dict[str, float | int | list[dict[str, float | int]]]:
    if bins < 2:
        raise ValueError("bins must be at least 2")

    reference = _calibration_frame(reference_y_true, reference_y_score, frame_name="reference")
    current = _calibration_frame(current_y_true, current_y_score, frame_name="current")
    edges = _score_edges(reference["y_score"], bins=bins)

    reference_bins = _binned_calibration(reference, edges=edges)
    current_bins = _binned_calibration(current, edges=edges)
    bin_rows = _combine_bin_rows(reference_bins, current_bins)

    reference_summary = _calibration_summary(reference, reference_bins)
    current_summary = _calibration_summary(current, current_bins)

    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "bins": int(len(bin_rows)),
        "reference_positive_rate": reference_summary["positive_rate"],
        "current_positive_rate": current_summary["positive_rate"],
        "positive_rate_delta": current_summary["positive_rate"] - reference_summary["positive_rate"],
        "reference_avg_score": reference_summary["avg_score"],
        "current_avg_score": current_summary["avg_score"],
        "avg_score_delta": current_summary["avg_score"] - reference_summary["avg_score"],
        "reference_log_loss": reference_summary["log_loss"],
        "current_log_loss": current_summary["log_loss"],
        "log_loss_delta": current_summary["log_loss"] - reference_summary["log_loss"],
        "reference_brier_score": reference_summary["brier_score"],
        "current_brier_score": current_summary["brier_score"],
        "brier_score_delta": current_summary["brier_score"] - reference_summary["brier_score"],
        "reference_calibration_mae": reference_summary["calibration_mae"],
        "current_calibration_mae": current_summary["calibration_mae"],
        "calibration_mae_delta": current_summary["calibration_mae"] - reference_summary["calibration_mae"],
        "max_bin_actual_rate_delta": _max_abs_delta(bin_rows, "actual_rate_delta"),
        "max_bin_avg_score_delta": _max_abs_delta(bin_rows, "avg_score_delta"),
        "bin_summary": bin_rows,
    }


def calibration_bin_frame(report: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(report.get("bin_summary", []))


def _calibration_frame(y_true, y_score, *, frame_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "y_true": pd.Series(np.asarray(y_true), copy=False).astype(float),
            "y_score": pd.Series(np.asarray(y_score), copy=False).astype(float),
        }
    )
    if frame.empty:
        raise ValueError(f"{frame_name} calibration data must not be empty")
    if frame["y_true"].isna().any() or frame["y_score"].isna().any():
        raise ValueError(f"{frame_name} calibration data contains null values")
    if not frame["y_true"].isin([0.0, 1.0]).all():
        raise ValueError(f"{frame_name} labels must be binary 0/1 values")
    if ((frame["y_score"] < 0.0) | (frame["y_score"] > 1.0)).any():
        raise ValueError(f"{frame_name} scores must be probabilities between 0 and 1")
    return frame


def _score_edges(reference_scores: pd.Series, *, bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(reference_scores.quantile(quantiles).to_numpy(dtype=float))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _binned_calibration(frame: pd.DataFrame, *, edges: np.ndarray) -> pd.DataFrame:
    binned = frame.copy()
    binned["bin"] = pd.cut(binned["y_score"], bins=edges, include_lowest=True, labels=False)
    grouped = (
        binned.groupby("bin", dropna=False, observed=True)
        .agg(
            rows=("y_true", "size"),
            positives=("y_true", "sum"),
            avg_score=("y_score", "mean"),
            actual_rate=("y_true", "mean"),
        )
        .reset_index()
        .sort_values("bin")
    )
    grouped["abs_calibration_error"] = (grouped["avg_score"] - grouped["actual_rate"]).abs()
    return grouped


def _calibration_summary(frame: pd.DataFrame, bins: pd.DataFrame) -> dict[str, float]:
    return {
        "positive_rate": float(frame["y_true"].mean()),
        "avg_score": float(frame["y_score"].mean()),
        "log_loss": float(log_loss(frame["y_true"], frame["y_score"], labels=[0, 1])),
        "brier_score": float(np.mean((frame["y_score"] - frame["y_true"]) ** 2)),
        "calibration_mae": float(
            np.average(bins["abs_calibration_error"], weights=bins["rows"])
        ),
    }


def _combine_bin_rows(reference_bins: pd.DataFrame, current_bins: pd.DataFrame) -> list[dict[str, float | int]]:
    merged = reference_bins.merge(
        current_bins,
        on="bin",
        how="outer",
        suffixes=("_reference", "_current"),
    ).sort_values("bin")
    rows: list[dict[str, float | int]] = []
    for rank, (_, row) in enumerate(merged.iterrows(), start=1):
        reference_actual = _float_or_zero(row.get("actual_rate_reference"))
        current_actual = _float_or_zero(row.get("actual_rate_current"))
        reference_score = _float_or_zero(row.get("avg_score_reference"))
        current_score = _float_or_zero(row.get("avg_score_current"))
        rows.append(
            {
                "bin": int(rank),
                "reference_rows": int(_float_or_zero(row.get("rows_reference"))),
                "current_rows": int(_float_or_zero(row.get("rows_current"))),
                "reference_positives": float(_float_or_zero(row.get("positives_reference"))),
                "current_positives": float(_float_or_zero(row.get("positives_current"))),
                "reference_avg_score": reference_score,
                "current_avg_score": current_score,
                "avg_score_delta": current_score - reference_score,
                "reference_actual_rate": reference_actual,
                "current_actual_rate": current_actual,
                "actual_rate_delta": current_actual - reference_actual,
            }
        )
    return rows


def _max_abs_delta(rows: list[dict[str, float | int]], key: str) -> float:
    if not rows:
        return 0.0
    return float(max(abs(float(row[key])) for row in rows))


def _float_or_zero(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)
