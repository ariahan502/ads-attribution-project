from __future__ import annotations

import numpy as np
import pandas as pd


def numeric_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    columns: list[str],
    bins: int = 10,
) -> list[dict[str, float | int | str | None]]:
    reports = []
    for column in columns:
        _require_column(reference_df, column=column, frame_name="reference")
        _require_column(current_df, column=column, frame_name="current")

        reference = pd.to_numeric(reference_df[column], errors="coerce")
        current = pd.to_numeric(current_df[column], errors="coerce")
        reports.append(
            {
                "column": column,
                "kind": "numeric",
                "reference_rows": int(len(reference)),
                "current_rows": int(len(current)),
                "reference_missing_rate": float(reference.isna().mean()),
                "current_missing_rate": float(current.isna().mean()),
                "reference_mean": _mean_or_none(reference),
                "current_mean": _mean_or_none(current),
                "mean_delta": _delta(_mean_or_none(current), _mean_or_none(reference)),
                "reference_std": _std_or_none(reference),
                "current_std": _std_or_none(current),
                "psi": _numeric_psi(reference.dropna(), current.dropna(), bins=bins),
            }
        )
    return reports


def categorical_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    columns: list[str],
    top_n: int = 20,
) -> list[dict[str, float | int | str | None]]:
    reports = []
    for column in columns:
        _require_column(reference_df, column=column, frame_name="reference")
        _require_column(current_df, column=column, frame_name="current")

        reference = reference_df[column]
        current = current_df[column]
        reference_counts = reference.value_counts(dropna=False, normalize=True)
        current_counts = current.value_counts(dropna=False, normalize=True)
        top_categories = list(reference_counts.head(top_n).index)
        categories = set(top_categories) | set(current_counts.head(top_n).index)
        new_category_mask = ~current.isin(reference.dropna().unique()) & current.notna()
        new_category_share = float(new_category_mask.mean())
        reports.append(
            {
                "column": column,
                "kind": "categorical",
                "reference_rows": int(len(reference)),
                "current_rows": int(len(current)),
                "reference_missing_rate": float(reference.isna().mean()),
                "current_missing_rate": float(current.isna().mean()),
                "reference_unique": int(reference.nunique(dropna=True)),
                "current_unique": int(current.nunique(dropna=True)),
                "new_category_share": new_category_share,
                "top_reference_category": _string_or_none(reference_counts.index[0])
                if not reference_counts.empty
                else None,
                "top_current_category": _string_or_none(current_counts.index[0]) if not current_counts.empty else None,
                "psi": _categorical_psi(reference_counts, current_counts, categories=categories),
            }
        )
    return reports


def _numeric_psi(reference: pd.Series, current: pd.Series, *, bins: int) -> float | None:
    if reference.empty or current.empty:
        return None

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(reference.quantile(quantiles).to_numpy(dtype=float))
    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts = pd.cut(reference, bins=edges, include_lowest=True).value_counts(sort=False)
    current_counts = pd.cut(current, bins=edges, include_lowest=True).value_counts(sort=False)
    return _psi_from_proportions(
        reference_counts.to_numpy(dtype=float) / len(reference),
        current_counts.to_numpy(dtype=float) / len(current),
    )


def _categorical_psi(
    reference_counts: pd.Series,
    current_counts: pd.Series,
    *,
    categories: set[object],
) -> float | None:
    if not categories:
        return None
    reference_props = np.array([float(reference_counts.get(category, 0.0)) for category in categories])
    current_props = np.array([float(current_counts.get(category, 0.0)) for category in categories])
    return _psi_from_proportions(reference_props, current_props)


def _psi_from_proportions(reference_props: np.ndarray, current_props: np.ndarray) -> float:
    epsilon = 1e-6
    reference_safe = np.clip(reference_props, epsilon, None)
    current_safe = np.clip(current_props, epsilon, None)
    return float(np.sum((current_safe - reference_safe) * np.log(current_safe / reference_safe)))


def _require_column(df: pd.DataFrame, *, column: str, frame_name: str) -> None:
    if column not in df.columns:
        raise ValueError(f"{frame_name} data is missing drift column: {column}")


def _mean_or_none(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _std_or_none(series: pd.Series) -> float | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.std(ddof=0))


def _delta(current: float | None, reference: float | None) -> float | None:
    if current is None or reference is None:
        return None
    return current - reference


def _string_or_none(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value)
