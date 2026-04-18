from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()
    transformed["log_cost"] = np.log1p(transformed["cost"])
    transformed["log_cpo"] = np.log1p(transformed["cpo"])
    return transformed


def add_click_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()

    cleaned_time_since_last_click = transformed["time_since_last_click"].replace(-1, np.nan)
    transformed["time_since_last_click"] = cleaned_time_since_last_click
    transformed["has_prev_click"] = cleaned_time_since_last_click.notna().astype(int)

    return transformed


def add_click_recency_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()

    cleaned_time_since_last_click = transformed["time_since_last_click"].replace(-1, np.nan)
    recency_for_log = cleaned_time_since_last_click.fillna(0.0)

    transformed["log_time_since_last_click"] = np.log1p(recency_for_log)

    recency_bucket = pd.cut(
        recency_for_log,
        bins=[-0.5, 0.5, 3600, 86400, 604800, np.inf],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    )
    transformed["recency_bucket"] = recency_bucket.astype(int)

    return transformed
