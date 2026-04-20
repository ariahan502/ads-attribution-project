from __future__ import annotations

import pandas as pd

from ads_project.features.basic import (
    add_click_recency_features,
    add_click_recency_transform_features,
    add_log_cost_features,
    add_rank_features,
)


def build_ctr_features(df: pd.DataFrame) -> pd.DataFrame:
    transformed = add_log_cost_features(df)
    transformed = add_click_recency_features(transformed)
    return transformed


def build_ctr_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    transformed = build_ctr_features(df)
    transformed = add_click_recency_transform_features(transformed)
    return transformed


def build_semisynthetic_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    transformed = build_ctr_features_v2(df)
    transformed = add_rank_features(transformed)
    return transformed


FEATURE_BUILDERS = {
    "ctr_notebook_v1": build_ctr_features,
    "ctr_notebook_v2": build_ctr_features_v2,
    "semisynthetic_rank_v1": build_semisynthetic_rank_features,
}


def apply_feature_builder(df: pd.DataFrame, *, builder_name: str | None) -> pd.DataFrame:
    if builder_name in (None, "none"):
        return df

    try:
        builder = FEATURE_BUILDERS[builder_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported feature_builder: {builder_name}") from exc

    return builder(df)
