from __future__ import annotations

import pandas as pd
import pytest

from ads_project.features import apply_feature_builder


def test_semisynthetic_rank_builder_adds_expected_features() -> None:
    df = pd.DataFrame(
        {
            "cost": [0.0, 3.0, 1.0],
            "cpo": [5.0, 1.0, 3.0],
            "time_since_last_click": [-1, 0, 3600],
        }
    )

    transformed = apply_feature_builder(df, builder_name="semisynthetic_rank_v1")

    expected_columns = {
        "log_cost",
        "log_cpo",
        "has_prev_click",
        "log_time_since_last_click",
        "recency_bucket",
        "cost_rank",
        "cpo_rank",
        "recency_rank",
    }
    assert expected_columns.issubset(transformed.columns)
    assert transformed["has_prev_click"].tolist() == [0, 1, 1]
    assert transformed["time_since_last_click"].isna().tolist() == [True, False, False]
    assert transformed["cost_rank"].between(0, 1).all()
    assert transformed["cpo_rank"].between(0, 1).all()
    assert transformed["recency_rank"].between(0, 1).all()


def test_apply_feature_builder_rejects_unknown_builder() -> None:
    with pytest.raises(ValueError, match="Unsupported feature_builder"):
        apply_feature_builder(pd.DataFrame({"cost": [1]}), builder_name="does_not_exist")
