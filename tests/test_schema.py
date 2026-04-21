from __future__ import annotations

import pandas as pd
import pytest

from ads_project.data.schema import validate_baseline_source_quality, validate_baseline_source_schema


def _baseline_source_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [1, 2],
            "uid": [10, 11],
            "campaign": [100, 101],
            "click": [0, 1],
            "cost": [0.5, 1.5],
            "cpo": [2.0, 3.0],
            "time_since_last_click": [-1, 3600],
            "cat1": [1, 2],
            "cat2": [1, 2],
            "cat3": [1, 2],
            "cat4": [1, 2],
            "cat5": [1, 2],
            "cat6": [1, 2],
            "cat7": [1, 2],
            "cat8": [1, 2],
            "cat9": [1, 2],
        }
    )


def test_baseline_source_schema_accepts_valid_source() -> None:
    df = _baseline_source_df()

    validate_baseline_source_schema(df)
    validate_baseline_source_quality(df)


def test_baseline_source_schema_rejects_missing_required_column() -> None:
    df = _baseline_source_df().drop(columns=["campaign"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_baseline_source_schema(df)


def test_baseline_source_quality_rejects_duplicate_keys() -> None:
    df = _baseline_source_df()
    df.loc[1, ["uid", "timestamp", "campaign"]] = df.loc[0, ["uid", "timestamp", "campaign"]]

    with pytest.raises(ValueError, match="duplicate uid/timestamp/campaign keys"):
        validate_baseline_source_quality(df)
