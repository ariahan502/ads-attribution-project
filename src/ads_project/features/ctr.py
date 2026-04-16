from __future__ import annotations

import numpy as np
import pandas as pd


def build_ctr_features(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()

    transformed["log_cost"] = np.log1p(transformed["cost"])
    transformed["log_cpo"] = np.log1p(transformed["cpo"])

    cleaned_time_since_last_click = transformed["time_since_last_click"].replace(-1, np.nan)
    transformed["time_since_last_click"] = cleaned_time_since_last_click
    transformed["has_prev_click"] = cleaned_time_since_last_click.notna().astype(int)

    return transformed


def add_campaign_ctr_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    campaign_col: str,
    label_col: str,
    output_col: str = "campaign_ctr",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    campaign_ctr = train_encoded.groupby(campaign_col)[label_col].mean()
    global_ctr = float(train_encoded[label_col].mean())

    train_mapped = train_encoded[campaign_col].map(campaign_ctr)
    test_mapped = test_encoded[campaign_col].map(campaign_ctr)
    test_unseen_mask = test_mapped.isna()

    train_encoded[output_col] = train_mapped.fillna(global_ctr)
    test_encoded[output_col] = test_mapped.fillna(global_ctr)

    metadata: dict[str, float | int | str] = {
        "encoding_name": output_col,
        "campaign_ctr_global_prior": global_ctr,
        "campaign_ctr_train_groups": int(campaign_ctr.shape[0]),
        "campaign_ctr_test_unseen_rows": int(test_unseen_mask.sum()),
        "campaign_ctr_test_unseen_rate": float(test_unseen_mask.mean()),
    }
    return train_encoded, test_encoded, metadata
