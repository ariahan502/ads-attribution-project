from __future__ import annotations

import pandas as pd


def add_campaign_ctr_encoding(
    train_df: pd.DataFrame,
    other_splits: dict[str, pd.DataFrame],
    *,
    campaign_col: str,
    label_col: str,
    output_col: str = "campaign_ctr",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, float | int | str]]:
    train_encoded = train_df.copy()
    encoded_other_splits = {name: split_df.copy() for name, split_df in other_splits.items()}

    campaign_ctr = train_encoded.groupby(campaign_col)[label_col].mean()
    global_ctr = float(train_encoded[label_col].mean())

    train_mapped = train_encoded[campaign_col].map(campaign_ctr)
    train_encoded[output_col] = train_mapped.fillna(global_ctr)

    unseen_counts: dict[str, int] = {}
    unseen_rates: dict[str, float] = {}
    for split_name, split_df in encoded_other_splits.items():
        split_mapped = split_df[campaign_col].map(campaign_ctr)
        unseen_mask = split_mapped.isna()
        split_df[output_col] = split_mapped.fillna(global_ctr)
        unseen_counts[split_name] = int(unseen_mask.sum())
        unseen_rates[split_name] = float(unseen_mask.mean())

    metadata: dict[str, float | int | str] = {
        "encoding_name": output_col,
        "campaign_ctr_global_prior": global_ctr,
        "campaign_ctr_train_groups": int(campaign_ctr.shape[0]),
    }
    for split_name, unseen_count in unseen_counts.items():
        metadata[f"campaign_ctr_{split_name}_unseen_rows"] = unseen_count
    for split_name, unseen_rate in unseen_rates.items():
        metadata[f"campaign_ctr_{split_name}_unseen_rate"] = unseen_rate

    return train_encoded, encoded_other_splits, metadata
