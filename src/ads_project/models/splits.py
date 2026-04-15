from __future__ import annotations

import math

import pandas as pd


def time_ordered_train_test_split(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    split_index = math.floor(len(ordered) * train_fraction)

    train = ordered.iloc[:split_index].copy()
    test = ordered.iloc[split_index:].copy()
    return train, test
