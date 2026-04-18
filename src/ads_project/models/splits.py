from __future__ import annotations

import math

import pandas as pd


def time_ordered_train_validation_test_split(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    train_end = math.floor(len(ordered) * train_fraction)
    validation_end = math.floor(len(ordered) * (train_fraction + validation_fraction))

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    return train, validation, test


def time_ordered_train_test_split(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, _, test = time_ordered_train_validation_test_split(
        df,
        timestamp_col=timestamp_col,
        train_fraction=train_fraction,
        validation_fraction=0.0,
    )
    return train, test
