from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype


@dataclass(frozen=True)
class ColumnContract:
    name: str
    kind: str


BASELINE_REQUIRED_SOURCE_COLUMNS: tuple[ColumnContract, ...] = (
    ColumnContract("timestamp", "integer"),
    ColumnContract("uid", "integer"),
    ColumnContract("campaign", "integer"),
    ColumnContract("click", "integer"),
    ColumnContract("cost", "numeric"),
    ColumnContract("cpo", "numeric"),
    ColumnContract("time_since_last_click", "integer"),
    ColumnContract("cat1", "integer"),
    ColumnContract("cat2", "integer"),
    ColumnContract("cat3", "integer"),
    ColumnContract("cat4", "integer"),
    ColumnContract("cat5", "integer"),
    ColumnContract("cat6", "integer"),
    ColumnContract("cat7", "integer"),
    ColumnContract("cat8", "integer"),
    ColumnContract("cat9", "integer"),
)


def validate_baseline_source_schema(df: pd.DataFrame) -> None:
    _validate_contract(df, BASELINE_REQUIRED_SOURCE_COLUMNS, schema_name="baseline source")
    validate_binary_label(df, label_col="click")


def validate_baseline_training_schema(
    df: pd.DataFrame,
    *,
    label_col: str,
    timestamp_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> None:
    contracts = [
        ColumnContract(timestamp_col, "integer"),
        ColumnContract(label_col, "integer"),
    ]
    contracts.extend(ColumnContract(feature, "numeric") for feature in numeric_features)
    contracts.extend(ColumnContract(feature, "integer") for feature in categorical_features)

    deduped_contracts: list[ColumnContract] = []
    seen: set[str] = set()
    for contract in contracts:
        if contract.name in seen:
            continue
        seen.add(contract.name)
        deduped_contracts.append(contract)

    _validate_contract(df, deduped_contracts, schema_name="baseline training")
    validate_binary_label(df, label_col=label_col)


def validate_binary_label(df: pd.DataFrame, *, label_col: str) -> None:
    label_values = set(pd.Series(df[label_col]).dropna().unique().tolist())
    invalid = sorted(value for value in label_values if value not in {0, 1})
    if invalid:
        raise ValueError(f"{label_col} must be binary with values in {{0, 1}}; found {invalid}")


def validate_baseline_source_quality(df: pd.DataFrame) -> None:
    required_non_null = [contract.name for contract in BASELINE_REQUIRED_SOURCE_COLUMNS]
    null_columns = [column for column in required_non_null if df[column].isna().any()]
    if null_columns:
        raise ValueError(f"baseline source has nulls in required columns: {null_columns}")

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        raise ValueError(f"baseline source contains {duplicate_rows} fully duplicated rows")

    duplicate_keys = int(df.duplicated(subset=["uid", "timestamp", "campaign"]).sum())
    if duplicate_keys:
        raise ValueError(
            "baseline source contains duplicate uid/timestamp/campaign keys: "
            f"{duplicate_keys}"
        )

    _validate_min_value(df, column="timestamp", minimum=0)
    _validate_min_value(df, column="cost", minimum=0)
    _validate_min_value(df, column="cpo", minimum=0)
    _validate_min_value(df, column="time_since_last_click", minimum=-1)
    _validate_min_value(df, column="campaign", minimum=0)
    for feature in [f"cat{i}" for i in range(1, 10)]:
        _validate_min_value(df, column=feature, minimum=0)


def _validate_contract(
    df: pd.DataFrame,
    contracts: list[ColumnContract] | tuple[ColumnContract, ...],
    *,
    schema_name: str,
) -> None:
    missing_columns = [contract.name for contract in contracts if contract.name not in df.columns]
    if missing_columns:
        raise ValueError(f"{schema_name} is missing required columns: {missing_columns}")

    wrong_type_columns: list[str] = []
    for contract in contracts:
        series = df[contract.name]
        if contract.kind == "numeric" and not is_numeric_dtype(series):
            wrong_type_columns.append(f"{contract.name} expected numeric, found {series.dtype}")
        elif contract.kind == "integer" and not is_integer_dtype(series):
            wrong_type_columns.append(f"{contract.name} expected integer, found {series.dtype}")

    if wrong_type_columns:
        raise ValueError(f"{schema_name} has incompatible column types: {wrong_type_columns}")


def _validate_min_value(df: pd.DataFrame, *, column: str, minimum: float) -> None:
    actual_min = float(df[column].min())
    if actual_min < minimum:
        raise ValueError(
            f"baseline source column {column} must be >= {minimum}; found minimum {actual_min}"
        )
