from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_raw_tsv(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="gzip", nrows=nrows)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
