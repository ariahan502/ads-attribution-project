from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_tabular(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    source_path = Path(path)

    if source_path.suffix == ".parquet":
        df = pd.read_parquet(source_path)
        if nrows is not None:
            return df.iloc[:nrows].copy()
        return df

    if source_path.suffix in {".gz", ".tsv", ".csv"}:
        compression = "gzip" if source_path.suffix == ".gz" else "infer"
        separator = "\t" if ".tsv" in source_path.name or source_path.suffix == ".gz" else ","
        return pd.read_csv(source_path, sep=separator, compression=compression, nrows=nrows)

    raise ValueError(f"Unsupported tabular source format: {source_path}")


def read_raw_tsv(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="gzip", nrows=nrows)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
