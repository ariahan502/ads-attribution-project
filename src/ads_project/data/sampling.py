from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ads_project.data.io import read_tabular, write_parquet
from ads_project.data.schema import validate_baseline_source_schema


@dataclass(frozen=True)
class SampleSpec:
    source_path: Path
    sample_path: Path
    nrows: int


def build_head_sample(spec: SampleSpec) -> tuple[int, int]:
    df = read_tabular(spec.source_path, nrows=spec.nrows)
    validate_baseline_source_schema(df)
    write_parquet(df, spec.sample_path)
    return df.shape
