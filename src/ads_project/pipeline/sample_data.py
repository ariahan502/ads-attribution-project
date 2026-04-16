from __future__ import annotations

import argparse
from pathlib import Path

from ads_project.config import load_yaml_config
from ads_project.data.sampling import SampleSpec, build_head_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a parquet sample from the raw TSV dataset.")
    parser.add_argument(
        "--config",
        default="configs/sample.yaml",
        help="Path to YAML config for sample generation.",
    )
    return parser.parse_args()


def sample_spec_from_config(config_path: str | Path) -> SampleSpec:
    config = load_yaml_config(config_path)
    source_path = config.get("source_path", config.get("raw_path"))
    if source_path is None:
        raise ValueError("sample config must define source_path or raw_path")
    return SampleSpec(
        source_path=Path(source_path),
        sample_path=Path(config["sample_path"]),
        nrows=int(config.get("nrows", 1_000_000)),
    )


def main() -> None:
    args = parse_args()
    spec = sample_spec_from_config(args.config)

    print(f"Loading source data from: {spec.source_path}")
    rows, cols = build_head_sample(spec)
    print(f"Loaded shape: ({rows}, {cols})")
    print(f"Saved sample to: {spec.sample_path}")


if __name__ == "__main__":
    main()
