from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ads_project.artifacts import make_run_dir, write_json, write_model, write_yaml
from ads_project.config import load_yaml_config
from ads_project.data.io import read_parquet
from ads_project.evaluation.metrics import binary_classification_metrics
from ads_project.models.baseline import BaselineSpec, fit_baseline_model, predict_scores
from ads_project.models.splits import time_ordered_train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline CTR model.")
    parser.add_argument(
        "--config",
        default="configs/ctr_baseline.yaml",
        help="Path to YAML config for baseline CTR training.",
    )
    return parser.parse_args()


def _coerce_str_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return value


def baseline_spec_from_config(config: dict[str, Any]) -> BaselineSpec:
    return BaselineSpec(
        label=str(config["label"]),
        numeric_features=_coerce_str_list(config["numeric_features"], field_name="numeric_features"),
        categorical_features=_coerce_str_list(
            config["categorical_features"], field_name="categorical_features"
        ),
        max_iter=int(config.get("max_iter", 100)),
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    spec = baseline_spec_from_config(config)

    dataset_path = Path(config["dataset_path"])
    timestamp_col = str(config.get("timestamp_col", "timestamp"))
    train_fraction = float(config.get("train_fraction", 0.8))
    output_dir = Path(config.get("output_dir", "artifacts/runs"))
    run_name = config.get("run_name", "ctr_baseline")
    max_rows = config.get("max_rows")

    print(f"Loading training data from: {dataset_path}")
    df = read_parquet(dataset_path)

    if max_rows is not None:
        df = df.iloc[: int(max_rows)].copy()
        print(f"Using max_rows subset: {len(df)}")

    train_df, test_df = time_ordered_train_test_split(
        df,
        timestamp_col=timestamp_col,
        train_fraction=train_fraction,
    )
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    model = fit_baseline_model(train_df, spec=spec)
    test_scores = predict_scores(model, test_df, spec=spec)
    metrics = binary_classification_metrics(test_df[spec.label], test_scores)

    run_dir = make_run_dir(output_dir, run_name=run_name)
    write_yaml(config, run_dir / "config.yaml")
    write_json(
        {
            "dataset_path": str(dataset_path),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "label": spec.label,
            "numeric_features": spec.numeric_features,
            "categorical_features": spec.categorical_features,
            **metrics,
        },
        run_dir / "metrics.json",
    )
    write_model(model, run_dir / "model.joblib")

    print(f"ROC AUC: {metrics['roc_auc']:.6f}")
    print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
