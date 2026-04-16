from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ads_project.artifacts import make_run_dir, write_json, write_model, write_yaml
from ads_project.config import load_yaml_config
from ads_project.data.io import read_parquet
from ads_project.data.schema import validate_baseline_training_schema
from ads_project.evaluation.metrics import (
    binary_classification_metrics,
    calibration_and_lift_summary,
    slice_level_report,
)
from ads_project.features import add_campaign_ctr_encoding, build_ctr_features
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


def apply_feature_builder(df, *, builder_name: str | None):
    if builder_name in (None, "none"):
        return df
    if builder_name == "ctr_notebook_v1":
        return build_ctr_features(df)
    raise ValueError(f"Unsupported feature_builder: {builder_name}")


def apply_train_only_encodings(
    train_df,
    test_df,
    *,
    spec: BaselineSpec,
    encoding_names: list[str],
):
    metadata: dict[str, Any] = {}
    train_encoded = train_df
    test_encoded = test_df

    for encoding_name in encoding_names:
        if encoding_name == "campaign_ctr":
            train_encoded, test_encoded, encoding_metadata = add_campaign_ctr_encoding(
                train_encoded,
                test_encoded,
                campaign_col="campaign",
                label_col=spec.label,
                output_col="campaign_ctr",
            )
            metadata.update(encoding_metadata)
            continue
        raise ValueError(f"Unsupported train_only_encoding: {encoding_name}")

    return train_encoded, test_encoded, metadata


def numeric_features_before_train_only_encodings(
    spec: BaselineSpec,
    *,
    encoding_names: list[str],
) -> list[str]:
    train_only_generated = set()
    if "campaign_ctr" in encoding_names:
        train_only_generated.add("campaign_ctr")
    return [feature for feature in spec.numeric_features if feature not in train_only_generated]


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
    feature_builder = config.get("feature_builder")
    evaluation_top_campaigns = int(config.get("evaluation_top_campaigns", 10))
    evaluation_time_slices = int(config.get("evaluation_time_slices", 5))
    train_only_encodings = _coerce_str_list(
        config.get("train_only_encodings", []),
        field_name="train_only_encodings",
    )

    print(f"Loading training data from: {dataset_path}")
    df = read_parquet(dataset_path)

    if max_rows is not None:
        df = df.iloc[: int(max_rows)].copy()
        print(f"Using max_rows subset: {len(df)}")

    validate_baseline_training_schema(
        df,
        label_col=spec.label,
        timestamp_col=timestamp_col,
        numeric_features=["cost", "cpo", "time_since_last_click"],
        categorical_features=spec.categorical_features,
    )
    print("Validated baseline source schema")

    df = apply_feature_builder(df, builder_name=feature_builder)
    if feature_builder not in (None, "none"):
        print(f"Applied feature builder: {feature_builder}")

    validate_baseline_training_schema(
        df,
        label_col=spec.label,
        timestamp_col=timestamp_col,
        numeric_features=numeric_features_before_train_only_encodings(
            spec,
            encoding_names=train_only_encodings,
        ),
        categorical_features=spec.categorical_features,
    )
    print("Validated baseline pre-encoding schema")

    train_df, test_df = time_ordered_train_test_split(
        df,
        timestamp_col=timestamp_col,
        train_fraction=train_fraction,
    )
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    encoding_metadata: dict[str, Any] = {}
    if train_only_encodings:
        train_df, test_df, encoding_metadata = apply_train_only_encodings(
            train_df,
            test_df,
            spec=spec,
            encoding_names=train_only_encodings,
        )
        print(f"Applied train-only encodings: {', '.join(train_only_encodings)}")

    validate_baseline_training_schema(
        train_df,
        label_col=spec.label,
        timestamp_col=timestamp_col,
        numeric_features=spec.numeric_features,
        categorical_features=spec.categorical_features,
    )
    validate_baseline_training_schema(
        test_df,
        label_col=spec.label,
        timestamp_col=timestamp_col,
        numeric_features=spec.numeric_features,
        categorical_features=spec.categorical_features,
    )
    print("Validated baseline training schema")

    model = fit_baseline_model(train_df, spec=spec)
    test_scores = predict_scores(model, test_df, spec=spec)
    metrics = binary_classification_metrics(test_df[spec.label], test_scores)
    evaluation_summary = calibration_and_lift_summary(test_df[spec.label], test_scores)
    slice_report = slice_level_report(
        test_df.assign(pred_score=test_scores),
        label_col=spec.label,
        score_col="pred_score",
        campaign_col="campaign",
        timestamp_col=timestamp_col,
        top_campaigns=evaluation_top_campaigns,
        time_slices=evaluation_time_slices,
    )

    run_dir = make_run_dir(output_dir, run_name=run_name)
    write_yaml(config, run_dir / "config.yaml")
    write_json(
        {
            "dataset_path": str(dataset_path),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "label": spec.label,
            "feature_builder": feature_builder,
            "train_only_encodings": train_only_encodings,
            "numeric_features": spec.numeric_features,
            "categorical_features": spec.categorical_features,
            **metrics,
            **encoding_metadata,
        },
        run_dir / "metrics.json",
    )
    write_json(evaluation_summary, run_dir / "evaluation_summary.json")
    write_json(slice_report, run_dir / "slice_evaluation.json")
    write_model(model, run_dir / "model.joblib")

    print(f"ROC AUC: {metrics['roc_auc']:.6f}")
    print(f"PR AUC: {metrics['pr_auc']:.6f}")
    print(f"Log Loss: {metrics['log_loss']:.6f}")
    print(
        "Saved slice evaluation "
        f"(top {evaluation_top_campaigns} campaigns, {evaluation_time_slices} time slices)"
    )
    print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
