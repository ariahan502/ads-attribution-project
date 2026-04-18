from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from ads_project.artifacts import (
    build_run_manifest,
    current_git_commit,
    make_run_dir,
    write_json,
    write_model,
    write_yaml,
)
from ads_project.config import load_yaml_config
from ads_project.data.io import read_parquet
from ads_project.data.schema import (
    validate_baseline_source_quality,
    validate_baseline_source_schema,
    validate_baseline_training_schema,
)
from ads_project.evaluation.metrics import (
    binary_classification_metrics,
    calibration_and_lift_summary,
    slice_level_report,
)
from ads_project.features import add_campaign_ctr_encoding, apply_feature_builder
from ads_project.models.baseline import BaselineSpec, fit_model, predict_scores
from ads_project.models.splits import time_ordered_train_validation_test_split


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


def _coerce_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return deepcopy(value)


def baseline_spec_from_config(config: dict[str, Any]) -> BaselineSpec:
    model_type = str(config.get("model_type", "logistic_regression"))
    model_params = _coerce_mapping(config.get("model_params"), field_name="model_params")

    if model_type == "logistic_regression" and "max_iter" in config and "max_iter" not in model_params:
        model_params["max_iter"] = int(config.get("max_iter", 100))

    return BaselineSpec(
        label=str(config["label"]),
        numeric_features=_coerce_str_list(config["numeric_features"], field_name="numeric_features"),
        categorical_features=_coerce_str_list(
            config["categorical_features"], field_name="categorical_features"
        ),
        model_type=model_type,
        model_params=model_params,
    )


def apply_train_only_encodings(
    train_df,
    other_splits: dict[str, Any],
    *,
    spec: BaselineSpec,
    encoding_names: list[str],
):
    metadata: dict[str, Any] = {}
    train_encoded = train_df
    encoded_other_splits = {name: split_df for name, split_df in other_splits.items()}

    for encoding_name in encoding_names:
        if encoding_name == "campaign_ctr":
            train_encoded, encoded_other_splits, encoding_metadata = add_campaign_ctr_encoding(
                train_encoded,
                encoded_other_splits,
                campaign_col="campaign",
                label_col=spec.label,
                output_col="campaign_ctr",
            )
            metadata.update(encoding_metadata)
            continue
        raise ValueError(f"Unsupported train_only_encoding: {encoding_name}")

    return train_encoded, encoded_other_splits, metadata


def numeric_features_before_train_only_encodings(
    spec: BaselineSpec,
    *,
    encoding_names: list[str],
) -> list[str]:
    train_only_generated = set()
    if "campaign_ctr" in encoding_names:
        train_only_generated.add("campaign_ctr")
    return [feature for feature in spec.numeric_features if feature not in train_only_generated]


def run_ctr_training(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    spec = baseline_spec_from_config(config)

    dataset_path = Path(config["dataset_path"])
    timestamp_col = str(config.get("timestamp_col", "timestamp"))
    train_fraction = float(config.get("train_fraction", 0.8))
    validation_fraction = float(config.get("validation_fraction", 0.0))
    output_dir = Path(config.get("output_dir", "artifacts/runs"))
    run_name = str(config.get("run_name", "ctr_baseline"))
    max_rows = config.get("max_rows")
    feature_builder = config.get("feature_builder")
    evaluation_top_campaigns = int(config.get("evaluation_top_campaigns", 10))
    evaluation_time_slices = int(config.get("evaluation_time_slices", 5))
    train_only_encodings = _coerce_str_list(
        config.get("train_only_encodings", []),
        field_name="train_only_encodings",
    )
    config_to_write = config_snapshot if config_snapshot is not None else config

    print(f"Loading training data from: {dataset_path}")
    df = read_parquet(dataset_path)

    if max_rows is not None:
        df = df.iloc[: int(max_rows)].copy()
        print(f"Using max_rows subset: {len(df)}")

    validate_baseline_source_schema(df)
    print("Validated baseline source schema")
    validate_baseline_source_quality(df)
    print("Validated baseline source quality")

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

    train_df, validation_df, test_df = time_ordered_train_validation_test_split(
        df,
        timestamp_col=timestamp_col,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    print(
        f"Train rows: {len(train_df)} | Validation rows: {len(validation_df)} | Test rows: {len(test_df)}"
    )

    encoding_metadata: dict[str, Any] = {}
    if train_only_encodings:
        train_df, encoded_other_splits, encoding_metadata = apply_train_only_encodings(
            train_df,
            {
                "validation": validation_df,
                "test": test_df,
            },
            spec=spec,
            encoding_names=train_only_encodings,
        )
        validation_df = encoded_other_splits["validation"]
        test_df = encoded_other_splits["test"]
        print(f"Applied train-only encodings: {', '.join(train_only_encodings)}")

    validate_baseline_training_schema(
        train_df,
        label_col=spec.label,
        timestamp_col=timestamp_col,
        numeric_features=spec.numeric_features,
        categorical_features=spec.categorical_features,
    )
    if len(validation_df) > 0:
        validate_baseline_training_schema(
            validation_df,
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

    model = fit_model(train_df, spec=spec)
    validation_metrics = None
    if len(validation_df) > 0:
        validation_scores = predict_scores(model, validation_df, spec=spec)
        validation_metrics = binary_classification_metrics(validation_df[spec.label], validation_scores)
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
    write_yaml(config_to_write, run_dir / "config.yaml")
    run_summary = {
        "dataset_path": str(dataset_path),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "label": spec.label,
        "model_type": spec.model_type,
        "model_params": spec.resolved_model_params,
        "feature_builder": feature_builder,
        "train_only_encodings": train_only_encodings,
        "numeric_features": spec.numeric_features,
        "categorical_features": spec.categorical_features,
        "encoding_metadata": encoding_metadata,
    }
    write_json(run_summary, run_dir / "run_summary.json")
    write_json(
        {
            **metrics,
        },
        run_dir / "test_metrics.json",
    )
    if validation_metrics is not None:
        write_json(validation_metrics, run_dir / "validation_metrics.json")
    write_json(evaluation_summary, run_dir / "evaluation_summary.json")
    write_json(slice_report, run_dir / "slice_evaluation.json")
    write_model(model, run_dir / "model.joblib")
    manifest = build_run_manifest(
        run_dir=run_dir,
        run_name=run_name,
        pipeline_name="train_ctr",
        config_path=config_path,
        dataset_path=dataset_path,
        train_rows=len(train_df),
        validation_rows=len(validation_df),
        test_rows=len(test_df),
        metrics=metrics,
        validation_metrics=validation_metrics,
        git_commit=current_git_commit(cwd=Path.cwd()),
        extra_metadata={
            "label": spec.label,
            "model_type": spec.model_type,
            "model_params": spec.resolved_model_params,
            "feature_builder": feature_builder,
            "train_only_encodings": train_only_encodings,
            "numeric_features": spec.numeric_features,
            "categorical_features": spec.categorical_features,
        },
    )
    write_json(manifest, run_dir / "manifest.json")

    print(f"ROC AUC: {metrics['roc_auc']:.6f}")
    print(f"PR AUC: {metrics['pr_auc']:.6f}")
    print(f"Log Loss: {metrics['log_loss']:.6f}")
    if validation_metrics is not None:
        print(f"Validation ROC AUC: {validation_metrics['roc_auc']:.6f}")
    print(f"Model type: {spec.model_type}")
    print("Saved run manifest: manifest.json")
    print(
        "Saved slice evaluation "
        f"(top {evaluation_top_campaigns} campaigns, {evaluation_time_slices} time slices)"
    )
    print(f"Saved run artifacts to: {run_dir}")

    return {
        "run_dir": run_dir,
        "run_summary": run_summary,
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "evaluation_summary": evaluation_summary,
        "slice_report": slice_report,
        "manifest": manifest,
        "model_type": spec.model_type,
        "model_params": spec.resolved_model_params,
        "run_name": run_name,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    run_ctr_training(config, config_path=args.config)


if __name__ == "__main__":
    main()
