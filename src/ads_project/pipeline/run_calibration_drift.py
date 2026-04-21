from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ads_project.artifacts import build_run_manifest, current_git_commit, make_run_dir, write_csv, write_json, write_yaml
from ads_project.config import load_yaml_config
from ads_project.data.io import read_parquet
from ads_project.data.schema import (
    validate_baseline_source_quality,
    validate_baseline_source_schema,
    validate_baseline_training_schema,
)
from ads_project.features import apply_feature_builder
from ads_project.models.baseline import fit_model, predict_scores
from ads_project.models.splits import time_ordered_train_validation_test_split
from ads_project.monitoring import calibration_bin_frame, calibration_drift_report
from ads_project.pipeline.train_ctr import (
    apply_train_only_encodings,
    baseline_spec_from_config,
    numeric_features_before_train_only_encodings,
    _coerce_str_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CTR calibration drift checks.")
    parser.add_argument(
        "--config",
        default="configs/calibration_drift_smoke.yaml",
        help="Path to YAML config for calibration drift reporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_calibration_drift(load_yaml_config(args.config), config_path=args.config)


def run_calibration_drift(
    config: dict[str, Any],
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path)
    spec = baseline_spec_from_config(config)

    dataset_path = Path(config["dataset_path"])
    timestamp_col = str(config.get("timestamp_col", "timestamp"))
    train_fraction = float(config.get("train_fraction", 0.7))
    validation_fraction = float(config.get("validation_fraction", 0.1))
    output_dir = Path(config.get("output_dir", "artifacts/runs"))
    run_name = str(config.get("run_name", "calibration_drift"))
    max_rows = config.get("max_rows")
    feature_builder = config.get("feature_builder")
    calibration_bins = int(config.get("calibration_bins", 10))
    train_only_encodings = _coerce_str_list(
        config.get("train_only_encodings", []),
        field_name="train_only_encodings",
    )

    print(f"Loading calibration drift data from: {dataset_path}")
    df = read_parquet(dataset_path)
    if max_rows is not None:
        df = df.iloc[: int(max_rows)].copy()
        print(f"Using max_rows subset: {len(df)}")

    validate_baseline_source_schema(df)
    validate_baseline_source_quality(df)
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

    train_df, validation_df, current_df = time_ordered_train_validation_test_split(
        df,
        timestamp_col=timestamp_col,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    print(
        f"Train rows: {len(train_df)} | Reference rows: {len(validation_df)} | Current rows: {len(current_df)}"
    )

    if validation_df.empty:
        raise ValueError("calibration drift reporting requires a non-empty validation/reference split")

    encoding_metadata: dict[str, Any] = {}
    if train_only_encodings:
        train_df, encoded_other_splits, encoding_metadata = apply_train_only_encodings(
            train_df,
            {
                "reference": validation_df,
                "current": current_df,
            },
            spec=spec,
            encoding_names=train_only_encodings,
        )
        validation_df = encoded_other_splits["reference"]
        current_df = encoded_other_splits["current"]
        print(f"Applied train-only encodings: {', '.join(train_only_encodings)}")

    for split_name, split_df in (
        ("train", train_df),
        ("reference", validation_df),
        ("current", current_df),
    ):
        validate_baseline_training_schema(
            split_df,
            label_col=spec.label,
            timestamp_col=timestamp_col,
            numeric_features=spec.numeric_features,
            categorical_features=spec.categorical_features,
        )
        print(f"Validated {split_name} calibration drift schema")

    model = fit_model(train_df, spec=spec)
    reference_scores = predict_scores(model, validation_df, spec=spec)
    current_scores = predict_scores(model, current_df, spec=spec)
    report = calibration_drift_report(
        validation_df[spec.label],
        reference_scores,
        current_df[spec.label],
        current_scores,
        bins=calibration_bins,
    )
    bin_frame = calibration_bin_frame(report)

    summary = {key: value for key, value in report.items() if key != "bin_summary"}
    run_summary = {
        "dataset_path": str(dataset_path),
        "label": spec.label,
        "model_type": spec.model_type,
        "model_params": spec.resolved_model_params,
        "feature_builder": feature_builder,
        "train_only_encodings": train_only_encodings,
        "numeric_features": spec.numeric_features,
        "categorical_features": spec.categorical_features,
        "calibration_bins": calibration_bins,
        "encoding_metadata": encoding_metadata,
        "row_counts": {
            "train_rows": len(train_df),
            "reference_rows": len(validation_df),
            "current_rows": len(current_df),
        },
    }

    run_dir = make_run_dir(output_dir, run_name=run_name)
    write_yaml(config, run_dir / "config.yaml")
    write_json(run_summary, run_dir / "run_summary.json")
    write_json(report, run_dir / "calibration_drift.json")
    write_json(summary, run_dir / "calibration_drift_summary.json")
    write_csv(bin_frame, run_dir / "calibration_drift.csv")

    manifest = build_run_manifest(
        run_dir=run_dir,
        run_name=run_name,
        pipeline_name="run_calibration_drift",
        config_path=config_path,
        dataset_path=dataset_path,
        train_rows=len(train_df),
        validation_rows=len(validation_df),
        test_rows=len(current_df),
        metrics=summary,
        validation_metrics=None,
        git_commit=current_git_commit(cwd=Path.cwd()),
        artifacts={
            "config": "config.yaml",
            "run_summary": "run_summary.json",
            "calibration_drift": "calibration_drift.json",
            "calibration_drift_summary": "calibration_drift_summary.json",
            "calibration_drift_csv": "calibration_drift.csv",
        },
        extra_metadata={
            "report_type": "ctr_calibration_drift",
            "reference_split": "validation",
            "current_split": "test",
        },
    )
    write_json(manifest, run_dir / "manifest.json")

    print(f"Reference calibration MAE: {summary['reference_calibration_mae']:.6f}")
    print(f"Current calibration MAE: {summary['current_calibration_mae']:.6f}")
    print(f"Calibration MAE delta: {summary['calibration_mae_delta']:.6f}")
    print(f"Saved calibration drift artifacts to: {run_dir}")

    return {
        "run_dir": run_dir,
        "run_summary": run_summary,
        "report": report,
        "summary": summary,
        "manifest": manifest,
    }


if __name__ == "__main__":
    main()
