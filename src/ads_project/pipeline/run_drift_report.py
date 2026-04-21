from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from ads_project.artifacts import build_run_manifest, current_git_commit, make_run_dir, write_csv, write_json, write_yaml
from ads_project.config import load_yaml_config
from ads_project.data.io import read_parquet
from ads_project.data.schema import validate_baseline_source_quality, validate_baseline_source_schema
from ads_project.features import apply_feature_builder
from ads_project.models.splits import time_ordered_train_validation_test_split
from ads_project.monitoring import categorical_drift_report, numeric_drift_report
from ads_project.pipeline.run_uplift import _coerce_str_list, uplift_spec_from_config
from ads_project.uplift import add_semisynthetic_uplift_columns, fit_doubly_robust_baseline, predict_doubly_robust_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature and score drift checks.")
    parser.add_argument(
        "--config",
        default="configs/drift_semisynthetic_xgboost_smoke.yaml",
        help="Path to YAML config for drift reporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    dataset_path = Path(config["dataset_path"])
    feature_builder = config.get("feature_builder")
    timestamp_col = str(config.get("timestamp_col", "timestamp"))
    train_fraction = float(config.get("train_fraction", 0.7))
    validation_fraction = float(config.get("validation_fraction", 0.1))
    output_dir = Path(config.get("output_dir", "artifacts/runs"))
    run_name = str(config.get("run_name", "drift_report"))
    max_rows = config.get("max_rows")
    seed = int(config.get("synthetic_seed", 42))
    score_cols = _coerce_str_list(
        config.get("score_cols", ["observational_score", "doubly_robust_score"]),
        field_name="score_cols",
    )
    drift_numeric_features = _coerce_str_list(
        config.get("drift_numeric_features", config.get("numeric_features", [])),
        field_name="drift_numeric_features",
    )
    drift_categorical_features = _coerce_str_list(
        config.get("drift_categorical_features", config.get("categorical_features", [])),
        field_name="drift_categorical_features",
    )
    psi_bins = int(config.get("psi_bins", 10))
    categorical_top_n = int(config.get("categorical_top_n", 20))

    print(f"Loading drift data from: {dataset_path}")
    df = read_parquet(dataset_path)
    if max_rows is not None:
        df = df.iloc[: int(max_rows)].copy()
        print(f"Using max_rows subset: {len(df)}")

    validate_baseline_source_schema(df)
    validate_baseline_source_quality(df)
    df = apply_feature_builder(df, builder_name=feature_builder)
    if feature_builder not in (None, "none"):
        print(f"Applied feature builder: {feature_builder}")

    true_effect_col = str(config.get("true_effect_col", "true_treatment_effect"))
    synthetic_config = dict(config)
    synthetic_config["treatment_col"] = str(config.get("synthetic_treatment_col", "synthetic_treatment"))
    synthetic_config["outcome_col"] = str(config.get("synthetic_outcome_col", "synthetic_outcome"))
    spec = uplift_spec_from_config(synthetic_config)
    df = add_semisynthetic_uplift_columns(
        df,
        treatment_col=spec.treatment_col,
        outcome_col=spec.outcome_col,
        true_effect_col=true_effect_col,
        seed=seed,
    )

    reference_df, validation_df, current_df = time_ordered_train_validation_test_split(
        df,
        timestamp_col=timestamp_col,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    print(
        f"Reference rows: {len(reference_df)} | Validation rows: {len(validation_df)} | Current rows: {len(current_df)}"
    )

    models = fit_doubly_robust_baseline(reference_df, spec=spec)
    reference_scores = predict_doubly_robust_scores(reference_df, models=models, spec=spec)
    current_scores = predict_doubly_robust_scores(current_df, models=models, spec=spec)
    reference_scored = reference_df.copy()
    current_scored = current_df.copy()
    for column in reference_scores.columns:
        reference_scored[column] = reference_scores[column]
        current_scored[column] = current_scores[column]

    numeric_feature_report = numeric_drift_report(
        reference_scored,
        current_scored,
        columns=drift_numeric_features,
        bins=psi_bins,
    )
    categorical_feature_report = categorical_drift_report(
        reference_scored,
        current_scored,
        columns=drift_categorical_features,
        top_n=categorical_top_n,
    )
    score_report = numeric_drift_report(
        reference_scored,
        current_scored,
        columns=score_cols,
        bins=psi_bins,
    )
    feature_report = [*numeric_feature_report, *categorical_feature_report]
    feature_frame = pd.DataFrame(feature_report)
    score_frame = pd.DataFrame(score_report)

    summary = {
        "reference_rows": len(reference_df),
        "validation_rows": len(validation_df),
        "current_rows": len(current_df),
        "numeric_feature_count": len(drift_numeric_features),
        "categorical_feature_count": len(drift_categorical_features),
        "score_count": len(score_cols),
        "max_feature_psi": _max_metric(feature_report, metric="psi"),
        "max_score_psi": _max_metric(score_report, metric="psi"),
    }
    run_summary = {
        "dataset_path": str(dataset_path),
        "feature_builder": feature_builder,
        "synthetic_seed": seed,
        "score_cols": score_cols,
        "drift_numeric_features": drift_numeric_features,
        "drift_categorical_features": drift_categorical_features,
        "psi_bins": psi_bins,
        "categorical_top_n": categorical_top_n,
        "learner_type": spec.learner_type,
        "learner_params": spec.resolved_learner_params,
        "row_counts": {
            "reference_rows": len(reference_df),
            "validation_rows": len(validation_df),
            "current_rows": len(current_df),
        },
    }

    run_dir = make_run_dir(output_dir, run_name=run_name)
    write_yaml(config, run_dir / "config.yaml")
    write_json(run_summary, run_dir / "run_summary.json")
    write_json(summary, run_dir / "drift_summary.json")
    write_json({"features": feature_report}, run_dir / "feature_drift.json")
    write_json({"scores": score_report}, run_dir / "score_drift.json")
    write_csv(feature_frame, run_dir / "feature_drift.csv")
    write_csv(score_frame, run_dir / "score_drift.csv")

    manifest = build_run_manifest(
        run_dir=run_dir,
        run_name=run_name,
        pipeline_name="run_drift_report",
        config_path=config_path,
        dataset_path=dataset_path,
        train_rows=len(reference_df),
        validation_rows=len(validation_df),
        test_rows=len(current_df),
        metrics=summary,
        validation_metrics=None,
        git_commit=current_git_commit(cwd=Path.cwd()),
        artifacts={
            "config": "config.yaml",
            "run_summary": "run_summary.json",
            "drift_summary": "drift_summary.json",
            "feature_drift": "feature_drift.json",
            "feature_drift_csv": "feature_drift.csv",
            "score_drift": "score_drift.json",
            "score_drift_csv": "score_drift.csv",
        },
        extra_metadata={
            "report_type": "feature_and_score_drift",
            "reference_split": "train",
            "current_split": "test",
        },
    )
    write_json(manifest, run_dir / "manifest.json")

    print(f"Max feature PSI: {summary['max_feature_psi']:.6f}")
    print(f"Max score PSI: {summary['max_score_psi']:.6f}")
    print(f"Saved drift artifacts to: {run_dir}")


def _max_metric(rows: list[dict[str, Any]], *, metric: str) -> float | None:
    values = [row.get(metric) for row in rows if row.get(metric) is not None]
    if not values:
        return None
    return float(max(values))


if __name__ == "__main__":
    main()
