from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ads_project.artifacts import build_run_manifest, current_git_commit, make_run_dir, write_json, write_model, write_yaml
from ads_project.config import load_yaml_config
from ads_project.data.io import read_parquet
from ads_project.data.schema import validate_baseline_source_quality, validate_baseline_source_schema
from ads_project.features import apply_feature_builder
from ads_project.models.splits import time_ordered_train_validation_test_split
from ads_project.pipeline.run_uplift import _coerce_str_list, _evaluate_split, uplift_spec_from_config
from ads_project.uplift import (
    add_semisynthetic_uplift_columns,
    fit_doubly_robust_baseline,
    known_effect_ranking_report,
    predict_doubly_robust_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate uplift rankings on semi-synthetic outcomes.")
    parser.add_argument(
        "--config",
        default="configs/uplift_semisynthetic_smoke.yaml",
        help="Path to YAML config for semi-synthetic uplift evaluation.",
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
    run_name = str(config.get("run_name", "uplift_semisynthetic"))
    max_rows = config.get("max_rows")
    seed = int(config.get("synthetic_seed", 42))
    true_effect_col = str(config.get("true_effect_col", "true_treatment_effect"))
    score_cols = _coerce_str_list(
        config.get("score_cols", ["observational_score", "doubly_robust_score"]),
        field_name="score_cols",
    )

    print(f"Loading semi-synthetic uplift data from: {dataset_path}")
    df = read_parquet(dataset_path)
    if max_rows is not None:
        df = df.iloc[: int(max_rows)].copy()
        print(f"Using max_rows subset: {len(df)}")

    validate_baseline_source_schema(df)
    validate_baseline_source_quality(df)
    df = apply_feature_builder(df, builder_name=feature_builder)
    if feature_builder not in (None, "none"):
        print(f"Applied feature builder: {feature_builder}")

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

    train_df, validation_df, test_df = time_ordered_train_validation_test_split(
        df,
        timestamp_col=timestamp_col,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    print(
        f"Train rows: {len(train_df)} | Validation rows: {len(validation_df)} | Test rows: {len(test_df)}"
    )

    models = fit_doubly_robust_baseline(train_df, spec=spec)
    validation_scores = predict_doubly_robust_scores(validation_df, models=models, spec=spec)
    test_scores = predict_doubly_robust_scores(test_df, models=models, spec=spec)

    validation_metrics = _evaluate_split(
        validation_df,
        split_name="validation",
        spec=spec,
        scores_df=validation_scores,
    )
    test_metrics = _evaluate_split(test_df, split_name="test", spec=spec, scores_df=test_scores)

    test_scored = test_df.copy()
    for column in test_scores.columns:
        test_scored[column] = test_scores[column]
    known_effect_report = known_effect_ranking_report(
        test_scored,
        score_cols=score_cols,
        true_effect_col=true_effect_col,
    )

    run_summary = {
        "dataset_path": str(dataset_path),
        "feature_builder": feature_builder,
        "treatment_col": spec.treatment_col,
        "outcome_col": spec.outcome_col,
        "true_effect_col": true_effect_col,
        "synthetic_seed": seed,
        "numeric_features": spec.numeric_features,
        "categorical_features": spec.categorical_features,
        "row_counts": {
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
        },
    }

    run_dir = make_run_dir(output_dir, run_name=run_name)
    write_yaml(config, run_dir / "config.yaml")
    write_json(run_summary, run_dir / "run_summary.json")
    write_json(validation_metrics, run_dir / "validation_metrics.json")
    write_json(test_metrics, run_dir / "test_metrics.json")
    write_json(known_effect_report, run_dir / "known_effect_ranking.json")
    write_model(models["propensity_model"], run_dir / "propensity_model.joblib")
    write_model(models["treated_outcome_model"], run_dir / "treated_outcome_model.joblib")
    write_model(models["control_outcome_model"], run_dir / "control_outcome_model.joblib")
    write_model(models["tau_model"], run_dir / "doubly_robust_model.joblib")

    manifest = build_run_manifest(
        run_dir=run_dir,
        run_name=run_name,
        pipeline_name="run_semisynthetic_uplift",
        config_path=config_path,
        dataset_path=dataset_path,
        train_rows=len(train_df),
        validation_rows=len(validation_df),
        test_rows=len(test_df),
        metrics=test_metrics,
        validation_metrics=validation_metrics,
        git_commit=current_git_commit(cwd=Path.cwd()),
        artifacts={
            "config": "config.yaml",
            "run_summary": "run_summary.json",
            "validation_metrics": "validation_metrics.json",
            "test_metrics": "test_metrics.json",
            "known_effect_ranking": "known_effect_ranking.json",
            "propensity_model": "propensity_model.joblib",
            "treated_outcome_model": "treated_outcome_model.joblib",
            "control_outcome_model": "control_outcome_model.joblib",
            "doubly_robust_model": "doubly_robust_model.joblib",
        },
        extra_metadata={
            "report_type": "semi_synthetic_uplift_evaluation",
            "method_names": score_cols,
            "true_effect_col": true_effect_col,
        },
    )
    write_json(manifest, run_dir / "manifest.json")

    for score_col in score_cols:
        score_report = known_effect_report[score_col]
        print(
            f"{score_col} Spearman vs true effect: "
            f"{score_report['spearman_corr_with_true_effect']:.6f}"
        )
    print(f"Saved semi-synthetic uplift artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
