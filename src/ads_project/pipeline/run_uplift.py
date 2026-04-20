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
from ads_project.uplift import (
    UpliftSpec,
    fit_doubly_robust_baseline,
    policy_curve_diagnostics,
    predict_doubly_robust_scores,
    ranking_diagnostics,
)
from ads_project.uplift.baselines import propensity_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run observational uplift baselines from config.")
    parser.add_argument(
        "--config",
        default="configs/uplift_smoke.yaml",
        help="Path to YAML config for uplift baselines.",
    )
    return parser.parse_args()


def _coerce_str_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return value


def uplift_spec_from_config(config: dict[str, Any]) -> UpliftSpec:
    return UpliftSpec(
        treatment_col=str(config["treatment_col"]),
        outcome_col=str(config["outcome_col"]),
        numeric_features=_coerce_str_list(config["numeric_features"], field_name="numeric_features"),
        categorical_features=_coerce_str_list(config["categorical_features"], field_name="categorical_features"),
        propensity_clip=float(config.get("propensity_clip", 0.05)),
        max_iter=int(config.get("max_iter", 100)),
        ridge_alpha=float(config.get("ridge_alpha", 1.0)),
        learner_type=str(config.get("learner_type", "linear")),
        learner_params=dict(config.get("learner_params") or {}),
    )


def _evaluate_split(df, *, split_name: str, spec: UpliftSpec, scores_df):
    enriched = df.copy()
    for column in scores_df.columns:
        enriched[column] = scores_df[column]

    score_correlation = enriched[["observational_score", "doubly_robust_score"]].corr().iloc[0, 1]

    return {
        "split": split_name,
        "rows": int(len(df)),
        "treatment_rate": float(df[spec.treatment_col].mean()),
        "outcome_rate": float(df[spec.outcome_col].mean()),
        "propensity_metrics": propensity_metrics(df[spec.treatment_col], enriched["propensity_score"]),
        "observational_score_diagnostics": ranking_diagnostics(
            enriched,
            score_col="observational_score",
            treatment_col=spec.treatment_col,
            outcome_col=spec.outcome_col,
        ),
        "doubly_robust_score_diagnostics": ranking_diagnostics(
            enriched,
            score_col="doubly_robust_score",
            treatment_col=spec.treatment_col,
            outcome_col=spec.outcome_col,
        ),
        "observational_policy_curve": policy_curve_diagnostics(
            enriched,
            score_col="observational_score",
            treatment_col=spec.treatment_col,
            outcome_col=spec.outcome_col,
        ),
        "doubly_robust_policy_curve": policy_curve_diagnostics(
            enriched,
            score_col="doubly_robust_score",
            treatment_col=spec.treatment_col,
            outcome_col=spec.outcome_col,
        ),
        "score_correlation": None if score_correlation != score_correlation else float(score_correlation),
    }


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)
    spec = uplift_spec_from_config(config)

    dataset_path = Path(config["dataset_path"])
    feature_builder = config.get("feature_builder")
    timestamp_col = str(config.get("timestamp_col", "timestamp"))
    train_fraction = float(config.get("train_fraction", 0.7))
    validation_fraction = float(config.get("validation_fraction", 0.1))
    output_dir = Path(config.get("output_dir", "artifacts/runs"))
    run_name = str(config.get("run_name", "uplift"))
    max_rows = config.get("max_rows")

    print(f"Loading uplift data from: {dataset_path}")
    df = read_parquet(dataset_path)
    if max_rows is not None:
        df = df.iloc[: int(max_rows)].copy()
        print(f"Using max_rows subset: {len(df)}")

    validate_baseline_source_schema(df)
    validate_baseline_source_quality(df)
    df = apply_feature_builder(df, builder_name=feature_builder)
    if feature_builder not in (None, "none"):
        print(f"Applied feature builder: {feature_builder}")

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
    test_metrics = _evaluate_split(
        test_df,
        split_name="test",
        spec=spec,
        scores_df=test_scores,
    )

    run_summary = {
        "dataset_path": str(dataset_path),
        "feature_builder": feature_builder,
        "treatment_col": spec.treatment_col,
        "outcome_col": spec.outcome_col,
        "numeric_features": spec.numeric_features,
        "categorical_features": spec.categorical_features,
        "propensity_clip": spec.propensity_clip,
        "learner_type": spec.learner_type,
        "learner_params": spec.resolved_learner_params,
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
    write_model(models["propensity_model"], run_dir / "propensity_model.joblib")
    write_model(models["treated_outcome_model"], run_dir / "treated_outcome_model.joblib")
    write_model(models["control_outcome_model"], run_dir / "control_outcome_model.joblib")
    write_model(models["tau_model"], run_dir / "doubly_robust_model.joblib")

    manifest = build_run_manifest(
        run_dir=run_dir,
        run_name=run_name,
        pipeline_name="run_uplift",
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
            "propensity_model": "propensity_model.joblib",
            "treated_outcome_model": "treated_outcome_model.joblib",
            "control_outcome_model": "control_outcome_model.joblib",
            "doubly_robust_model": "doubly_robust_model.joblib",
        },
        extra_metadata={
            "report_type": "observational_adjusted_uplift_baseline",
            "method_names": ["observational_two_model", "doubly_robust_learner"],
        },
    )
    write_json(manifest, run_dir / "manifest.json")

    print(
        "Validation propensity ROC AUC: "
        f"{_format_metric(validation_metrics['propensity_metrics']['roc_auc'])}"
    )
    print(f"Test propensity ROC AUC: {_format_metric(test_metrics['propensity_metrics']['roc_auc'])}")
    print(
        "Test top-decile observed conversion rate "
        f"(observational score): {test_metrics['observational_score_diagnostics']['top_observed_conversion_rate']:.6f}"
    )
    print(
        "Test top-decile observed conversion rate "
        f"(doubly robust score): {test_metrics['doubly_robust_score_diagnostics']['top_observed_conversion_rate']:.6f}"
    )
    print(f"Saved uplift artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
