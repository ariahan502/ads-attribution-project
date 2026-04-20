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
from ads_project.pipeline.run_uplift import _coerce_str_list, uplift_spec_from_config
from ads_project.policy import policy_simulation_report
from ads_project.uplift import add_semisynthetic_uplift_columns, fit_doubly_robust_baseline, predict_doubly_robust_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline targeting policy simulation from uplift scores.")
    parser.add_argument(
        "--config",
        default="configs/policy_semisynthetic_xgboost_smoke.yaml",
        help="Path to YAML config for policy simulation.",
    )
    return parser.parse_args()


def _coerce_float_list(value: Any, *, field_name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of numbers")
    return [float(item) for item in value]


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
    run_name = str(config.get("run_name", "policy_simulation"))
    max_rows = config.get("max_rows")
    seed = int(config.get("synthetic_seed", 42))
    policy_random_seed = int(config.get("policy_random_seed", 42))
    true_effect_col = str(config.get("true_effect_col", "true_treatment_effect"))
    score_cols = _coerce_str_list(
        config.get("score_cols", ["observational_score", "doubly_robust_score"]),
        field_name="score_cols",
    )
    top_fractions = _coerce_float_list(
        config.get("top_fractions", [0.01, 0.05, 0.1, 0.2, 0.3]),
        field_name="top_fractions",
    )

    print(f"Loading policy simulation data from: {dataset_path}")
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
    test_scores = predict_doubly_robust_scores(test_df, models=models, spec=spec)
    test_scored = test_df.copy()
    for column in test_scores.columns:
        test_scored[column] = test_scores[column]

    report = policy_simulation_report(
        test_scored,
        score_cols=score_cols,
        top_fractions=top_fractions,
        outcome_col=spec.outcome_col,
        treatment_col=spec.treatment_col,
        true_effect_col=true_effect_col,
        random_seed=policy_random_seed,
    )
    policy_frame = pd.DataFrame(report["policies"])

    run_summary = {
        "dataset_path": str(dataset_path),
        "feature_builder": feature_builder,
        "treatment_col": spec.treatment_col,
        "outcome_col": spec.outcome_col,
        "true_effect_col": true_effect_col,
        "synthetic_seed": seed,
        "policy_random_seed": policy_random_seed,
        "score_cols": score_cols,
        "top_fractions": top_fractions,
        "learner_type": spec.learner_type,
        "learner_params": spec.resolved_learner_params,
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
    write_json(report, run_dir / "policy_simulation.json")
    write_csv(policy_frame, run_dir / "policy_simulation.csv")

    manifest = build_run_manifest(
        run_dir=run_dir,
        run_name=run_name,
        pipeline_name="run_policy_simulation",
        config_path=config_path,
        dataset_path=dataset_path,
        train_rows=len(train_df),
        validation_rows=len(validation_df),
        test_rows=len(test_df),
        metrics={"policy_count": len(policy_frame)},
        validation_metrics=None,
        git_commit=current_git_commit(cwd=Path.cwd()),
        artifacts={
            "config": "config.yaml",
            "run_summary": "run_summary.json",
            "policy_simulation": "policy_simulation.json",
            "policy_simulation_csv": "policy_simulation.csv",
        },
        extra_metadata={
            "report_type": "semi_synthetic_policy_simulation",
            "method_names": score_cols,
            "true_effect_col": true_effect_col,
            "caveat": "Semi-synthetic policy results validate ranking mechanics; observational policy results still require causal assumptions.",
        },
    )
    write_json(manifest, run_dir / "manifest.json")

    top_policy = (
        policy_frame[policy_frame["policy_name"] == f"top_{score_cols[-1]}"]
        .sort_values("top_fraction")
        .iloc[0]
    )
    print(
        f"Smallest-budget {score_cols[-1]} policy true-effect lift: "
        f"{top_policy.get('true_effect_lift', float('nan')):.6f}"
    )
    print(f"Saved policy simulation artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
