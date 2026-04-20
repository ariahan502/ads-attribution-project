from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ads_project.artifacts import build_run_manifest, current_git_commit, make_run_dir, write_csv, write_json, write_yaml
from ads_project.config import load_yaml_config
from ads_project.data.io import read_parquet, write_parquet
from ads_project.data.schema import validate_baseline_source_quality, validate_baseline_source_schema
from ads_project.features import apply_feature_builder
from ads_project.models.splits import time_ordered_train_validation_test_split
from ads_project.pipeline.run_policy_simulation import _coerce_float_list
from ads_project.pipeline.run_uplift import _coerce_str_list, uplift_spec_from_config
from ads_project.policy import batch_score_summary, build_batch_score_output
from ads_project.uplift import add_semisynthetic_uplift_columns, fit_doubly_robust_baseline, predict_doubly_robust_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic row-level batch scoring outputs.")
    parser.add_argument(
        "--config",
        default="configs/batch_score_semisynthetic_xgboost_smoke.yaml",
        help="Path to YAML config for batch scoring.",
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
    run_name = str(config.get("run_name", "batch_scoring"))
    max_rows = config.get("max_rows")
    seed = int(config.get("synthetic_seed", 42))
    true_effect_col = str(config.get("true_effect_col", "true_treatment_effect"))
    score_cols = _coerce_str_list(
        config.get("score_cols", ["observational_score", "doubly_robust_score"]),
        field_name="score_cols",
    )
    id_columns = _coerce_str_list(
        config.get("id_columns", ["uid", "timestamp", "campaign"]),
        field_name="id_columns",
    )
    optional_output_columns = _coerce_str_list(
        config.get("optional_output_columns", []),
        field_name="optional_output_columns",
    )
    top_fractions = _coerce_float_list(
        config.get("top_fractions", [float(config.get("recommended_top_fraction", 0.1))]),
        field_name="top_fractions",
    )
    preferred_score_col = str(config.get("preferred_score_col", score_cols[-1]))
    recommended_top_fraction = float(config.get("recommended_top_fraction", top_fractions[0]))

    print(f"Loading batch scoring data from: {dataset_path}")
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

    train_df, validation_df, scoring_df = time_ordered_train_validation_test_split(
        df,
        timestamp_col=timestamp_col,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    print(
        f"Train rows: {len(train_df)} | Validation rows: {len(validation_df)} | Scoring rows: {len(scoring_df)}"
    )

    models = fit_doubly_robust_baseline(train_df, spec=spec)
    scores = predict_doubly_robust_scores(scoring_df, models=models, spec=spec)
    scored_output = build_batch_score_output(
        scoring_df,
        score_df=scores,
        id_columns=id_columns,
        score_cols=score_cols,
        preferred_score_col=preferred_score_col,
        recommended_top_fraction=recommended_top_fraction,
        optional_columns=optional_output_columns,
    )
    summary = batch_score_summary(
        scored_output,
        preferred_score_col=preferred_score_col,
        recommended_top_fraction=recommended_top_fraction,
        true_effect_col=true_effect_col,
    )

    output_schema: dict[str, str] = {
        column: str(dtype) for column, dtype in scored_output.dtypes.items()
    }
    run_summary = {
        "dataset_path": str(dataset_path),
        "feature_builder": feature_builder,
        "synthetic_seed": seed,
        "score_cols": score_cols,
        "preferred_score_col": preferred_score_col,
        "recommended_top_fraction": recommended_top_fraction,
        "id_columns": id_columns,
        "optional_output_columns": optional_output_columns,
        "learner_type": spec.learner_type,
        "learner_params": spec.resolved_learner_params,
        "row_counts": {
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "scoring_rows": len(scoring_df),
        },
        "output_schema": output_schema,
    }

    run_dir = make_run_dir(output_dir, run_name=run_name)
    write_yaml(config, run_dir / "config.yaml")
    write_json(run_summary, run_dir / "run_summary.json")
    write_json(summary, run_dir / "batch_score_summary.json")
    write_csv(scored_output, run_dir / "batch_scores.csv")
    write_parquet(scored_output, run_dir / "batch_scores.parquet")

    manifest = build_run_manifest(
        run_dir=run_dir,
        run_name=run_name,
        pipeline_name="run_batch_scoring",
        config_path=config_path,
        dataset_path=dataset_path,
        train_rows=len(train_df),
        validation_rows=len(validation_df),
        test_rows=len(scoring_df),
        metrics=summary,
        validation_metrics=None,
        git_commit=current_git_commit(cwd=Path.cwd()),
        artifacts={
            "config": "config.yaml",
            "run_summary": "run_summary.json",
            "batch_score_summary": "batch_score_summary.json",
            "batch_scores_csv": "batch_scores.csv",
            "batch_scores_parquet": "batch_scores.parquet",
        },
        extra_metadata={
            "report_type": "semi_synthetic_batch_scoring",
            "preferred_score_col": preferred_score_col,
            "recommended_top_fraction": recommended_top_fraction,
            "caveat": "Semi-synthetic scores validate batch output mechanics; observational deployment still requires causal assumptions or experiments.",
        },
    )
    write_json(manifest, run_dir / "manifest.json")

    print(f"Recommended rows: {summary['recommended_rows']} / {summary['rows']}")
    print(
        "Recommended expected incremental conversions: "
        f"{summary.get('recommended_expected_incremental_conversions', float('nan')):.6f}"
    )
    print(f"Saved batch scoring artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
