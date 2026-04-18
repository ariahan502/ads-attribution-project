from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from ads_project.artifacts import make_run_dir, write_json, write_yaml
from ads_project.config import load_yaml_config
from ads_project.pipeline.train_ctr import run_ctr_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple CTR models on the same config base.")
    parser.add_argument(
        "--config",
        default="configs/ctr_compare_smoke.yaml",
        help="Path to YAML config for CTR model comparison.",
    )
    return parser.parse_args()


def _coerce_models(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("models must be a non-empty list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("Each model entry must be a mapping")
    return [deepcopy(item) for item in value]


def _metric_or_nan(value: Any) -> float:
    if value is None:
        return float("-inf")
    return float(value)


def _merge_config(base_config: dict[str, Any], model_config: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base_config)
    for key, value in model_config.items():
        if key == "name":
            continue
        merged[key] = deepcopy(value)
    return merged


def main() -> None:
    args = parse_args()
    comparison_config = load_yaml_config(args.config)
    base_config_path = Path(comparison_config["base_config"])
    base_config = load_yaml_config(base_config_path)
    models = _coerce_models(comparison_config["models"])
    output_dir = Path(comparison_config.get("output_dir", "artifacts/runs"))
    run_name = str(comparison_config.get("run_name", "ctr_compare"))

    comparison_run_dir = make_run_dir(output_dir, run_name=run_name)
    write_yaml(comparison_config, comparison_run_dir / "config.yaml")

    results: list[dict[str, Any]] = []
    for model_config in models:
        model_name = str(model_config.get("name", model_config.get("run_name", "model")))
        print(f"Running model comparison candidate: {model_name}")
        merged_config = _merge_config(base_config, model_config)
        training_result = run_ctr_training(
            merged_config,
            config_path=base_config_path,
            config_snapshot=merged_config,
        )
        results.append(
            {
                "name": model_name,
                "run_name": training_result["run_name"],
                "run_dir": str(training_result["run_dir"]),
                "run_summary": training_result["run_summary"],
                "model_type": training_result["model_type"],
                "model_params": training_result["model_params"],
                "validation_metrics": training_result["validation_metrics"],
                "test_metrics": training_result["metrics"],
                "evaluation_summary": {
                    "baseline_positive_rate": training_result["evaluation_summary"]["baseline_positive_rate"],
                    "calibration_bin_mae": training_result["evaluation_summary"]["calibration_bin_mae"],
                    "top_bin_lift": training_result["evaluation_summary"]["top_bin_lift"],
                    "bottom_bin_lift": training_result["evaluation_summary"]["bottom_bin_lift"],
                },
            }
        )

    rankings = {
        "test_roc_auc_desc": sorted(
            results, key=lambda row: _metric_or_nan(row["test_metrics"]["roc_auc"]), reverse=True
        ),
        "test_pr_auc_desc": sorted(
            results, key=lambda row: _metric_or_nan(row["test_metrics"]["pr_auc"]), reverse=True
        ),
        "test_log_loss_asc": sorted(results, key=lambda row: float(row["test_metrics"]["log_loss"])),
        "test_brier_score_asc": sorted(results, key=lambda row: float(row["test_metrics"]["brier_score"])),
    }

    best_model = rankings["test_roc_auc_desc"][0]
    comparison_report = {
        "comparison_run_id": comparison_run_dir.name,
        "base_config": str(base_config_path),
        "results": results,
        "best_by_test_roc_auc": {
            "name": best_model["name"],
            "run_dir": best_model["run_dir"],
            "roc_auc": best_model["test_metrics"]["roc_auc"],
            "pr_auc": best_model["test_metrics"]["pr_auc"],
            "log_loss": best_model["test_metrics"]["log_loss"],
            "brier_score": best_model["test_metrics"]["brier_score"],
        },
        "rankings": rankings,
    }
    write_json(comparison_report, comparison_run_dir / "comparison.json")

    print(f"Saved comparison report to: {comparison_run_dir / 'comparison.json'}")
    print(f"Best test ROC AUC model: {best_model['name']} ({best_model['test_metrics']['roc_auc']:.6f})")


if __name__ == "__main__":
    main()
