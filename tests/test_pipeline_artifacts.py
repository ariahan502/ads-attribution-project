from __future__ import annotations

import json

from ads_project.pipeline.run_calibration_drift import run_calibration_drift
from ads_project.pipeline.train_ctr import run_ctr_training


def test_ctr_training_writes_expected_artifact_bundle(tmp_path) -> None:
    config = {
        "dataset_path": "data/fixtures/ctr_smoke_seed.parquet",
        "timestamp_col": "timestamp",
        "label": "click",
        "feature_builder": "ctr_notebook_v2",
        "train_only_encodings": ["campaign_ctr"],
        "model_type": "logistic_regression",
        "model_params": {"max_iter": 100},
        "numeric_features": [
            "log_cost",
            "log_cpo",
            "time_since_last_click",
            "has_prev_click",
            "log_time_since_last_click",
            "campaign_ctr",
        ],
        "categorical_features": [
            "cat1",
            "cat2",
            "cat3",
            "cat4",
            "cat5",
            "cat6",
            "cat7",
            "cat8",
            "cat9",
            "campaign",
            "recency_bucket",
        ],
        "train_fraction": 0.7,
        "validation_fraction": 0.1,
        "evaluation_top_campaigns": 3,
        "evaluation_time_slices": 3,
        "output_dir": str(tmp_path),
        "run_name": "test_ctr",
    }

    result = run_ctr_training(config, config_path="test_ctr.yaml")

    run_dir = result["run_dir"]
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "validation_metrics.json").exists()
    assert (run_dir / "test_metrics.json").exists()
    assert (run_dir / "slice_evaluation.json").exists()
    assert (run_dir / "model.joblib").exists()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["pipeline_name"] == "train_ctr"
    assert manifest["artifacts"]["model"] == "model.joblib"


def test_calibration_drift_pipeline_writes_expected_artifacts(tmp_path) -> None:
    config = {
        "dataset_path": "data/fixtures/ctr_smoke_seed.parquet",
        "timestamp_col": "timestamp",
        "label": "click",
        "feature_builder": "ctr_notebook_v2",
        "train_only_encodings": ["campaign_ctr"],
        "model_type": "logistic_regression",
        "model_params": {"max_iter": 100},
        "numeric_features": [
            "log_cost",
            "log_cpo",
            "time_since_last_click",
            "has_prev_click",
            "log_time_since_last_click",
            "campaign_ctr",
        ],
        "categorical_features": [
            "cat1",
            "cat2",
            "cat3",
            "cat4",
            "cat5",
            "cat6",
            "cat7",
            "cat8",
            "cat9",
            "campaign",
            "recency_bucket",
        ],
        "train_fraction": 0.7,
        "validation_fraction": 0.1,
        "calibration_bins": 5,
        "output_dir": str(tmp_path),
        "run_name": "test_calibration_drift",
    }

    result = run_calibration_drift(config, config_path="test_calibration_drift.yaml")

    run_dir = result["run_dir"]
    assert (run_dir / "calibration_drift.json").exists()
    assert (run_dir / "calibration_drift_summary.json").exists()
    assert (run_dir / "calibration_drift.csv").exists()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["pipeline_name"] == "run_calibration_drift"
    assert manifest["metadata"]["report_type"] == "ctr_calibration_drift"
