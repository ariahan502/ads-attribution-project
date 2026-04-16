from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import yaml


def make_run_dir(base_dir: str | Path, run_name: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = run_name or "run"
    run_dir = Path(base_dir) / f"{timestamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_model(model: Any, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def current_git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def build_run_manifest(
    *,
    run_dir: str | Path,
    run_name: str,
    pipeline_name: str,
    config_path: str | Path,
    dataset_path: str | Path,
    train_rows: int,
    test_rows: int,
    metrics: dict[str, Any],
    git_commit: str | None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir_path = Path(run_dir)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir_path.name,
        "run_name": run_name,
        "pipeline_name": pipeline_name,
        "created_at": datetime.now().isoformat(),
        "git_commit": git_commit,
        "dataset_path": str(dataset_path),
        "row_counts": {
            "train_rows": train_rows,
            "test_rows": test_rows,
        },
        "metrics_summary": metrics,
        "artifacts": {
            "config": "config.yaml",
            "metrics": "metrics.json",
            "evaluation_summary": "evaluation_summary.json",
            "slice_evaluation": "slice_evaluation.json",
            "model": "model.joblib",
        },
        "config_source": str(config_path),
    }
    if extra_metadata:
        manifest["metadata"] = extra_metadata
    return manifest
