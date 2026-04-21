#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

python -m compileall -q src scripts
python -m pytest -q

python -m ads_project.pipeline.sample_data --config configs/sample_smoke.yaml
python -m ads_project.pipeline.train_ctr --config configs/ctr_smoke_v2_split.yaml
python -m ads_project.pipeline.run_attribution --config configs/attribution_smoke.yaml
python -m ads_project.pipeline.run_semisynthetic_uplift --config configs/uplift_semisynthetic_rank_xgboost_smoke.yaml
python -m ads_project.pipeline.run_policy_simulation --config configs/policy_semisynthetic_xgboost_smoke.yaml
python -m ads_project.pipeline.run_batch_scoring --config configs/batch_score_semisynthetic_xgboost_smoke.yaml
