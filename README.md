# Ads Attribution Project

This repository is a reproducible, config-driven offline ads modeling and decisioning pipeline.

The project supports:

- generate a reproducible parquet sample
- train a split-aware CTR model from config
- compare logistic regression and XGBoost on the same feature set
- generate descriptive attribution summaries from package code
- evaluate uplift rankings with observational and semi-synthetic workflows
- simulate budgeted targeting policies and write row-level batch scoring outputs
- monitor feature and score drift
- inspect structured run bundles under `artifacts/runs/`

This repo is careful about causal claims: CTR and attribution outputs are decision-support signals, while semi-synthetic uplift evaluation validates ranking mechanics against known treatment effects.

For a concise reviewer-facing summary of intended use, strongest results, assumptions, and limitations, see:

- `MODEL_CARD.md`

## Portfolio Snapshot

This project demonstrates an end-to-end ads decisioning workflow:

- predictive modeling for click-through rate ranking
- descriptive attribution for campaign review
- uplift diagnostics with explicit causal caveats
- semi-synthetic treatment-effect validation
- budgeted policy simulation and decision reporting
- deterministic batch scoring outputs
- feature and score drift monitoring
- smoke CI and lightweight pytest coverage

Latest semi-synthetic validation shows the XGBoost doubly robust uplift score recovering the known treatment-effect ranking with Spearman correlation `0.988762`. The 10% policy decision report captures `0.993502` of oracle expected incremental conversions under the controlled benchmark.

## Current Repo Status

What works today:

- deterministic smoke sample generation from a tracked fixture
- GitHub Actions smoke CI for the self-contained fixture path
- deterministic sample generation from the raw local dataset
- reusable CTR feature builders under `src/ads_project/features/`
- train-only campaign CTR encoding with unseen-campaign fallback
- time-aware train/validation/test splits
- config-driven model comparison between logistic regression and XGBoost
- config-driven descriptive attribution reporting for last-touch and linear multi-touch
- config-driven observational uplift baselines with propensity and doubly robust scoring
- uplift policy-curve diagnostics for comparing observational and adjusted rankings across top-k groups
- semi-synthetic uplift evaluation with a known treatment-effect signal
- policy simulation, decision reporting, and deterministic batch scoring
- feature and score drift reporting
- structured run bundles with split counts, validation metrics, test metrics, evaluation summaries, and slice-level evaluation

## Setup

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Commands use the `src/` layout directly:

```bash
PYTHONPATH=src python -m ...
```

## Quality Gate

Run the self-contained smoke gate locally with:

```bash
bash scripts/ci_smoke.sh
```

This is the same command used by GitHub Actions. It runs the lightweight pytest suite, compiles the package, and runs smoke versions of sample generation, CTR training, attribution, semi-synthetic uplift, policy simulation, batch scoring, and drift reporting from the tracked fixture.

## Quickstart

The fastest self-contained path does not require the full raw dataset.

### 1. Build the smoke sample

```bash
PYTHONPATH=src python -m ads_project.pipeline.sample_data --config configs/sample_smoke.yaml
```

This reads from the tracked fixture:

- `data/fixtures/ctr_smoke_seed.parquet`

and writes:

- `data/samples/sample_smoke.parquet`

### 2. Train the split-aware smoke CTR baseline

```bash
PYTHONPATH=src python -m ads_project.pipeline.train_ctr --config configs/ctr_smoke_v2_split.yaml
```

This path uses:

- feature builder: `ctr_notebook_v2`
- train-only encoding: `campaign_ctr`
- split shape: `train=70%`, `validation=10%`, `test=20%`

### 3. Inspect the newest run bundle

Each run writes to:

- `artifacts/runs/<timestamp>_<run_name>/`

For split-aware CTR runs, the most useful files are:

- `config.yaml`
- `run_summary.json`
- `validation_metrics.json`
- `test_metrics.json`
- `evaluation_summary.json`
- `slice_evaluation.json`
- `model.joblib`
- `manifest.json`

## Full Sample Path

If the raw dataset is available locally at:

- `data/raw/criteo_attribution_dataset.tsv.gz`

you can generate the 1M-row working sample with:

```bash
PYTHONPATH=src python -m ads_project.pipeline.sample_data --config configs/sample.yaml
```

This writes:

- `data/samples/sample_1m.parquet`

## Training Paths

### Split-aware baseline training

The current default split-aware baseline config is:

- `configs/ctr_baseline_v2_split.yaml`

Run it with:

```bash
PYTHONPATH=src python -m ads_project.pipeline.train_ctr --config configs/ctr_baseline_v2_split.yaml
```

This trains one model, evaluates on both validation and test, and writes a run bundle.

### Model comparison

To compare logistic regression and XGBoost on the same split-aware baseline:

```bash
PYTHONPATH=src python -m ads_project.pipeline.compare_ctr --config configs/ctr_compare_baseline_v2_split.yaml
```

This creates:

- one run bundle per model
- one comparison bundle containing `comparison.json`

The comparison report currently ranks models by test ROC AUC and also includes validation metrics for each candidate.

## Attribution Path

The repo now includes a first descriptive attribution workflow from package code.

### Smoke attribution run

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_attribution --config configs/attribution_smoke.yaml
```

### Baseline attribution run

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_attribution --config configs/attribution_baseline.yaml
```

These runs currently produce:

- `summary.json`
- `campaign_summary.json`
- `campaign_summary.csv`
- `campaign_decision_report.json`
- `campaign_decision_report.csv`
- `campaign_report.json`
- `manifest.json`

The current schemes are:

- `last_touch`
- `multi_touch_linear`
- `time_decay`

The default time-decay setting is:

- `time_decay_rate: 0.5`

The attribution outputs are descriptive decision-support summaries, not causal estimates.

The current campaign summary includes:

- campaign volume and conversion totals
- total cost
- side-by-side `last_touch`, `multi_touch_linear`, and `time_decay`
- scheme deltas
- simple proxy ROI columns for linear and time-decay attribution

The current decision-facing report adds:

- spend share, click share, and conversion share
- attributed conversion share under time decay
- simple efficiency indices
- heuristic priority buckets such as `scale_candidate`, `review_candidate`, and `monitor`

These prioritization fields are intended as decision-support hints, not automated budget policy. Small-spend campaigns can look unusually efficient, so the report should still be reviewed alongside raw volume.

## Uplift Path

The repo includes a first observational uplift workflow. It is useful for reproducible ranking diagnostics, but it is not a causal identification pipeline.

### Smoke uplift run

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_uplift --config configs/uplift_smoke.yaml
```

### Baseline uplift run

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_uplift --config configs/uplift_baseline.yaml
```

These runs currently produce:

- `run_summary.json`
- `validation_metrics.json`
- `test_metrics.json`
- `propensity_model.joblib`
- `treated_outcome_model.joblib`
- `control_outcome_model.joblib`
- `doubly_robust_model.joblib`
- `manifest.json`

The current uplift diagnostics include:

- propensity model metrics
- top/bottom ranking diagnostics
- policy curves across top-k fractions
- observed conversion lift versus the full split baseline
- treated/control observed outcome gaps within selected groups

Latest baseline validation showed that the simpler observational ranking outperformed the first doubly robust ranking on observed top-k conversion diagnostics. That result should guide the next methodological improvement rather than be presented as causal evidence.

### Semi-synthetic uplift evaluation

The repo also includes a controlled semi-synthetic path that injects synthetic treatment, outcome, and known treatment-effect columns into the existing feature table. This gives the uplift methods a known ranking target.

Smoke run:

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_semisynthetic_uplift --config configs/uplift_semisynthetic_smoke.yaml
```

Baseline run:

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_semisynthetic_uplift --config configs/uplift_semisynthetic_baseline.yaml
```

Rank-feature benchmark run:

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_semisynthetic_uplift --config configs/uplift_semisynthetic_rank_baseline.yaml
```

The rank-feature variant is benchmark-oriented: it adds full-table rank transforms for semi-synthetic evaluation, but it should not be treated as a serving-time-safe production feature set without a separate point-in-time design.

XGBoost rank-feature benchmark run:

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_semisynthetic_uplift --config configs/uplift_semisynthetic_rank_xgboost_baseline.yaml
```

These runs produce `known_effect_ranking.json`, which compares each learned score against the known treatment effect using rank correlations and top-k true-effect lift.

Latest 1M-row validation showed weak recovery of the known effect ranking:

- observational score Spearman correlation with true effect: `-0.164690`
- doubly robust score Spearman correlation with true effect: `0.022254`
- top-decile true-effect lift for observational score: `1.118392`
- top-decile true-effect lift for doubly robust score: `1.121761`
- oracle top-decile true-effect lift: `1.738078`

This is a useful negative result: the evaluation harness is now in place, but the current uplift estimators need improvement before the project should make strong uplift-ranking claims.

Adding semi-synthetic rank features improved recovery slightly but did not solve the problem:

- doubly robust Spearman correlation improved from `0.022254` to `0.067060`
- observational score Spearman correlation improved from `-0.164690` to `-0.149611`
- observational top-decile true-effect lift improved from `1.118392` to `1.138518`
- doubly robust top-decile true-effect lift stayed roughly flat at `1.121761`

This suggests the benchmark is sensitive enough to detect incremental changes, but stronger uplift estimators are still needed.

Adding nonlinear XGBoost learners to the same benchmark produced much stronger recovery:

- observational score Spearman correlation with true effect: `0.977790`
- doubly robust score Spearman correlation with true effect: `0.988762`
- observational top-decile true-effect lift: `1.717863`
- doubly robust top-decile true-effect lift: `1.726784`
- oracle top-decile true-effect lift: `1.738078`

This is evidence that the semi-synthetic signal is learnable with the current feature surface and a stronger estimator. It still should not be read as causal validation of the observational production data.

## Policy Simulation Path

The repo includes a first offline targeting policy simulation workflow on top of the semi-synthetic uplift benchmark.

Smoke run:

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_policy_simulation --config configs/policy_semisynthetic_xgboost_smoke.yaml
```

Baseline run:

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_policy_simulation --config configs/policy_semisynthetic_xgboost_baseline.yaml
```

These runs currently produce:

- `policy_simulation.json`
- `policy_simulation.csv`
- `policy_decision_report.json`
- `policy_decision_report.csv`
- `run_summary.json`
- `manifest.json`

The simulation report compares deterministic random targeting, learned top-k policies, and an oracle true-effect policy across configured budget fractions. The decision report summarizes the preferred policy, a review-ready budget recommendation, oracle capture rate, regret, and caveats.

Latest 1M-row semi-synthetic policy validation showed that the XGBoost doubly robust ranking nearly matched oracle under the known treatment effect:

- 1% budget: DR expected incremental conversions `410.196986` versus oracle `423.218691`
- 10% budget: DR expected incremental conversions `3801.412001` versus oracle `3826.275180`
- 30% budget: DR expected incremental conversions `10026.723471` versus oracle `10058.940772`

The current decision report recommends reviewing the 10% budget policy:

- selected rows: `20000`
- expected incremental conversions: `3801.412001`
- incremental conversions versus random: `1603.589549`
- oracle capture rate: `0.993502`
- oracle regret: `24.863179`

This validates the mechanics of budgeted policy comparison under controlled semi-synthetic outcomes. It is still a decision-support simulation, not proof that the same policy would create causal lift on observational production logs.

## Batch Scoring Path

The repo includes a first deterministic row-level batch scoring workflow for the semi-synthetic XGBoost uplift path.

Smoke run:

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_batch_scoring --config configs/batch_score_semisynthetic_xgboost_smoke.yaml
```

Baseline run:

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_batch_scoring --config configs/batch_score_semisynthetic_xgboost_baseline.yaml
```

These runs currently produce:

- `batch_scores.csv`
- `batch_scores.parquet`
- `batch_score_summary.json`
- `run_summary.json`
- `manifest.json`

The scored output schema includes row identifiers, uplift scores, policy rank, score percentile, configured top-fraction, and a `recommended_policy` flag.

Latest 1M-row batch scoring validation produced:

- scored rows: `200001`
- recommended rows: `20000`
- recommended expected incremental conversions: `3801.412001`
- recommended expected incremental conversions per 1k selected: `190.070600`

As with the policy report, this validates the mechanics of batch scoring under controlled semi-synthetic outcomes. It is not a deployment-ready causal policy for observational production logs.

## Monitoring Path

The repo includes a first feature and score drift workflow for the semi-synthetic XGBoost uplift path.

Smoke run:

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_drift_report --config configs/drift_semisynthetic_xgboost_smoke.yaml
```

Baseline run:

```bash
PYTHONPATH=src python -m ads_project.pipeline.run_drift_report --config configs/drift_semisynthetic_xgboost_baseline.yaml
```

These runs currently produce:

- `drift_summary.json`
- `feature_drift.json`
- `feature_drift.csv`
- `score_drift.json`
- `score_drift.csv`
- `manifest.json`

The report compares the train/reference split against the held-out current/scoring split using numeric and categorical summary statistics plus population stability index (PSI).

Latest 1M-row drift validation produced:

- reference rows: `700000`
- current rows: `200001`
- max feature PSI: `0.007994`
- max score PSI: `0.003635`

## Config Guide

Key configs currently in use:

- `configs/sample_smoke.yaml`
  - smoke sample from tracked fixture
- `configs/sample.yaml`
  - 1M-row local sample from raw dataset
- `configs/ctr_smoke_v2_split.yaml`
  - smoke training on split-aware CTR baseline
- `configs/ctr_baseline_v2_split.yaml`
  - main split-aware CTR baseline on the 1M sample
- `configs/ctr_compare_baseline_v2_split.yaml`
  - split-aware model comparison between logistic regression and XGBoost
- `configs/attribution_smoke.yaml`
  - smoke attribution report
- `configs/attribution_baseline.yaml`
  - baseline attribution report on the 1M sample
- `configs/uplift_smoke.yaml`
  - smoke observational uplift baseline
- `configs/uplift_baseline.yaml`
  - baseline observational uplift run on the 1M sample
- `configs/uplift_semisynthetic_smoke.yaml`
  - smoke semi-synthetic uplift evaluation with known treatment effect
- `configs/uplift_semisynthetic_baseline.yaml`
  - 1M-row semi-synthetic uplift evaluation with known treatment effect
- `configs/uplift_semisynthetic_rank_smoke.yaml`
  - smoke semi-synthetic uplift evaluation with rank-style benchmark features
- `configs/uplift_semisynthetic_rank_baseline.yaml`
  - 1M-row semi-synthetic uplift evaluation with rank-style benchmark features
- `configs/uplift_semisynthetic_rank_xgboost_smoke.yaml`
  - smoke semi-synthetic uplift evaluation with rank-style benchmark features and XGBoost learners
- `configs/uplift_semisynthetic_rank_xgboost_baseline.yaml`
  - 1M-row semi-synthetic uplift evaluation with rank-style benchmark features and XGBoost learners
- `configs/policy_semisynthetic_xgboost_smoke.yaml`
  - smoke policy simulation on semi-synthetic XGBoost uplift scores
- `configs/policy_semisynthetic_xgboost_baseline.yaml`
  - 1M-row policy simulation on semi-synthetic XGBoost uplift scores
- `configs/batch_score_semisynthetic_xgboost_smoke.yaml`
  - smoke row-level batch scoring on semi-synthetic XGBoost uplift scores
- `configs/batch_score_semisynthetic_xgboost_baseline.yaml`
  - 1M-row row-level batch scoring on semi-synthetic XGBoost uplift scores
- `configs/drift_semisynthetic_xgboost_smoke.yaml`
  - smoke feature and score drift report on semi-synthetic XGBoost uplift scores
- `configs/drift_semisynthetic_xgboost_baseline.yaml`
  - 1M-row feature and score drift report on semi-synthetic XGBoost uplift scores

Older configs are still kept for historical comparison and intermediate experiments:

- `configs/ctr_baseline.yaml`
- `configs/ctr_smoke.yaml`
- `configs/ctr_baseline_v2_features.yaml`
- `configs/ctr_compare_baseline.yaml`
- `configs/ctr_compare_baseline_v2_features.yaml`

## Current CTR Pipeline Shape

Current active feature surface in the split-aware baseline:

- numeric:
  - `log_cost`
  - `log_cpo`
  - `time_since_last_click`
  - `has_prev_click`
  - `log_time_since_last_click`
  - `campaign_ctr`
- categorical:
  - `cat1` ... `cat9`
  - `campaign`
  - `recency_bucket`

Current safeguards:

- source schema validation
- source quality validation
- row-wise feature building before split
- time-aware train/validation/test split
- train-only `campaign_ctr` encoding
- separate validation and test metrics in split-aware bundles

See the local note for feature assumptions and leakage risks:

- `doc/feature-ctr-features/point-in-time-notes.md`

## Repository Layout

```text
.
├── configs/
├── scripts/
├── src/
│   └── ads_project/
│       ├── artifacts.py
│       ├── config.py
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       └── pipeline/
├── data/
│   ├── fixtures/
│   ├── raw/
│   └── samples/
├── artifacts/
└── notebooks/
```

## Code Organization

- `src/ads_project/config.py`
  - YAML config loading
- `src/ads_project/data/`
  - parquet IO, sampling, and schema checks
- `src/ads_project/features/`
  - reusable feature builders and train-only encodings
- `src/ads_project/models/`
  - baseline model specs and time-aware split helpers
- `src/ads_project/evaluation/`
  - classification metrics, calibration/lift summaries, slice-level reports
- `src/ads_project/uplift/`
  - observational adjustment baselines for propensity and doubly robust scoring
- `src/ads_project/pipeline/`
  - runnable CLI entrypoints for sampling, training, model comparison, attribution, and uplift

## Notes On Interpretation

- CTR metrics here describe predictive ranking quality, not causal effect.
- Attribution summaries here are descriptive and comparative, not causal.
- The uplift pipeline now includes package-level observational adjustment baselines, but it is still not a causal identification pipeline.
- The current uplift configs use observational `click` as treatment, so scores should be treated as adjusted association-based ranking rather than estimated incrementality.
- The first doubly robust baseline is useful as a reproducible comparison point, but current policy-curve diagnostics do not support replacing the simpler observational ranking on this dataset.
- The semi-synthetic evaluation path gives the project a known-effect benchmark, and current results show that uplift-ranking quality is still weak.

See the local note for the current uplift framing:

- `doc/feature-uplift/observational-method-notes.md`
- `campaign_ctr` is train-only and safe for the current split-aware flow, but fold-aware fitting will be needed if the repo adds cross-validation or hyperparameter search.
- `cpo` and `time_since_last_click` still have documented point-in-time caveats in the local leakage note.

## Legacy Notebooks

Notebooks remain useful for EDA and narrative analysis:

- `notebooks/01_eda.ipynb`
- `notebooks/02_ctr_model.ipynb`

The source of truth for reusable logic is now the config-backed package code under `src/ads_project/`.

## Current Best Next Step

The next highest-value task is improving the uplift estimators or feature set so semi-synthetic ranking recovery improves against the known treatment-effect benchmark.
