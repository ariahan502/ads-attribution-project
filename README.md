# Ads Attribution Project

This repository is an offline ads modeling project that is being moved from notebook-heavy exploration into a reproducible, config-driven pipeline.

The current strongest path in the repo is:

- generate a reproducible parquet sample
- train a split-aware CTR model from config
- compare logistic regression and XGBoost on the same feature set
- generate descriptive attribution summaries from package code
- inspect a structured run bundle under `artifacts/runs/`

This repo does not claim causal uplift from observational click logs. The current CTR pipeline is a reproducible modeling baseline and experiment surface, not proof of incremental ad value.

## Current Repo Status

What works today:

- deterministic smoke sample generation from a tracked fixture
- deterministic sample generation from the raw local dataset
- reusable CTR feature builders under `src/ads_project/features/`
- train-only campaign CTR encoding with unseen-campaign fallback
- time-aware train/validation/test splits
- config-driven model comparison between logistic regression and XGBoost
- config-driven descriptive attribution reporting for last-touch and linear multi-touch
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
- `src/ads_project/pipeline/`
  - runnable CLI entrypoints for sampling, training, and model comparison

## Notes On Interpretation

- CTR metrics here describe predictive ranking quality, not causal effect.
- Attribution summaries here are descriptive and comparative, not causal.
- Uplift and incrementality work are still planned separately.
- The current uplift-style work is still observational notebook analysis, not a package-level causal pipeline.
- Until stronger methods are added, any uplift-style score should be treated as association-based ranking rather than estimated incrementality.

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

The next highest-value task is upgrading the observational uplift framing into stronger methodology with adjustment baselines and uplift-specific evaluation, while keeping repo claims aligned with what the data can actually support today.
