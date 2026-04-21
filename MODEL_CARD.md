# Ads Attribution Project Model Card

## Summary

This project is an offline ads decisioning pipeline built from event-level ad logs. It turns a notebook-style prototype into a reproducible package with config-driven commands for data preparation, CTR modeling, attribution reporting, uplift diagnostics, policy simulation, batch scoring, artifact tracking, and smoke CI.

The strongest current story is:

- prepare a reproducible sample from a tracked smoke fixture or local raw data
- train split-aware CTR baselines with reusable feature builders
- compare logistic regression and XGBoost CTR models
- produce descriptive attribution reports by campaign
- evaluate uplift methods carefully without overclaiming causality
- validate uplift ranking on a semi-synthetic benchmark with known treatment effects
- translate uplift scores into policy simulation, decision reports, and batch scoring outputs

## Intended Use

Use this repo as an offline analytics and modeling portfolio project for ads, growth, and marketing data science workflows.

Appropriate uses:

- demonstrating production-style project structure for data science work
- running reproducible local experiments from YAML configs
- comparing CTR models, attribution rules, and uplift-ranking methods
- generating decision-support reports and row-level scoring outputs
- discussing causal limitations and experimental next steps

Inappropriate uses:

- claiming causal lift from observational ad logs alone
- deploying the semi-synthetic rank features directly as serving-time production features
- automating budget allocation without business review or experiment validation

## Data

The full raw Criteo attribution dataset is expected locally at:

- `data/raw/criteo_attribution_dataset.tsv.gz`

The repo also includes a tiny tracked fixture for smoke tests:

- `data/fixtures/ctr_smoke_seed.parquet`

Large generated samples and run artifacts are intentionally kept out of Git:

- `data/samples/`
- `artifacts/runs/`

## Modeling And Evaluation

CTR modeling includes:

- time-aware train/validation/test splits
- reusable feature builders
- train-only campaign CTR encoding with unseen-campaign fallback
- logistic regression and XGBoost model comparison
- ROC AUC, PR AUC, log loss, Brier score, calibration, lift, and slice-level reports

Attribution reporting includes:

- last-touch attribution
- linear multi-touch attribution
- time-decay attribution
- campaign-level decision-support summaries

Uplift modeling includes:

- observational two-model and doubly robust baselines
- propensity diagnostics and top-k policy curves
- a semi-synthetic benchmark with known treatment effects
- configurable linear and XGBoost uplift learners

## Current Strongest Results

On the 1M-row semi-synthetic uplift benchmark, XGBoost learners with rank-style benchmark features recover the known treatment-effect ordering strongly:

- observational score Spearman correlation with true effect: `0.977790`
- doubly robust score Spearman correlation with true effect: `0.988762`
- doubly robust top-decile true-effect lift: `1.726784`
- oracle top-decile true-effect lift: `1.738078`

The policy simulation and batch scoring layers use the doubly robust score as the preferred policy score. The latest 1M-row decision report recommends reviewing the top 10% policy:

- selected rows: `20000`
- expected incremental conversions: `3801.412001`
- incremental conversions versus random: `1603.589549`
- oracle capture rate: `0.993502`
- oracle regret: `24.863179`

These results validate the benchmark and decisioning mechanics. They do not prove real-world causal lift on observational logs.

## Reproducibility

Run the self-contained smoke gate:

```bash
bash scripts/ci_smoke.sh
```

The smoke gate runs the lightweight pytest suite, compiles the package, and runs:

- fixture-backed sample generation
- split-aware CTR training
- attribution reporting
- semi-synthetic XGBoost uplift evaluation
- policy simulation
- batch scoring
- feature and score drift reporting

Each pipeline writes a run bundle under:

- `artifacts/runs/<timestamp>_<run_name>/`

Run bundles include config snapshots, metrics or reports, manifests, and relevant model or scoring artifacts.

## Limitations

The project deliberately separates predictive, descriptive, and causal claims:

- CTR models predict clicks; they do not estimate incremental ad value.
- Attribution reports are descriptive decision-support views; they are not causal estimates.
- Observational uplift diagnostics are useful for ranking analysis, but they depend on strong causal assumptions.
- Semi-synthetic uplift results show that the pipeline can recover a known injected signal; they do not guarantee real-world incrementality.
- Rank-style semi-synthetic features are benchmark-oriented and need a separate point-in-time design before any production interpretation.

## Recommended Next Steps

- Add a short portfolio/business summary for non-technical readers.
- Add calibration drift reports for predictive model calibration.
- Add an observational policy report with explicit assumptions and caveats.
- Optionally add a lightweight local scoring API after the batch path remains stable.
