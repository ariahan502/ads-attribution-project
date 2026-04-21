# Ads Attribution Project Roadmap

## Goal

Build and maintain a reproducible offline ads decisioning pipeline for event-level ad logs.

The project supports data preparation, CTR modeling, attribution reporting, uplift diagnostics, semi-synthetic treatment-effect validation, policy simulation, batch scoring, monitoring, and CI-backed quality checks.

## Principles

- Keep predictive modeling, descriptive attribution, causal analysis, and decisioning clearly separated.
- Prefer reproducible, config-driven workflows over one-off analysis.
- Save run artifacts with enough metadata to compare results across runs.
- Treat observational uplift results as decision support unless validated by experiment or stronger identification assumptions.
- Keep smoke paths self-contained so the project can be validated without the full raw dataset.

## Current Capabilities

- Fixture-backed smoke sample generation
- Full local sample generation from raw data
- Split-aware CTR training and model comparison
- Data schema and quality validation
- Segment, calibration, lift, and slice-level CTR evaluation
- Descriptive attribution reports with last-touch, linear, and time-decay schemes
- Observational uplift diagnostics with propensity and doubly robust baselines
- Semi-synthetic uplift evaluation with known treatment effects
- XGBoost uplift learners for nonlinear benchmark recovery
- Budgeted policy simulation and decision reports
- Deterministic row-level batch scoring outputs
- Feature, score, and calibration drift reports
- Lightweight pytest coverage and GitHub Actions smoke CI

## Near-Term Roadmap

### 1. Local Quality Tooling

Keep the local quality command and optional pre-commit hook aligned with CI as the smoke surface grows.

Expected outputs:

- documented local quality command
- maintained `.pre-commit-config.yaml`

Validation:

- local quality command passes from a clean checkout
- CI continues to use the same or stricter checks

### 2. Observational Policy Report

Extend decision reporting to observational data with explicit assumptions and caveats.

Expected outputs:

- observational policy report artifacts
- clear labeling that observational results are not causal proof

Validation:

- report runs from config
- output schema matches the semi-synthetic policy report where practical

### 3. Optional Local Scoring API

Add a lightweight local API only after the batch scoring path remains stable.

Expected outputs:

- minimal FastAPI app
- documented local run command
- simple request/response schema

Validation:

- API smoke check returns scores for a small fixture input
- batch scoring remains the source of truth for offline workflows

## Operating Notes

- Large raw data, generated samples, and run artifacts stay out of Git.
- `data/fixtures/ctr_smoke_seed.parquet` is the tracked smoke input.
- `scripts/ci_smoke.sh` is the local quality gate used by GitHub Actions.
- Run bundles are written under `artifacts/runs/<timestamp>_<run_name>/`.
