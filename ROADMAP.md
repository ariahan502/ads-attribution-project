# Ads Attribution Project Roadmap

## Goal

Turn this repository from a promising notebook-based prototype into a production-style offline ads decisioning project that is strong enough to showcase for growth and ads data science roles.

The target story for the repo should become:

"Given ad-event logs, build a reproducible pipeline that prepares data, trains and evaluates models, compares attribution methods, estimates incremental impact carefully, and translates results into campaign or user-level targeting decisions."

## Guiding Principles

- Prioritize additions that most improve credibility with hiring managers.
- Keep each increment independently valuable and independently verifiable.
- Separate descriptive analysis, predictive modeling, causal estimation, and decisioning.
- Prefer production-style structure over bigger-but-fragile notebook scope.
- Document assumptions and limitations clearly, especially around causality.

## Priority Order

1. Reproducibility and structure
2. Stronger model evaluation
3. Data validation and feature pipeline
4. Attribution as a decision support layer
5. Better uplift and causal methodology
6. Policy simulation and budgeted targeting
7. Batch scoring interfaces
8. Monitoring, CI, and communication polish

## Phase 1: Foundation and Reproducibility

### Why this matters

This is the highest-leverage step. It changes the repo from "interesting analysis" to "engineered project."

### Scope

- Convert notebook logic into a package-style layout under `src/ads_project/`:
  - `src/ads_project/data/`
  - `src/ads_project/features/`
  - `src/ads_project/models/`
  - `src/ads_project/evaluation/`
  - `src/ads_project/attribution/`
  - `src/ads_project/uplift/`
  - `src/ads_project/pipeline/`
- Keep notebooks only for EDA, interpretation, and reporting.
- Add config files under `configs/`.
- Add deterministic CLI entrypoints such as:
  - `prepare_data`
  - `train_ctr`
  - `evaluate_ctr`
  - `run_attribution`
  - `run_uplift`
- Fix file paths and align README, code, and actual repo structure.
- Add a lightweight environment and run guide.

### Data strategy

- Do not commit the full raw dataset to Git.
- Add one reproducible path for inputs:
  - a documented download script for the original dataset, or
  - a deterministic sample-generation command, or
  - a tiny committed fixture for smoke tests
- Make the smoke-test path work without requiring the full 623 MB raw file.

### Validation

- A fresh user can set up the environment and reproduce baseline outputs from the command line.
- A smoke test runs successfully on a small sample or committed fixture.
- Metrics and artifacts are written to a predictable output directory.

### What it brings

- Strong immediate signal of engineering maturity.
- Easier future iteration on every later phase.

## Phase 2: Experiment Tracking and Artifact Management

### Why this matters

Production-ready DS work needs reproducible experiments, not just notebook outputs.

### Scope

- Add config-driven runs with saved config snapshots.
- Save:
  - trained model artifact
  - feature list
  - metrics JSON
  - evaluation plots
  - run metadata
- Use MLflow if desired, or a simpler structured local tracking directory.
- Version outputs by timestamp or run ID.

### Validation

- Two runs with different configs produce separate, comparable result bundles.
- A reader can tell exactly which parameters produced a given result.

### What it brings

- Better reproducibility.
- More credible experimentation workflow.

## Phase 3: Stronger CTR Modeling and Evaluation

### Why this matters

For ads DS, AUC alone is not enough. Calibration and ranking quality matter a lot.

### Scope

- Add train, validation, and test splitting instead of only train and test.
- Compare:
  - logistic regression baseline
  - XGBoost or LightGBM tuned model
  - one additional tree model only if it adds signal
- Add metrics:
  - ROC AUC
  - PR AUC
  - log loss
  - calibration error or calibration plots
  - lift and gain by decile
- Evaluate by time slice and campaign segment.
- Add feature importance and simple ablation analysis.

### Validation

- A single report compares models on all metrics.
- Calibration and lift plots are generated automatically.
- Segment-level metrics are reproducible across reruns.

### What it brings

- Much stronger modeling signal for ads ranking use cases.

## Phase 4: Data Validation and Feature Pipeline

### Why this matters

This is one of the clearest differences between a project and a production-style pipeline.

### Scope

- Define an explicit input schema.
- Add checks for:
  - nulls
  - duplicates
  - invalid ranges
  - unexpected categories
  - row-count sanity
- Encapsulate feature engineering into reusable functions or transformers.
- Document point-in-time availability for every feature.
- Remove or redesign any leakage-prone transformations.

### Validation

- Unit tests cover core feature builders.
- Data checks fail loudly on malformed input.
- A schema report is produced for each run.

### What it brings

- Better reliability.
- Better answers to interview questions about leakage and inference-time correctness.

## Phase 5: Attribution Layer as Business Decision Support

### Why this matters

Attribution is useful in ads projects when presented carefully and tied to decisions.

### Scope

- Keep and formalize:
  - last-touch attribution
  - linear multi-touch attribution
- Add:
  - time-decay attribution
  - campaign-level attribution summaries
  - attribution comparison tables by campaign
- Build a campaign decision view with:
  - spend
  - clicks
  - conversions
  - attributed conversions
  - proxy ROI
- Explicitly label attribution as descriptive, not causal.

### Validation

- Reproducible campaign reports are generated under each attribution scheme.
- Differences across methods are stable and explainable across date slices.

### What it brings

- Better business framing.
- Better credibility with marketing and growth audiences.

## Phase 6: Better Uplift and Causal Methodology

### Why this matters

This is the most conceptually valuable area, but also the easiest to overclaim. It should be upgraded carefully.

### Scope

- Rename the current method to something more precise if treatment is observational.
- Document the causal assumptions and limitations.
- Add more rigorous methods such as:
  - propensity score estimation
  - doubly robust estimation
  - DR-learner or X-learner
- If possible, build a semi-synthetic evaluation setup where treatment effect is partly known.
- Add uplift-specific evaluation:
  - uplift curves
  - Qini-style comparisons
  - top-k treatment policy comparisons

### Validation

- On semi-synthetic data, the method recovers known effect ordering reasonably well.
- On observational data, uplift ranking is evaluated with uplift-aware metrics rather than only raw conversion rate comparisons.
- Assumptions and limitations are documented in the report.

### What it brings

- High-value differentiation for growth and ads DS roles.
- Better interview defensibility around incrementality.

## Phase 7: Policy Simulation and Budgeted Decisioning

### Why this matters

This is where the project moves from prediction to decision optimization.

### Scope

- Define ranking policies:
  - rank by predicted CTR
  - rank by expected value
  - rank by estimated uplift
- Add a budget constraint or serving cap.
- Simulate offline targeting decisions and compare policy outcomes.
- Produce user-level and campaign-level allocation summaries.

### Validation

- The pipeline can compare policies under the same budget.
- Each policy produces measurable tradeoffs in conversions, value, and estimated incrementality.
- Results are summarized in a single decision report.

### What it brings

- Strong direct relevance to growth and ads decision systems.

## Phase 8: Batch Scoring and Delivery Interfaces

### Why this matters

Many real DS systems are batch-first. For this project, a strong batch pipeline matters more than a heavy service layer.

### Scope

- Add a batch scoring job that writes scored outputs to partitioned files.
- Define output contracts for scored user or campaign tables.
- Optionally add a very small local scoring API only after the offline pipeline is stable.
- Keep any service layer intentionally lightweight and secondary to reproducibility, evaluation, and decisioning.

### Validation

- Batch scoring produces deterministic outputs with expected schema.
- Integration or smoke tests verify output contracts.
- Any optional local service runs through documented commands.

### What it brings

- Practical production-readiness signal without overbuilding.

## Phase 9: Monitoring, Drift, and Governance

### Why this matters

Monitoring is a strong sign that the author understands the full lifecycle, not just model fitting.

### Scope

- Add feature drift checks.
- Add score distribution monitoring.
- Add calibration drift checks across time windows.
- Version:
  - model artifacts
  - configs
  - training windows
  - feature schemas

### Validation

- Drifted input data triggers warnings or visible monitoring differences.
- A recurring monitoring report can be generated from historical runs.

### What it brings

- Clear production lifecycle awareness.

## Phase 10: CI, Docs, and Presentation Polish

### Why this matters

This is the final layer that makes the project easy to trust and easy to evaluate quickly.

### Scope

- Add GitHub Actions for:
  - linting
  - unit tests
  - smoke pipeline run
- Add `pre-commit`.
- Rewrite the README with:
  - clean setup
  - architecture summary
  - assumptions
  - example outputs
  - honest causal framing
- Add a short model card and a business-facing summary report.

### Validation

- CI passes on every push.
- A reviewer can understand the whole project from the README and reports without opening notebooks first.

### What it brings

- Better first impression.
- Better recruiter and hiring-manager accessibility.

## Recommended Milestones

### Milestone 1: Make It Reproducible

Deliverables:

- package-style repo structure under `src/ads_project/`
- configs
- CLI entrypoints
- fixed paths and cleaned README
- data access path for reproducible smoke tests

Success criteria:

- end-to-end rerun works from a fresh environment on a small sample or committed fixture

### Milestone 2: Make The Modeling Credible

Deliverables:

- stronger CTR evaluation
- saved artifacts
- data validation
- feature pipeline tests

Success criteria:

- model comparison report is reproducible and stronger than notebook-only output

### Milestone 3: Make The Business Story Strong

Deliverables:

- attribution comparison framework
- campaign decision tables
- policy simulation with budget constraints

Success criteria:

- repo can demonstrate how different ranking and attribution choices change decisions

### Milestone 4: Make The Incrementality Story Defensible

Deliverables:

- better uplift framing
- stronger causal method
- uplift-specific evaluation

Success criteria:

- an interviewer can challenge the causal assumptions and the repo still holds up

### Milestone 5: Make It Production-Style

Deliverables:

- batch scoring
- optional local scoring API
- monitoring
- CI and docs polish

Success criteria:

- project feels like a realistic offline decisioning system rather than a notebook demo

## Suggested Execution Order

If time is limited, build in this order:

1. Foundation and reproducibility
2. CTR evaluation and artifact tracking
3. Data validation and feature engineering pipeline
4. Attribution reporting
5. Causal and uplift upgrades
6. Policy simulation
7. Batch scoring
8. Monitoring and CI polish

## What To Be Careful About

- Do not overclaim causality from observational click logs.
- Do not add system components before the offline evaluation is solid.
- Do not keep expanding notebook scope when core pipeline structure is still weak.
- Do not optimize for fancy tooling ahead of reproducibility and validation.

## Resume and Portfolio Outcome

If the roadmap is executed well, the project can evolve from:

- "notebook project with good ideas"

to:

- "production-style ads decisioning pipeline with reproducible modeling, attribution analysis, incrementality-aware targeting, and deployable scoring interfaces"

That second version is much stronger for a Statistics new grad or early-career candidate targeting growth and ads data science roles.
