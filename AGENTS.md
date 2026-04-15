# AGENTS.md

This file describes how to work effectively in this repository as an engineering agent.

## Project Intent

This repo is being evolved from a notebook-heavy prototype into a reproducible offline ads modeling project.

The current near-term goal is not to add every planned capability at once. The goal is to move the project forward in small, validated slices that improve structure, reproducibility, and credibility.

## Working Style

When making progress in this repo:

- Prefer small end-to-end slices over large speculative refactors.
- Leave the repo in a runnable state after each change.
- Favor reusable Python modules under `src/ads_project/` over notebook-only logic.
- Favor config-backed commands over hardcoded scripts.
- Validate changes with an actual command whenever practical.
- If a baseline result is weak, keep the infrastructure and use the result to guide the next improvement step.

Good progress in this repo looks like:

- one new command
- one new reusable module
- one measurable validation result
- one clear next step

## New Session Bootstrap

If you are starting in a fresh session, do not rely on prior chat context. Rebuild context from the repo itself.

Use this sequence:

1. Read `AGENTS.md`
2. Check `git status`
3. Read `ROADMAP.md` for the high-level direction
4. Read local planning docs if they exist:
   - `doc/execution-plan.md`
   - relevant `doc/feature-xxx/` notes
5. Inspect the currently implemented pipeline surface:
   - `configs/`
   - `src/ads_project/pipeline/`
   - `src/ads_project/models/`
   - `src/ads_project/data/`
6. Validate the current baseline commands before starting new work when practical

If the user says “check where we are and pick the next task,” default to:

1. identify the highest-priority task marked `New`, `Scoped`, `Partial`, or `In Progress` in `doc/execution-plan.md`
2. prefer the smallest task that can be completed with a real validation step
3. if the task is still too large, break it down in a local planning doc before coding

Before making changes, summarize:

- current repo state
- current highest-value next task
- planned validation command

## Repo Structure

Tracked source-of-truth code should live in:

- `src/ads_project/`
- `configs/`
- `scripts/`
- `README.md`
- `ROADMAP.md`

Notebooks are useful, but they are not the long-term source of truth for reusable logic.

Use notebooks for:

- EDA
- narrative analysis
- interpretation
- temporary experimentation

Move logic out of notebooks when it needs to be repeated, validated, or composed with other steps.

## Data Rules

Do not commit the large raw dataset.

Current expectations:

- raw data lives under `data/raw/`
- generated local samples live under `data/samples/`
- run artifacts live under `artifacts/runs/`

These are local working assets unless the user explicitly decides otherwise.

Before introducing any new data dependency, make sure the path is explicit in config or documentation.

## Config And Pipeline Rules

When adding a new workflow:

1. Put reusable logic in `src/ads_project/...`
2. Add a config file under `configs/` if the workflow has changeable parameters
3. Add a pipeline entrypoint under `src/ads_project/pipeline/`
4. Keep legacy scripts as thin wrappers only if they still provide convenience
5. Validate the workflow by actually running it

Avoid burying operational settings directly inside notebook cells or one-off scripts.

## Artifact Rules

When a pipeline run produces outputs, prefer a stable run bundle shape like:

- `artifacts/runs/<timestamp>_<run_name>/config.yaml`
- `artifacts/runs/<timestamp>_<run_name>/metrics.json`
- `artifacts/runs/<timestamp>_<run_name>/model.joblib`

Generated artifacts should usually stay local unless the user explicitly wants them tracked.

## Planning Rules

The repo uses a local-only planning workspace under `doc/`.

Important:

- `doc/` is intentionally excluded via `.git/info/exclude`
- planning docs are for iterative local execution planning
- do not assume planning docs are meant to be committed unless the user asks

Use:

- `doc/execution-plan.md` for the tracker and overall execution sequence
- `doc/feature-xxx/` folders for breaking large items into agent-sized tasks

If implementation work reveals a missing task, dependency, or risk, update the local planning docs.

## Validation Expectations

Prefer real validation over theoretical claims.

Examples:

- run the sample generation command after changing sample logic
- run the training pipeline after changing model orchestration
- inspect emitted metrics and artifact paths

If a full run is expensive, do a smoke run first.

If validation is skipped, say so clearly and explain why.

Current known validation commands:

- sample generation:
  - `python scripts/make_sample.py`
  - `PYTHONPATH=src python -m ads_project.pipeline.sample_data --config configs/sample.yaml`
- baseline CTR training:
  - `PYTHONPATH=src python -m ads_project.pipeline.train_ctr --config configs/ctr_baseline.yaml`

Use a smoke config override when faster feedback is needed.

## Baseline Modeling Guidance

The current CTR pipeline exists to create a reproducible baseline, not to claim a strong final model.

That means:

- weak metrics are still useful if the run is reproducible
- feature engineering should be added incrementally and compared against the baseline
- avoid jumping straight to more complex models before the baseline path is stable

When improving the CTR flow, the preferred order is:

1. extract reusable feature engineering
2. add train-only encodings carefully
3. rerun and compare metrics
4. expand evaluation
5. only then add more model complexity

## Safety Checks For Future Agents

Before committing changes:

- check `git status`
- make sure generated data and artifacts are not being staged accidentally
- watch for ignore rules that are too broad
  - example: `models/` can accidentally ignore `src/ads_project/models/`

Before changing project structure:

- confirm whether the change affects imports, config paths, or notebook assumptions
- prefer incremental migration over disruptive renames

## What To Do Next By Default

Unless the user redirects, the next highest-value work is:

1. extract notebook feature engineering into reusable modules
2. add train-only campaign CTR encoding
3. rerun the CTR baseline and compare results
4. add richer evaluation metrics and reports

If that work turns out to be too large, break it down in `doc/feature-ctr-evaluation/` before coding.

## Session Handoff Standard

At the end of a work session, leave enough state for the next agent to continue without chat history:

- committed code for tracked repo changes
- updated local planning docs if priorities or discoveries changed
- at least one concrete validation result
- a short statement of the next recommended task

If you discover new follow-up work, add it to `doc/execution-plan.md` instead of leaving it only in a chat reply.
