# Ads Attribution Project

This repository is a working codebase for experimenting with ad attribution data, sample generation, and a baseline click-through-rate (CTR) model.

The current focus is practical and reproducible:

- create a local sample from the raw dataset
- train a baseline CTR model from config
- save run artifacts in a predictable place
- keep the code organized under `src/ads_project/`

This repo does not claim causal uplift results from observational click logs. The baseline model is useful as a starting point, not as proof of incremental ad value.

## Current workflow

1. Place the raw dataset at `data/raw/criteo_attribution_dataset.tsv.gz`
2. Generate a parquet sample
3. Train the baseline CTR model on the sample
4. Inspect artifacts under `artifacts/runs/`

The raw dataset and generated sample are treated as local-only working files.

## Setup

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The project is organized as a `src/` layout, so the runnable module commands below use `PYTHONPATH=src`.

## Run the sample pipeline

Generate a parquet sample from the raw TSV:

```bash
python scripts/make_sample.py
```

Or run the shared module entrypoint directly:

```bash
PYTHONPATH=src python -m ads_project.pipeline.sample_data --config configs/sample.yaml
```

The default sample config is in `configs/sample.yaml`. It reads from:

- `data/raw/criteo_attribution_dataset.tsv.gz`

and writes to:

- `data/samples/sample_1m.parquet`

## Train the baseline CTR model

Run the baseline logistic regression pipeline:

```bash
PYTHONPATH=src python -m ads_project.pipeline.train_ctr --config configs/ctr_baseline.yaml
```

The default training config is in `configs/ctr_baseline.yaml`. It expects the sample parquet created above.

Each run writes artifacts to a timestamped directory under:

- `artifacts/runs/<timestamp>_ctr_baseline/`

Typical outputs include:

- `config.yaml`
- `metrics.json`
- `model.joblib`

## Repository layout

```text
.
├── configs/
│   ├── sample.yaml
│   └── ctr_baseline.yaml
├── scripts/
│   └── make_sample.py
├── src/
│   └── ads_project/
│       ├── artifacts.py
│       ├── config.py
│       ├── data/
│       ├── evaluation/
│       ├── models/
│       └── pipeline/
├── data/
│   ├── raw/
│   └── samples/
└── artifacts/
```

## Code organization

- `src/ads_project/config.py` loads YAML config files.
- `src/ads_project/data/` holds low-level data IO and sampling helpers.
- `src/ads_project/models/` holds the baseline model and time-ordered split logic.
- `src/ads_project/evaluation/` computes model metrics.
- `src/ads_project/artifacts.py` writes run outputs.
- `src/ads_project/pipeline/` contains the runnable CLI entrypoints.

## Notes on interpretation

- The baseline CTR model is intentionally simple and may perform weakly.
- Use it to validate data flow, feature wiring, and artifact tracking.
- Do not interpret CTR scores as causal uplift.
- If you want incremental impact or lift estimation, that needs a separate causal design.

## Legacy notebooks and references

The repository still includes notebook and raw-data reference material for exploration:

- `notebooks/01_eda.ipynb`
- `notebooks/02_ctr_model.ipynb`
- `data/raw/README.md`
- `data/raw/Experiments.ipynb`

These are useful for context, but the runnable project now lives in the config-backed scripts and `src/ads_project/` package.

## Next direction

The most natural next steps are:

- extract additional feature engineering into reusable modules
- add leakage-safe train-only encodings
- compare improved features against the current baseline
- keep each step small enough to validate end to end
