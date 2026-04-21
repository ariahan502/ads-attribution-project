# Business Summary

## Objective

This project evaluates how ads event data can support better campaign review, ranking, targeting, and monitoring decisions in an offline setting.

The pipeline is designed to answer four practical questions:

- Which campaigns and interactions appear most valuable under descriptive attribution rules?
- How well do CTR models rank likely click outcomes?
- Can uplift-ranking methods recover known treatment-effect patterns in a controlled benchmark?
- Which users or events would be selected under a budgeted targeting policy?

## What The System Produces

The project produces reproducible run bundles for:

- CTR model training and evaluation
- campaign-level attribution reporting
- uplift diagnostics and semi-synthetic validation
- budgeted policy simulation
- row-level batch scoring
- feature, score, and calibration drift monitoring

Each run stores config snapshots, summaries, metrics, and report artifacts under `artifacts/runs/`.

## Key Result

On the 1M-row semi-synthetic benchmark, the XGBoost doubly robust uplift score recovers the known treatment-effect ordering strongly:

- Spearman correlation with true effect: `0.988762`
- top-decile true-effect lift: `1.726784`
- oracle top-decile true-effect lift: `1.738078`

The 10% budget policy captures `0.993502` of oracle expected incremental conversions under the same controlled benchmark.

## Decision Use

The outputs are intended for offline decision support:

- compare campaign performance under multiple attribution schemes
- inspect CTR ranking and calibration quality
- validate uplift-ranking mechanics on known semi-synthetic effects
- review targeting policies under fixed budget fractions
- generate deterministic scored outputs for downstream analysis
- monitor feature, score, and calibration changes over time

## Important Caveat

The project does not claim causal lift from observational logs alone.

CTR models estimate click likelihood. Attribution reports describe how credit changes under different rules. Semi-synthetic uplift experiments validate whether the pipeline can recover an injected treatment-effect signal. Real-world causal policy claims would require stronger identification assumptions or controlled experiments.

## Reproducibility

Run the self-contained quality gate with:

```bash
bash scripts/ci_smoke.sh
```

This command uses the tracked fixture and does not require the full raw dataset.
