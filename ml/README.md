# ML-B Scientific Evaluation

Frozen-methodology bake-off code for `docs/V2_ML_SPEC.md`. See
`reports/ml/ML_B_EXPERIMENT_REPORT.md` for results, interpretation, and the
Interview & Deep-Dive Notes; `reports/ml/PRE_EXPERIMENT_REPORT.md` for the
research done before any code was written.

## Run everything

```
python -m ml.run_all
```

Resets `reports/ml/results/experiment_log.jsonl` and regenerates every file
under `reports/ml/results/`. Requires the repo's pinned `requirements.txt`
environment. Seed 42 throughout — every result is deterministic given the
same environment and the same `data/evaluation/tier_b_benchmark.csv` /
`data/raw/synthetic_24mo.csv` / `models/kmeans_model.pkl` inputs.

## Run one part

```
python -m ml.categorization.run_bakeoff     # Part A only
python -m ml.forecasting.run_bakeoff        # Part B core bake-off only
python -m ml.forecasting.history_sensitivity  # Section 15 only
python -m ml.forecasting.sparsity_analysis    # Section 16 only
```

## Layout

- `common/` — splitting (merchant-grouped, category-stratified), metrics (WAPE/MAE/RMSE/MAPE, categorization bundle), experiment logging. Shared by both parts, no leakage-sensitive logic lives here.
- `categorization/` — the 3 frozen candidates (`candidates.py`) and the Part A orchestrator (`run_bakeoff.py`). Imports `pipeline.features.build_feature_matrix` read-only; never imports `pipeline.cluster.fit_and_evaluate`/`predict_categories` and never writes to `models/kmeans_model.pkl`.
- `forecasting/` — data prep (zero-filled monthly grid via the production K-Means artifact, read-only), calendar-boundary temporal evaluation (`temporal_eval.py`), the 4 frozen candidates (`candidates.py`, `baselines.py`), the 2 evaluated multi-step strategies (`strategies.py`), and the Section 15/16 experiments.
- `data/build_tier_b_benchmark.py` — regenerates `data/evaluation/tier_b_benchmark.csv` (hand-authored data; running this script does not change any row's content, only reproduces the same file from the same hand-specified source).

## Tests

```
python -m pytest tests/ml/ -v
```

Covers: merchant-group partition isolation, K-Means TRAIN-only fitting/mapping (via `KMeans.fit_predict` call-counting), calendar-boundary fold chronology, no-future-leakage in the RF/Ridge training matrix, WAPE/MAPE zero-actual correctness, and multi-step strategy correctness (an "echo" model proves the recursive strategy's features are actually recomputed from an extended history, not just relabeled).

`python -m pytest tests/` (the full existing suite, 248 tests including these) passes unchanged — ML-B added tests and evaluation code only; no file under `pipeline/`, `backend/`, `frontend/`, `config.py`, or `models/*.pkl` was modified.

## What this is not

Not model selection (ML-C), not final integration (ML-D), not a claim-certification pass (ML-E). No production behavior was changed. The FINAL UNTOUCHED TEST partition (categorization) and the reserved final 3 calendar months (forecasting) were never scored — see the main report §7/§35.
