# ML-E: Reproducibility Guide

For a fresh technical reviewer, starting from a clean clone at HEAD `9e1c8877ab59ccf7d27b60e7130d4adbfe65ecfa` (or later, once ML-D/ML-E are committed). Every command below depends only on files tracked in git plus `requirements.txt`-pinned packages — never on `.claude/`, a developer's existing `.venv`/`venv` contents, a personal SQLite DB, or private bank data.

## 0. Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 1. Generate the production categorization artifact

Not committed (`.gitignore`'s blanket `*.pkl` rule) — generated on demand from committed evidence:

```bash
python -m scripts.build_production_logreg_model
```

Expected output: `Fit on TRAIN only: 133 rows, 47 merchant groups`, writing `models/tfidf_logreg_v1.pkl`. This refuses to run (raises `SelectionNotFrozenError`) unless `reports/ml/ML_C_SELECTION_RECORD.json` exists and names `tfidf_logreg` as selected — it will never silently build an artifact for an unselected candidate.

Verify its metadata:

```bash
python -c "import joblib, json; print(json.dumps(joblib.load('models/tfidf_logreg_v1.pkl')['metadata'], indent=2, default=str))"
```

Expect `model_impl_version: "tfidf_logreg_v1"`, `fit_partition: "TRAIN"`, `fit_partition_n_rows: 133`, `fit_partition_n_merchant_groups: 47`, and a `recipe` block matching `reports/ml/ML_C_SELECTION_RECORD.json`'s `categorization_selection.exact_configuration` exactly (`C: 1.0, max_iter: 1000, random_state: 42`).

Forecasting has **no equivalent artifact to generate** — Naive is stateless code (`ml/forecasting/baselines.py::naive_predict`), recomputed fresh on every forecast run. There is nothing to build or version beyond the `model_impl_version = "naive_v1"` label already in `backend/services/forecast_service.py`.

## 2. Build the test-fixture artifact (for backend tests)

```bash
python tests/fixtures/build_test_logreg_model.py
```

Deterministic (seed 42, small in-memory synthetic sample — no dependency on `data/evaluation/`), writes `tests/fixtures/logreg_model_test.pkl`. This is a *different* artifact from the production one above — see `tests/fixtures/README.md`'s production-vs-test-artifact table. Backend tests that exercise `CategorizationService` depend on this file existing; `pytest` does not build it automatically.

## 3. Run focused ML-D tests

```bash
pytest tests/backend/services/test_categorization_service.py tests/backend/services/test_forecast_train_and_predict.py tests/backend/services/test_forecast_service.py -v
```

## 4. Run ML-B/ML-C permitted evidence checks

These re-derive/verify the frozen split and selection artifacts without ever touching a rejected candidate's FINAL score:

```bash
python -m ml.categorization.run_bakeoff     # VALIDATION comparison of all 3 categorization candidates
python -m ml.forecasting.run_bakeoff        # VALIDATION comparison of all forecasting candidates/strategies
python -m ml.forecasting.fold_stability     # 14-origin expanding-window stability summary
```

**Permitted reproducibility re-run of FINAL (selected candidate only)**, per this session's explicit addendum to the ML Spec's FINAL-test discipline — verification, not a new selection round:

```bash
python -m ml.categorization.run_final       # re-fits TfidfLogRegCandidate on TRAIN, scores FINAL_TEST once
python -m ml.forecasting.run_final          # re-derives the reserved period, scores it once with Naive
```

Both refuse to run unless `ML_C_SELECTION_RECORD.json` names their own candidate as selected, and both overwrite `reports/ml/results/final_*.json` with fresh `evaluation_timestamp_utc`/`git_commit` fields — **do not commit that overwrite** unless the numbers genuinely changed (they should not; see Verification below). If you ran these locally purely to verify, restore the committed file afterward:

```bash
git checkout -- reports/ml/results/final_categorization.json reports/ml/results/final_forecasting.json
```

**Never** run `run_final.py`-equivalent logic against a rejected candidate (K-Means, Linear SVM, Seasonal Naive, RF, Ridge) — no code path in this repo does this, and none should be added.

## 5. Verification: this session's own reproduction

Re-running step 4's `run_final.py` commands at HEAD `9e1c8877ab59ccf7d27b60e7130d4adbfe65ecfa` (two commits after the original ML-C freeze at `2c06181a12fb270e6e534564c98ccebd2998088c`) reproduced:

| Metric | Committed (`2c06181`) | Re-run (`9e1c887`) |
|---|---|---|
| Categorization macro F1 | 0.4405421207145345 | 0.4405421207145345 |
| Categorization accuracy | 0.4222222222222222 | 0.4222222222222222 |
| Forecasting combined WAPE | 0.18865752437529387 | 0.18865752437529387 |

Every field except `git_commit` and `evaluation_timestamp_utc` was byte-identical. This confirms the frozen split, frozen recipe, and frozen selection are fully deterministic and reproducible from committed evidence alone.

## 6. Verify production/selection artifact metadata agree

```bash
python -c "
import json
sel = json.load(open('reports/ml/ML_C_SELECTION_RECORD.json'))
print(sel['categorization_selection']['exact_configuration'])
"
```

Compare against step 1's printed artifact metadata — `C`, `max_iter`, `random_state` must match exactly (they do, by construction: `scripts/build_production_logreg_model.py` reads this same file).

## 7. Run backend tests

```bash
pytest
```

Expected: all tests under `tests/backend/`, `tests/ml/`, `tests/test_pipeline.py`, `tests/test_phase0_fixtures.py` pass (270 as of this ML-D/ML-E pass). Requires step 2's test fixture to exist first.

## 8. Run frontend tests / typecheck / build

```bash
cd frontend
npm install
npm test -- --run
npm run typecheck
npm run build
cd ..
```

## 9. Run E2E

```bash
npm install                # repo root — installs @playwright/test
npm run e2e:install        # once — downloads the Chromium browser
npm run e2e
```

`tests/e2e/global-setup.ts` builds the frontend and, if `models/tfidf_logreg_v1.pkl` is missing, bootstraps it from step 2's test-fixture script (copied into place) so the Import flow has a loadable categorizer — it never overwrites a real production artifact already present. If you want E2E to exercise the actual production-trained artifact rather than the test fixture, run step 1 before step 9.

## 10. Run reviewer mode

```bash
python -m backend.scripts.run_reviewer
```

Then open `http://127.0.0.1:8000` and confirm `/api/health` reports `"categorization_model": "loaded"` (requires step 1's artifact, or the E2E bootstrap copy, to exist at `models/tfidf_logreg_v1.pkl`).

## Generated vs. committed files (quick reference)

| File | Committed? | How to produce |
|---|---|---|
| `models/tfidf_logreg_v1.pkl` | No (gitignored) | `python -m scripts.build_production_logreg_model` |
| `tests/fixtures/logreg_model_test.pkl` | No (gitignored) | `python tests/fixtures/build_test_logreg_model.py` |
| `models/kmeans_model.pkl`, `models/rf_model.pkl` | No (gitignored, retired paths) | `python -m pipeline.cluster` / V1's `fit_and_forecast` |
| `data/evaluation/tier_b_benchmark.csv`, `tier_b_split_v1.json` | **Yes** | Frozen ML-B evidence — never regenerate/overwrite |
| `reports/ml/**` (all reports/results) | **Yes** | Frozen ML-B/C/E evidence — `run_final.py` re-runs may overwrite locally; restore via `git checkout` unless numbers genuinely changed |
| `frontend/dist/` | No (gitignored build output) | `npm run build` |
