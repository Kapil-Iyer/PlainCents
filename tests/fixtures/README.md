# Test Fixtures

## Production vs. test artifacts — explicit distinction

| | Production artifact | Test artifact |
|---|---|---|
| Path | `models/tfidf_logreg_v1.pkl` (ML-D: selected categorizer), `models/kmeans_model.pkl`, `models/rf_model.pkl` (retired, preserved as ML-B evidence) | `tests/fixtures/logreg_model_test.pkl`, `tests/fixtures/kmeans_model_test.pkl` |
| Created by | `scripts/build_production_logreg_model.py` (ML-D) / `pipeline.cluster.fit_and_evaluate()` / `pipeline.forecast.fit_and_forecast()`, run manually against real/synthetic or frozen Tier B data | `tests/fixtures/build_test_logreg_model.py` / `tests/fixtures/build_test_kmeans_model.py`, run on demand |
| Committed to git? | No — `.gitignore`'s blanket `*.pkl` rule excludes it | No — also generated on demand, not committed, so the same blanket rule applies consistently rather than needing a special-case exception |
| Used by | The running V2 application (`tfidf_logreg_v1.pkl`, via `CategorizationService`) | Backend unit/integration tests (`CategorizationService`) |

**No developer needs to manually reconstruct an unknown model artifact before tests can run.** Before running backend tests that touch `CategorizationService`, run:

```bash
python tests/fixtures/build_test_logreg_model.py
```

This is deterministic (fixed `random_state=42`, fixed in-memory synthetic sample — no dependency on `data/raw/` or `data/evaluation/`) and produces `tests/fixtures/logreg_model_test.pkl` with the same payload shape (`{vectorizer, model, model_impl_version, metadata}`) production's `models/tfidf_logreg_v1.pkl` has, so `CategorizationService` (Build Plan Phase 3; ML-D Production Integration) can load it identically to a production artifact.

To (re)build the actual production artifact (fit on the frozen Tier B TRAIN partition, not a test sample):

```bash
python -m scripts.build_production_logreg_model
```

### Retired K-Means fixture

`tests/fixtures/build_test_kmeans_model.py` / `tests/fixtures/kmeans_model_test.pkl` are preserved (ML-B evidence, `pipeline/cluster.py`'s own tests) but are no longer what `CategorizationService` loads — K-Means was not the ML-C selected candidate.

## TD CSV fixtures

See `td_csv/README.md`.
