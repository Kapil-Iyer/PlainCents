# Test Fixtures

## Production vs. test artifacts — explicit distinction

| | Production artifact | Test artifact |
|---|---|---|
| Path | `models/kmeans_model.pkl`, `models/rf_model.pkl` | `tests/fixtures/kmeans_model_test.pkl` |
| Created by | `pipeline.cluster.fit_and_evaluate()` / `pipeline.forecast.fit_and_forecast()`, run manually against real/synthetic data | `tests/fixtures/build_test_kmeans_model.py`, run on demand |
| Committed to git? | No — `.gitignore`'s blanket `*.pkl` rule excludes it | No — also generated on demand, not committed, so the same blanket rule applies consistently rather than needing a special-case exception |
| Used by | The running V1/V2 application | Backend unit/integration tests (`CategorizationService`) |

**No developer needs to manually reconstruct an unknown model artifact before tests can run.** Before running backend tests that touch `CategorizationService`, run:

```bash
python tests/fixtures/build_test_kmeans_model.py
```

This is deterministic (fixed `random_state=42`, fixed in-memory synthetic sample — no dependency on `data/raw/`, which is gitignored and not guaranteed present on a fresh clone) and produces `tests/fixtures/kmeans_model_test.pkl` with the same payload shape (`{kmeans, scaler, vectorizer, cluster_to_category}`) `pipeline.cluster.predict_categories()` expects, so `CategorizationService` (Build Plan Phase 3) can load it identically to a production artifact.

## TD CSV fixtures

See `td_csv/README.md`.
