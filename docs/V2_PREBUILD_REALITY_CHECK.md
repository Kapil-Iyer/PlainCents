# PlainCents V2 — Pre-Build Reality Check

**Date:** 2026-09-01  
**Purpose:** Assess V1 as a foundation for V2 before writing PRD, TRD, and Build Plan.  
**Rules:** Derived from actual repository code. No V2 code written. No V1 files modified.

**Verified runtime context (synthetic_24mo.csv, 779 rows):**

| Metric | Value |
|--------|-------|
| Held-out categorization accuracy | 90.0% (36/40) |
| Wilson 95% CI | [76.9%, 96.0%] |
| Silhouette | 0.5437 |
| ARI (all 779 rows) | 0.8073 |
| Unique categories assigned by K-Means | 7 (`"Other"` unmapped) |
| End-to-end forecast MAPE (K-Means labels) | 29.4% |
| Diagnostic MAPE (keyword labels) | 15.7% |

---

## SECTION 1 — V1 Reusability Audit

| Module | FastAPI as-is? | What breaks in HTTP handler | Effort |
|--------|----------------|----------------------------|--------|
| **`pipeline/ingest.py` — `load_and_clean()`** | **No** | Expects a **filesystem path** under `DATA_RAW/` (`ingest.py:86-90`). Upload handlers give bytes/temp files, not `data/raw/filename.csv`. No in-memory CSV API. Only TD/RBC/Scotiabank (`BANK_COLUMNS`). Debit/credit not netted (`ingest.py:138`). | **MEDIUM** — add `load_and_clean_from_bytes()` or temp-file wrapper; optional bank param from API |
| **`pipeline/features.py` — `build_feature_matrix()`** | **Yes** | Pure function: DataFrame in → `(X, scaler, vectorizer)` out. Works for 1-row or N-row DataFrames. Caller must supply fitted scaler/vectorizer when `fit=False`. | **NONE** (if called from a service that loads model artifacts) |
| **`pipeline/cluster.py` — `predict_categories()`** | **Partial** | Works on any-size DataFrame including 1 row, but **reloads `joblib` from disk on every call** (`cluster.py:160-162`). Raises `FileNotFoundError` if pkl missing. Returns full DataFrame, not a single prediction DTO. `category` column is effectively `predicted_category` but not named that way. | **SMALL** — wrap in `CategorizationService` with cached model; add `predict_one()` helper |
| **`pipeline/cluster.py` — `fit_and_evaluate()`** | **No for HTTP** | Requires ≥200 rows (`cluster.py:108-109`). Long-running. Overwrites global pkl. Training endpoint only, never per-request. | **SMALL** wrap — keep offline/admin only |
| **`pipeline/forecast.py` — `fit_and_forecast()`** | **No** | Retrains RF every call (`forecast.py:301-302`). Runs walk-forward validation first. Requires ≥12 unique months (`forecast.py:46-48`). Can take 10–30+ seconds. Couples training + inference + metric reporting in one function. | **LARGE** — split into `validate()`, `train()`, `predict_forecast()`; background job for training |
| **`pipeline/portfolio.py` — `build_portfolio()`** | **Partial** | Needs `conn`, `holdings` list, `session_id`. Calls **yfinance** (network, slow, 1–4s per cold cache). `build_portfolio` **inserts to DB** inside (`portfolio.py:107`) — side effect in ML-ish module. Append-only portfolio rows. | **MEDIUM** — separate fetch-PnL from persist; accept conn via DI |
| **`db/database.py` — all helpers** | **Partial** | `get_connection()` runs full `schema.sql` every open (`database.py:19-25`). **No UPDATE/DELETE** for transactions. Queries return **all sessions** unfiltered (`get_transactions`, etc.). `insert_*` only — not CRUD. Uses `iterrows()` — slow at scale but OK for MVP. | **LARGE** — new repository layer with CRUD, filtering, migrations |
| **`viz/report.py` — `generate_report()`** | **Yes** | Needs `conn`; writes PDF to `EXPORTS_DIR`. Reads unfiltered tables — may mix sessions/seed data. Blocking, few seconds. Fine as `POST /reports/pdf` background or sync. | **SMALL** — add session filter param |
| **`viz/powerbi_export.py` — `export_powerbi_csvs()`** | **Yes** | Needs `conn`; writes 4 CSVs. Already filters latest runtime session for txn/forecast/portfolio (`powerbi_export.py:63-65`). `forecast_accuracy` unfiltered. | **SMALL** |

### Summary

**Reusable core logic:** ingest cleaning rules, feature matrix, K-Means predict path, forecast feature math, portfolio price fetch, SQL schema concepts.

**Not HTTP-ready:** orchestration in `main.py`, batch-only ingest path, forecast train-on-every-call, DB append-only helpers, unfiltered reads.

---

## SECTION 2 — Database Reality Check

### 1. Schema changes REQUIRED for V2 CRUD

#### `transactions` — predicted vs confirmed

**CODE FACT today:** Single `category` column (`schema.sql:11`). `main.py` stores K-Means output there. No `raw_description`, `bank`, `updated_at`, or override flag.

**Required changes:**

```sql
-- Conceptual V2 columns (design in TRD, not implementing now)
predicted_category   TEXT          -- model output at import time
confirmed_category   TEXT          -- user-facing truth; NULL until confirmed
is_manual_override   INTEGER       -- 0/1
raw_description      TEXT          -- pre-clean merchant string
bank_source          TEXT          -- TD, RBC, etc.
updated_at           DATETIME
import_batch_id      TEXT          -- replaces or supplements session_id semantics
```

**Analytics rule:** Use `COALESCE(confirmed_category, predicted_category)` as effective category.

**Migration from V1:** Rename/migrate existing `category` → `predicted_category`; set `confirmed_category = category` for existing rows.

#### New tables likely needed

| Table | Purpose |
|-------|---------|
| `import_batches` | Track CSV uploads (filename, bank, row count, status, timestamps) |
| `holdings` | Portfolio CRUD (replace append-only `portfolio` snapshots) |
| `forecast_runs` | When forecast was computed, model version, MAPE snapshot |
| Optional: `users` | Only if multi-user later; skip for single-user MVP |

#### Existing tables — column changes

| Table | Change |
|-------|--------|
| `predictions` | Add `forecast_run_id`; UNIQUE on `(forecast_run_id, category, month_offset)` or latest-run flag |
| `portfolio` | Split into `holdings` (master) + `portfolio_snapshots` (optional history) |
| `forecast_vs_actual` | UNIQUE on `(category, forecast_month)`; flag `source` (seed vs runtime) |
| `monthly_summary` | Fix semantics: `forecast_next_month` should not be stamped on every historical month |

#### Indexes missing for V2

```sql
CREATE INDEX idx_transactions_date ON transactions(date);
CREATE INDEX idx_transactions_confirmed_category ON transactions(confirmed_category);
CREATE INDEX idx_transactions_import_batch ON transactions(import_batch_id);
CREATE INDEX idx_predictions_forecast_month ON predictions(forecast_month, category);
CREATE UNIQUE INDEX idx_fva_category_month ON forecast_vs_actual(category, forecast_month);
```

### 2. DB helpers — reuse vs replace

| Helper | V2 reuse |
|--------|----------|
| `get_connection()` | **Reuse** with migration runner added |
| `upsert_price_cache()` | **Reuse as-is** |
| `get_price_cache()` | **Reuse as-is** |
| `insert_transactions()` | **Replace** — need single-row insert, upsert on import dedup key |
| `get_transactions()` | **Replace** — add filters (date, category, pagination, latest batch) |
| `insert_predictions()` | **Replace** — tie to forecast_run, replace-latest semantics |
| `insert_portfolio()` | **Replace** — holdings CRUD |
| `upsert_monthly_summary()` | **Refactor** — compute server-side per request or materialized view |
| `insert_forecast_vs_actual()` | **Refactor** — upsert not append |
| All query helpers | **Wrap in repository** with session/batch scope |

### 3. Append-only transactions → CRUD

**CODE FACT:** `insert_transactions` does blind `INSERT` (`database.py:46-50`). No UNIQUE constraint. Re-running `main.py` duplicates all 779 rows.

**V2 needs:**

- Stable `id` per transaction (already have `INTEGER PRIMARY KEY`)
- `UPDATE` / `DELETE` by id
- Import dedup: UNIQUE on `(date, merchant, amount, bank_source)` or hash of raw row
- Stop using `session_id` as the only way to scope "current" data

### 4. `session_id` pattern

**CODE FACT:** `SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")` in `main.py:30`. Used on transactions, predictions, portfolio. PowerBI export picks latest matching `^\d{8}_\d{6}$` (`powerbi_export.py:26-28`).

**V2 change:** `session_id` is a batch-run artifact, not a product concept. Replace with:

- `import_batch_id` for imports
- `forecast_run_id` for forecasts
- Optional `user_id` later

Keep `session_id` only if you want audit history of pipeline runs — don't expose it to React users.

---

## SECTION 3 — ML Service Boundary

### 1. Single-transaction categorization

**Today:** `predict_categories(df)` works on 1 row but reloads pkl + builds 1×53 matrix each call.

**V2 needs:**

```
CategorizationService (singleton, startup)
  ├── load kmeans_model.pkl once
  └── predict_one(date, merchant, amount) → {predicted_category, cluster_id, confidence?: null for K-Means}
```

**Optional thin wrapper:** 1-row DataFrame → existing `build_feature_matrix` + `kmeans.predict` — no algorithm change needed.

**Effort:** SMALL (service wrapper), not LARGE (algorithm rewrite).

### 2. `fit_and_forecast()` on every request

**CODE FACT:** Every call runs `walk_forward_validate()` then `rf.fit(X_all, y_all)` (`forecast.py:270-302`). Unacceptable on HTTP GET.

**Right architecture:**

```
ForecastService
  ├── check_cold_start(transactions) → 422/200 with status payload
  ├── run_forecast_job()          # POST /forecasts/run — background or explicit
  │     ├── aggregate_monthly from DB (confirmed categories)
  │     ├── optional: walk_forward for metrics (admin only)
  │     ├── train RF
  │     └── persist predictions + forecast_run metadata
  └── get_latest_forecasts()      # GET /forecasts — read DB only
```

**Retrain triggers:** import completes, user confirms categories in bulk, manual "Refresh forecast" button — **not** page load.

### 3. `kmeans_model.pkl` dependency

**CODE FACT:** `predict_categories` raises if missing (`cluster.py:160-161`). README says run `python -m pipeline.cluster` first.

**V2 handling:**

1. **Startup check:** if pkl missing → log warning, categorization endpoint returns 503 with clear message
2. **Ship default model** in repo or download step in setup (controversial for git size)
3. **Admin endpoint** `POST /models/categorization/train` — offline only
4. **Document** in README/setup: first-run trains or copies bundled model

For fastest demo: **bundle a pre-trained pkl** or run training in setup script.

### 4. Minimum data for forecasting + cold-start API

**CODE FACT:** `aggregate_monthly` raises if `< 12` unique months (`forecast.py:46-48`).

Additionally, `build_forecast_features` drops rows needing 6 months of per-category history — practical minimum is **~12 calendar months of transactions** with coverage across categories.

**API response pattern:**

```json
{
  "status": "cold_start",
  "reason": "insufficient_history",
  "months_available": 3,
  "months_required": 12,
  "message": "More transaction history is required before a spending forecast can be generated."
}
```

HTTP **200 with status field** (product state) or **422** — pick one in TRD and stay consistent.

---

## SECTION 4 — V2 Repo Structure

```
PlainCents/
├── README.md
├── requirements.txt                    # NEW — missing in V1
├── pyproject.toml                      # NEW (optional)
├── .env.example                        # NEW
│
├── backend/                            # NEW — FastAPI app
│   ├── main.py                         # FastAPI entry, lifespan, model load
│   ├── api/
│   │   ├── routes/
│   │   │   ├── transactions.py       # NEW
│   │   │   ├── import_csv.py           # NEW
│   │   │   ├── categories.py         # NEW
│   │   │   ├── forecasts.py            # NEW
│   │   │   ├── portfolio.py            # NEW
│   │   │   ├── analytics.py            # NEW
│   │   │   └── reports.py              # NEW
│   │   └── schemas/                    # NEW — Pydantic request/response
│   │       ├── transaction.py
│   │       ├── forecast.py
│   │       └── ...
│   ├── services/                       # NEW
│   │   ├── ingestion_service.py        # wraps ingest
│   │   ├── categorization_service.py   # wraps cluster
│   │   ├── forecast_service.py         # wraps forecast (split)
│   │   ├── portfolio_service.py        # wraps portfolio
│   │   └── reporting_service.py        # wraps viz
│   └── repositories/                   # NEW
│       ├── transaction_repo.py
│       ├── forecast_repo.py
│       └── portfolio_repo.py
│
├── pipeline/                           # CARRIED OVER — mostly unchanged
│   ├── ingest.py                       # REFACTORED — add bytes/upload entry
│   ├── features.py                     # UNCHANGED
│   ├── cluster.py                      # UNCHANGED core; service wraps it
│   ├── forecast.py                     # REFACTORED — split train/predict
│   └── portfolio.py                    # REFACTORED — separate fetch vs persist
│
├── db/
│   ├── schema.sql                      # REFACTORED — V2 migrations
│   ├── migrations/                     # NEW
│   ├── database.py                     # REFACTORED — connection only
│   └── seed_synthetic_data.py          # CARRIED OVER — dev only
│
├── viz/                                # CARRIED OVER
│   ├── report.py                       # SMALL REFACTOR — session filter
│   └── powerbi_export.py               # SMALL REFACTOR
│
├── config.py                           # CARRIED OVER — extend paths
│
├── models/                             # GENERATED — gitignored
│   ├── kmeans_model.pkl
│   └── rf_model.pkl
│
├── scripts/                            # CARRIED OVER — dev tools
│   ├── generate_synthetic_24mo.py
│   └── ...
│
├── frontend/                           # NEW
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── pages/                      # Overview, Transactions, Import, etc.
│       ├── components/
│       ├── api/                        # fetch wrappers
│       └── types/
│
├── data/
│   ├── raw/                            # gitignored
│   └── exports/                        # gitignored
│
├── docs/
│   ├── TECHNICAL_WALKTHROUGH.md        # V1 reference
│   ├── V2_PREBUILD_REALITY_CHECK.md    # this document
│   ├── V2_PRD.md                       # NEW
│   ├── V2_TRD.md                       # NEW
│   └── V2_BUILD_PLAN.md                # NEW
│
├── tests/                              # NEW
│   ├── test_ingest.py
│   ├── test_api_transactions.py
│   └── ...
│
└── main.py                             # LEGACY V1 batch — keep until V2 parity, then deprecate
```

---

## SECTION 5 — Top 5 Technical Risks

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| **1** | **V1 DB append-only semantics exposed via API** | Duplicate transactions, wrong analytics, broken CRUD | Schema migration + repository layer before any React work |
| **2** | **`fit_and_forecast()` called synchronously from API** | Timeouts, RF retrain on every refresh, unusable UX | Split train/read; explicit forecast job; GET reads DB only |
| **3** | **K-Means model missing / stale on deploy** | 503 on import, blank categories | Startup health check; bundle default pkl; setup script |
| **4** | **Synthetic-trained model on real bank CSVs** | Bad categories, user distrust, bad forecasts | Ship V2 with honest UX; manual correction; defer metric claims; plan classifier swap |
| **5** | **Scope creep (Big Six + full dashboard + ML bake-off + PowerBI)** | Never ships | Phase V2: TD-only import + txn CRUD + basic dashboard first; add banks incrementally |

---

## SECTION 6 — Fast Path Recommendation

### 1. Minimum V1 hardening before V2 (only blockers)

| # | Blocker | Why |
|---|---------|-----|
| 1 | Add `requirements.txt` | Reproducible backend setup |
| 2 | Document/train path for `kmeans_model.pkl` | Import will fail without it |
| 3 | Design V2 schema (predicted/confirmed, holdings, no append-only txn) | CRUD impossible on V1 schema |
| 4 | Split `fit_and_forecast` conceptually in TRD | Prevents copying batch pattern into API |

**NOT blockers for starting V2:** fixing 90% eval methodology, GridSearchCV leakage, README stale status — fix in parallel or post-MVP.

### 2. Recommended V2 phase order

| Phase | Deliverable |
|-------|-------------|
| **0** | PRD + TRD + schema + API contract (frozen) |
| **1** | FastAPI skeleton + SQLite V2 schema + transaction repository |
| **2** | CSV upload (TD only) + ingest service + store with predicted_category |
| **3** | Transaction CRUD + category correction API |
| **4** | React: Transactions + Import pages |
| **5** | CategorizationService (cached pkl) |
| **6** | Forecast job + GET forecasts + cold-start UI |
| **7** | React: Overview + Forecasts dashboard |
| **8** | Portfolio CRUD + price refresh |
| **9** | Analytics endpoints + charts |
| **10** | PDF/PowerBI export endpoints |
| **11** | Tests + polish + demo data |
| **12** | ML re-validation (parallel track, not blocking demo) |

### 3. Minimal change vs build from scratch

| Feature | Reuse V1 | Build new |
|---------|----------|-----------|
| CSV cleaning logic | ✅ `ingest.py` core | Upload wrapper, bank adapter interface |
| TF-IDF + K-Means predict | ✅ `features.py` + `cluster.py` | Service singleton |
| RF forecast math | ✅ `forecast.py` functions | Service split, DB-backed |
| yfinance + cache | ✅ `portfolio.py` fetch logic | Holdings CRUD, API |
| PDF / PowerBI | ✅ `viz/*` | HTTP endpoints, filters |
| FastAPI routes | ❌ | All |
| React UI | ❌ | All |
| Transaction CRUD | ❌ | All |
| Repository layer | ❌ | All |
| Auth / users | ❌ | Defer |

### 4. Realistic time estimates (Claude Code + human review gates)

| Phase | Estimate |
|-------|----------|
| 0 — Docs frozen | 2–3 days |
| 1 — API + DB foundation | 3–4 days |
| 2–3 — Import + txn CRUD backend | 4–5 days |
| 4 — React txn/import | 4–5 days |
| 5–7 — ML services + forecast + dashboard | 5–7 days |
| 8–9 — Portfolio + analytics | 3–4 days |
| 10–11 — Reports + tests + polish | 3–4 days |
| **Working demo total** | **~4–5 weeks** part-time or **~2–3 weeks** full-time |
| 12 — ML re-validation | 1–2 weeks parallel |

---

## V2 Vision Assessment

### What aligns with V1 reality

1. **V2 wraps V1; it doesn't discard it.** Ingest, features, cluster predict, forecast math, portfolio cache, viz exports are real reuse candidates.
2. **React primary, PowerBI optional** — correct; V1 has no frontend.
3. **SQLite first** — correct for portfolio scope; repository abstraction is enough.
4. **`predicted_category` vs `confirmed_category`** — essential; V1's single `category` column is insufficient.
5. **Forecast as explicit operation, not on every GET** — mandatory; `fit_and_forecast()` today proves this.
6. **Cold-start as product state** — correct; code already raises at `<12` months.
7. **Don't claim V1 metrics on resume** — correct; 29.4% end-to-end, 15.7% is ablation only.
8. **CategorizationService abstraction** — correct; decouples UI from K-Means vs logistic regression later.
9. **Offline retraining only** — correct.
10. **Front-load PRD/TRD before coding** — correct given AI-generated V1 drift.

### Scope adjustments recommended

| Vision item | Recommendation |
|-------------|----------------|
| Big Six banks in V2 | Too early. Ship **TD first**, adapter interface second, other banks in V2.1. |
| K-Means vs Logistic Regression bake-off in V2 | Wrong phase. Ship with K-Means via service wrapper; bake-off is post-demo validation. |
| Full 6-page dashboard in v1 of V2 | Start with Overview + Transactions + Import + Forecasts. |
| Skip V1 understanding entirely | Risky. Audit facts (eval flaws, session duplication, metric meanings) must inform V2 schema. |
| 2–3 week full-time demo | Achievable for **narrow MVP** only. Full feature list is **6–8 weeks**. |
| User accounts / auth | Defer unless multi-user deployment is required. |
| PostgreSQL | Defer; document SQLite single-writer limit in TRD for interview defense. |

### Critical architectural warning

**V1's `session_id` + append-only design will infect V2 if `main.py` is wrapped naively.** The biggest refactor is not FastAPI or React — it is **making transactions a first-class CRUD entity** instead of a batch dump. Do schema + repository in Phase 1 before any UI.

---

## Bottom Line

V2 is a sound evolution of V1, not a rewrite. Use this document as input for V2 PRD/TRD with explicit **MVP vs later** tiers. Cut scope to **TD import + txn CRUD + categorization + cold-start forecast + simple dashboard** for the first shippable demo.

**Next step:** PRD → TRD → frozen API contract + schema → Phase 1 implementation.
