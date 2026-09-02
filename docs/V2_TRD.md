# PlainCents V2 — Technical Requirements Document

**Status:** FROZEN — V2 Technical Requirements
**Traces to:** `docs/V2_PRD.md` (FROZEN), `docs/V2_PREBUILD_REALITY_CHECK.md`
**Scope:** How V2 is built. Product scope is not decided here — see PRD for what/why. No code, migrations, or ML model-selection criteria are produced by this document.

---

## 0. How to read this document

Every major decision below states the PRD requirement or repository fact that drove it. Where the authoring prompt suggested a specific mechanism that conflicts with repository reality or the frozen PRD, the conflict is flagged inline and in §22 self-audit item A rather than silently followed.

---

## 1. System Architecture

```
React + TypeScript + Vite  (frontend, browser)
        ↓ REST / JSON over HTTP
      FastAPI                (backend, routes/controllers)
        ↓
    Service Layer            (application orchestration)
        ↓
   Repository Layer          (persistence access)
        ↓
       SQLite                (plaincents_v2.db)

ML/pipeline modules (categorization, forecasting, price fetch)
are invoked only from the Service Layer.
```

### 1.1 React/Vite frontend
- **Responsibilities:** render pages, collect user input, call the REST API, hold ephemeral UI/server-cache state.
- **Must not:** talk to SQLite, call `yfinance`, know about K-Means/RF internals, contain business rules (e.g., what makes a forecast stale).
- **Talks to:** FastAPI only, via `fetch`/HTTP.
- **V1 mapping:** none — V1 has no frontend.

### 1.2 FastAPI routes/controllers
- **Responsibilities:** HTTP concerns only — parse request, call exactly one service method, map the result/exception to an HTTP response per §15.
- **Must not:** contain SQL, call `pipeline/` or `viz/` modules directly, contain multi-step orchestration.
- **Talks to:** Service Layer (down), React (up, via JSON).
- **V1 mapping:** replaces `main.py`'s role as "the thing you run", but `main.py` itself is not deleted (§18).

### 1.3 Service Layer
- **Responsibilities:** business orchestration — the only place that coordinates more than one repository, decides when a forecast becomes stale, enforces demo/real mutual exclusivity, calls ML/pipeline code, calls `yfinance`.
- **Must not:** write raw SQL (delegates to repositories), be called directly by React.
- **Talks to:** Repository Layer, `pipeline/` modules, `yfinance` (down); Routes (up).
- **V1 mapping:** this layer does not exist in V1; it is built new, wrapping V1's `pipeline/*` functions.

### 1.4 Repository Layer
- **Responsibilities:** persistence access only — SQL, parameter binding, mapping rows to typed structures.
- **Must not:** contain business rules, call ML code, call `yfinance`, decide that a change in one table should trigger a change in another table's data.
- **Talks to:** SQLite (down); Services (up).
- **V1 mapping:** refactors `db/database.py`, which today mixes a couple of persistence-adjacent decisions (e.g., `INSERT OR REPLACE`) but no business orchestration — a reasonable starting point.

### 1.5 SQLite (`plaincents_v2.db`)
- **Responsibilities:** durable local storage.
- Single file, WAL mode (`db/database.py:22` already does this and is retained).

### 1.6 ML/pipeline modules
- **Responsibilities:** categorization inference (`pipeline/cluster.py`), forecast aggregation/fit/predict (`pipeline/forecast.py`), price fetch (`pipeline/portfolio.py`). Pure(ish) functions/classes, no HTTP or React awareness.
- **Talks to:** invoked only by the Service Layer, never by routes or repositories (§7, §8).

### 1.7 Local development topology

- **Vite dev server** (default `http://localhost:5173`) serves the React app with HMR.
- **FastAPI dev server** (`uvicorn backend.main:app --reload`, default `http://localhost:8000`) serves the API.
- **Frontend → backend:** Vite's dev proxy forwards `/api/*` to `http://localhost:8000`, so the browser only ever talks to one origin (`localhost:5173`) during development. This avoids CORS entirely in dev.
- **CORS:** FastAPI's CORS middleware allows exactly the Vite dev origin (`http://localhost:5173`) as a fallback for any request that bypasses the proxy (e.g., a manually opened API URL). No wildcard origin.
- **Packaged/local "production-like" run:** for a portfolio demo where a reviewer runs one command, recommend **building the Vite app to static files and having FastAPI serve them** (`StaticFiles` mount at `/`, API at `/api/*`, single process, single port). This is simpler for a reviewer ("run one command, open one URL") than running two dev servers, and avoids introducing a second deployment artifact. Justification: MVP is local-first, single-user, single-machine — there is no need for independently scalable frontend/backend processes, so collapsing to one process for the "demo mode" run is the right tradeoff. Development still uses two servers (for HMR).

### 1.8 Why SQLite

- Single-user, local-first (PRD §7, §9.9) — no concurrent-writer requirement.
- Zero operational overhead: no server process, no network config, matches "run it locally" demo goal.
- V1 already uses it successfully for the batch pipeline.

**Concurrency limitation (documented, not solved):** SQLite allows many concurrent readers but only one writer at a time; a long-running write (e.g., a large import) blocks other writers for its duration. WAL mode (already enabled) allows readers to proceed during a write, which covers the MVP's actual usage pattern (one local user, one browser tab, infrequent writes). If a future version needs multiple simultaneous writers (multi-user), that is a PostgreSQL-shaped problem — not solved here.

**Why not PostgreSQL now:** no requirement in the frozen PRD needs multi-writer concurrency, network access, or horizontal scale. Introducing it now would be infrastructure without a driving requirement (PRD §9.11, TRD §19). The Repository Layer (§8) isolates all SQL behind typed methods, so a future swap to PostgreSQL would touch the repository implementations only — services, routes, and the frontend would not need to change.

---

## 2. Repository Structure

```
PlainCents/
├── backend/                          # NEW
│   ├── main.py                       # FastAPI app, lifespan (model load, migrations)
│   ├── api/
│   │   └── routes/
│   │       ├── health.py
│   │       ├── demo.py
│   │       ├── imports.py
│   │       ├── transactions.py
│   │       ├── categories.py
│   │       ├── forecasts.py
│   │       ├── holdings.py
│   │       └── dashboard.py
│   ├── schemas/                      # Pydantic request/response models (§6)
│   ├── services/                     # §7
│   ├── repositories/                 # §8
│   └── config.py                     # V2 config (§16); imports shared constants from root config.py
│
├── pipeline/                         # REUSED/WRAPPED/REFACTORED from V1 — see §3
│   ├── ingest.py                     # REFACTORED (bytes-based entry point added)
│   ├── features.py                   # REUSED AS-IS
│   ├── cluster.py                    # WRAP (unchanged file; CategorizationService wraps its internals — see §3)
│   ├── forecast.py                   # REFACTORED (split responsibilities, see §12)
│   └── portfolio.py                  # REFACTORED (fetch/persist separated)
│
├── db/
│   ├── migrations/                   # NEW — sole source of truth for V2 DDL (§4.11, §4.12)
│   │   └── 001_initial_v2.sql        # Full initial V2 schema; later changes are 002_*.sql, etc.
│   ├── database.py                   # V1 — LEGACY/PRESERVED, untouched
│   ├── schema.sql                    # V1 — LEGACY/PRESERVED, untouched
│   └── seed_synthetic_data.py        # V1 — LEGACY/PRESERVED; V2 demo data is a separate NEW script (§14.1)
│
├── viz/                               # LEGACY/PRESERVED — not wired into V2 MVP (PRD §17)
│   ├── report.py
│   └── powerbi_export.py
│
├── config.py                          # V1 — PRESERVED; V2 reads shared constants (CATEGORIES) from it (§16)
├── main.py                            # V1 — PRESERVED, runnable, unchanged (§18)
│
├── frontend/                          # NEW
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── pages/                    # Dashboard, Transactions, Import, Forecast, Portfolio
│       ├── components/
│       ├── api/                      # typed fetch client
│       └── types/                    # TS types mirroring backend schemas
│
├── data/, models/, scripts/, powerbi/ # UNCHANGED
│
├── tests/
│   ├── test_pipeline.py              # V1 — PRESERVED
│   ├── backend/                      # NEW — unit/repository/API tests (§17)
│   └── fixtures/                     # NEW — TD CSV fixtures, test model artifacts
│
└── docs/
    ├── V2_PRD.md, V2_PREBUILD_REALITY_CHECK.md, V2_TRD.md (this file)
```

**Do not delete during implementation:** `main.py`, `db/database.py`, `db/schema.sql`, `db/seed_synthetic_data.py`, `viz/*`, `pipeline/*` original function signatures where V1 entry points (`python -m pipeline.cluster`, etc.) depend on them (§18.1).

---

## 3. V1 Reuse / Wrap / Refactor / Replace Map

Each classification below is based on the file actually read in this session (paths and line numbers cited).

| Module | Current responsibility | Classification | V2 responsibility | Changes needed | Compatibility risk |
|---|---|---|---|---|---|
| `pipeline/ingest.py` | `load_and_clean(csv_path, bank)` reads a **filesystem path** (`ingest.py:66-90`), detects/validates bank columns, parses dates via `BANK_DATE_FORMATS`, cleans merchant, drops unparseable rows, dedupes via `drop_duplicates()` (exact-row only, `ingest.py:143`). | **REFACTOR** | Add a bytes-based entry point so an uploaded file (not a filesystem path) can be parsed; keep all cleaning/parsing logic. | Add `load_and_clean_from_bytes(file_bytes: bytes, bank: str) -> pd.DataFrame` that wraps the existing logic (`pd.read_csv(io.BytesIO(file_bytes))` instead of `Path`), then delegates to the same column-mapping/date/merchant logic. `load_and_clean(csv_path, bank)` is left **untouched** so V1's CLI path keeps working. Note: V1's `drop_duplicates()` is exact full-row dedup, not the "already imported" dedup the PRD requires (§4.4) — that is new service-layer logic, not a change to this function. | Low — additive function, no existing signature changes. |
| `pipeline/features.py` | `build_feature_matrix(df, scaler, vectorizer, fit)` — pure function, works on any-size DataFrame (`features.py:11-68`). | **REUSE AS-IS** | Unchanged; called by `CategorizationService`. | None. | None — verified pure, no I/O. |
| `pipeline/cluster.py` | `predict_categories(df)` reloads `kmeans_model.pkl` from disk via `joblib.load` on **every call** (`cluster.py:160-172`); raises `FileNotFoundError` if missing (`cluster.py:161`). `fit_and_evaluate()` requires ≥200 rows (`cluster.py:108-109`), is training-only. | **WRAP** (single classification — not simultaneously REUSE AS-IS) | The V1 file remains **unchanged** initially. `CategorizationService` (§7.3) wraps/reuses `predict_categories()`'s internal logic (feature-build + `kmeans.predict` + cluster→category mapping) by loading the artifact once at startup and calling those same steps directly against the cached artifact. `fit_and_evaluate()` stays an offline/admin-only entry point, not part of the request path. | No changes to `cluster.py` itself required — the service wraps around it rather than editing it, per PRD §9.3's "may be reused behind an abstraction." V2 must **not** call `predict_categories()` itself (its per-request `joblib.load`) from any request path. | Low — the risky part (repeated disk load) is avoided by *not* calling `predict_categories()` directly from the hot path; the service reimplements its three inner steps using the once-loaded artifact. |
| `pipeline/forecast.py` | `fit_and_forecast(df)` does aggregation, **walk-forward validation** (full historical re-evaluation), **GridSearchCV** if MAPE > 15% (`forecast.py:281-299`), final fit, and 3-month prediction — all in one call, every time (`forecast.py:252-364`). No persistence inside the function. | **REFACTOR** | Split into pieces usable by an interactive `POST /api/forecasts/run`: aggregation + final-fit + predict must be reachable *without* mandatorily running walk-forward + GridSearchCV on every user click (§12.3). | Extract `aggregate_monthly()` and `build_forecast_features()` (already standalone functions — no change needed). Add a leaner `train_and_predict(monthly_df)` path used by the interactive endpoint that fits once with fixed default hyperparameters (no walk-forward, no GridSearchCV) and returns forecasts. Walk-forward validation + GridSearchCV remain available as the existing `fit_and_forecast()` / `__main__` diagnostic path for **offline** MAPE evaluation, not the user-triggered path. | Medium — this is the highest-risk refactor; see §12.1–§12.3 for full reasoning and the explicit flag that the current `fit_and_forecast()` is not a safe default for a synchronous HTTP request. |
| `pipeline/portfolio.py` | `fetch_price()` is cache-first with a 1-hour TTL baked into `get_cached_price()` (`portfolio.py:24-43`); `build_portfolio()` **fetches AND inserts to DB in one call** (`portfolio.py:77-108`), coupling price-fetch with persistence. | **REFACTOR** | Separate "get current holdings + cached prices" (no network) from "fetch fresh prices" (network + cache update), per PRD §9.7/§11.9 manual-only refresh. | `fetch_price(conn, ticker)` is reused as-is for the *refresh* path. A new read path (`PortfolioService.get_holdings_with_prices()`) reads holdings + `price_cache` directly, **never calling `fetch_price()`**, so opening the page cannot trigger a network call. `build_portfolio()`'s DB-insert-inside-fetch coupling is not reused for V2 CRUD holdings — V2 holdings persistence goes through `HoldingRepository` instead (§8). | Medium — must ensure no V2 code path calls `build_portfolio()` (which persists) from a read-only request. |
| `db/database.py` | `get_connection()` re-runs `schema.sql` on every open (`database.py:19-25`) — fine for V1's single small schema. All inserts are append-only (`insert_transactions`, `insert_portfolio`, `insert_forecast_vs_actual`); only `monthly_summary` and `price_cache` use `INSERT OR REPLACE`. No `UPDATE`/`DELETE` for transactions. Queries (`get_transactions`, etc.) are unfiltered — return all rows regardless of session (`database.py:176-193`). | **LEGACY / PRESERVE** for V1; **REPLACE** for V2 | V2 gets its own repository layer (§8) against a separate schema/DB file (§18.2) — `db/database.py` is not imported by V2 code. | None to this file. New `backend/repositories/*` implement CRUD, filtering, and constraints from scratch against the V2 schema defined by `db/migrations/001_initial_v2.sql` (§4.12). | None — V1 file untouched, V2 does not depend on it. |
| `db/schema.sql` | 6 tables, `session_id`-scoped, no transaction UNIQUE/foreign keys (`schema.sql:1-71`). | **LEGACY / PRESERVE** | Superseded conceptually by `db/migrations/001_initial_v2.sql` (§4.11, §4.12); V1 file untouched. | None. | None. |
| `db/seed_synthetic_data.py` | Standalone, idempotent (clears tables via `DELETE FROM`, `seed_synthetic_data.py:186-189`), generates 12 months of transactions/summaries/predictions/forecast_vs_actual/portfolio against **V1's schema** via `db/database.py` helpers. | **LEGACY / PRESERVE** for V1; a **NEW** V2-specific demo-data script is written for `schema_v2.sql` (§14.1), reusing this script's *data-generation patterns* (merchant lists, amount ranges, seasonal multipliers) but calling V2 repositories instead of `db/database.py`. | Demo data for the V2 MVP (PRD §10a). | New file, e.g. `backend/scripts/seed_v2_demo_data.py`; not a change to the V1 file. | None to V1 file. |
| `viz/report.py` | `generate_report(conn)` takes a raw V1 `sqlite3.Connection` and unfiltered query helpers (`report.py:236-247`). | **LEGACY / PRESERVE** | Not part of V2 MVP (PRD §17/§11.10). Left importable/runnable against the V1 DB exactly as today. | None. | None — no V2 code touches it. |
| `viz/powerbi_export.py` | `export_powerbi_csvs(conn)` filters transactions/forecasts/portfolio to the latest runtime `session_id` via regex (`powerbi_export.py:22-29`), but `forecast_accuracy` export is unfiltered (`powerbi_export.py:104-113`). | **LEGACY / PRESERVE** | Same as above — post-MVP, untouched. | None. | None. |
| `config.py` | Paths, `CATEGORIES` (8 names), `BANK_DATE_FORMATS`, `CHART_COLORS` (`config.py:1-49`). | **REUSE AS-IS (shared constant), WRAP (paths)** | `CATEGORIES` and `BANK_DATE_FORMATS` are imported as-is by V2 backend code so taxonomy/date-format logic is never duplicated. V2-specific config (DB path, CORS origins, etc.) lives in `backend/config.py`, which imports from root `config.py` rather than editing it. | None to `config.py` itself. | Low. |
| `main.py` | Batch orchestrator: ingest → cluster → forecast → portfolio → DB writes → report → PowerBI export, all against V1 schema (`main.py:1-168`). | **LEGACY / PRESERVE** | Remains the V1 entry point; not part of the V2 runtime; not deleted (§18.4). | None. | None. |

---

## 4. V2 Database / Persistence Design

### 4.1 Transactions

**Stored columns** (justification inline):

- `id` — surrogate PK.
- `date` (TEXT, `YYYY-MM-DD`) — matches V1's normalized string convention (`ingest.py:130`).
- `raw_description` (TEXT) — the pre-clean merchant string, kept because the PRD's Import Batch concept (§12 of PRD) implies traceability back to what the bank actually sent, and because dedup/troubleshooting benefits from it; V1 discards this (`ingest.py:113` renames columns and never keeps the original).
- `merchant` (TEXT) — cleaned/normalized, as V1 produces today.
- `amount` (REAL).
- `bank_source` (TEXT) — which bank format produced this row; needed to keep categorization/taxonomy bank-agnostic while still tracing provenance (PRD §9.2).
- `predicted_category` (TEXT) — model output at creation/import time. Never overwritten after being set (PRD §9.3).
- `confirmed_category` (TEXT, nullable) — user's explicit choice; NULL until the user corrects/confirms.
- `import_batch_id` (INTEGER, nullable FK) — NULL for manually created transactions, set for imported ones.
- `data_mode` (TEXT, `'demo'` or `'real'`) — see §4.5; stored per-row because transactions are the entity every other read (dashboard, forecast aggregation) filters by, and per-row filtering is the simplest way to guarantee demo rows never enter a real aggregate query.
- `dedup_key` (TEXT) — see §4.4.
- `created_at`, `updated_at` (DATETIME).

**Not stored:**
- `effective_category` — **derived**, not a column. Computed as `COALESCE(confirmed_category, predicted_category)`. Recommendation: expose it via a **SQL VIEW** (`v_transactions_effective`, §4.11) rather than a generated column or application-only derivation, because (a) it needs to be usable directly in aggregate SQL (dashboard/forecast queries group by it), (b) SQLite's generated-column support (`GENERATED ALWAYS AS`) works but a view is simpler to reason about and equally fast for this table size, and (c) keeping it out of the base table means no migration is needed if the derivation rule ever changes.
- `is_manual_override` — **not stored**. It is fully derivable as `confirmed_category IS NOT NULL`. Storing it would create a second source of truth that could drift from `confirmed_category` (e.g., if a correction is cleared, an `is_manual_override` flag could be forgotten). The prompt suggested considering this column explicitly and *not* adding it merely because it was mentioned — that evaluation is exactly why it is omitted here.
- `cluster_id` (V1 concept) — not carried into V2's stable schema. It is an implementation detail of the *current* K-Means categorizer (PRD §9.3 requires the categorizer be replaceable without schema coupling). If the initial `CategorizationService` wants to persist it for its own debugging, that belongs in a service-owned, ML-implementation-specific side table — not the core `transactions` table. Deferred; not created in this TRD.

### 4.2 Import Batches

Columns: `id`, `bank_source`, `original_filename`, `status` (`'previewing'` | `'confirmed'` | `'failed'`), `data_mode` (`'demo'` | `'real'` — always `'real'` in practice, since demo data does not go through the import flow, but stored for consistency with §4.5's uniform model), `rows_valid`, `rows_unparseable`, `rows_duplicate`, `rows_imported`, `created_at`, `confirmed_at` (nullable).

**Valid status transitions:** `previewing → confirmed`, `previewing → failed` (e.g., preview expired or file corrupt at confirm time). No transition out of `confirmed` or `failed` — an import batch's outcome is immutable once resolved; a user who wants a fresh preview starts a new batch.

### 4.3 Import Preview Staging

Evaluated options:

| Option | Verdict |
|---|---|
| A. Server-side staging table(s) | **Chosen.** |
| B. Serialized staging payload tied to import_batch | Rejected — a large blob column duplicates what a staging table already models relationally, and makes duplicate re-check at commit (needs SQL) harder. |
| C. Temporary filesystem artifact | Rejected — adds filesystem lifecycle/cleanup concerns (orphaned temp files on crash) for no benefit over a DB table the app already has open. |
| D. Client-held preview data, server re-parses on confirm | Rejected — re-uploading/re-sending the full parsed row set from the browser on confirm is fragile (large payload, trusts the client's view of duplicate/validity flags) and reintroduces the exact "revalidate at commit" work a staging table gives for free. |

**Chosen design:** a `staged_transactions` table holds one row per parsed CSV row for a given `import_batch_id`, with columns mirroring the final transaction shape plus `is_duplicate` (BOOLEAN, computed at preview time) and `is_valid` (BOOLEAN). `POST /api/imports` parses the upload, writes to `staged_transactions`, and returns a summary (never touching `transactions`). `POST /api/imports/{batch_id}/confirm` **re-runs duplicate detection** against the current `transactions` table (not just trusting the preview's flags — see next paragraph), copies valid non-duplicate staged rows into `transactions` with `predicted_category` set via `CategorizationService`, marks the batch `confirmed`, and deletes the batch's `staged_transactions` rows (cleanup).

**Staleness between preview and commit:** because confirm re-checks duplicates live, a transaction imported by a *different* batch after this preview was generated (unlikely in a single-user app, but possible across two open tabs) is still caught. **Server restart** between preview and confirm: `staged_transactions` persists in SQLite (not memory), so a restart does not lose the preview — confirm still works. A `previewing` batch has no expiry in the MVP (no PRD requirement demands one); this is noted as low-risk given single-user usage and left out of scope rather than adding an unrequested TTL/cleanup job (§19 non-goals: no background jobs).

### 4.4 Duplicate Detection

Evaluated options:

| Option | Verdict |
|---|---|
| A. `UNIQUE(date, amount, merchant, bank_source)` DB constraint | **Chosen**, with the caveat below. |
| B. Canonical-row SHA-256 hash + DB uniqueness | Rejected for MVP — functionally equivalent to A once description is normalized, but adds a hashing step with no benefit at this data scale; revisit if merchant-normalization fields multiply. |
| C. Application-only duplicate checking (no DB constraint) | Rejected — a DB constraint is the only mechanism that also protects manually created transactions and concurrent imports from ever landing as literal duplicates, not just newly-imported CSV rows. |
| D. Another approach | Not needed given A covers the required cases below. |

**Reasoning against the specific edge cases:**
- **Exact duplicate re-import** (same CSV twice): caught by `(date, amount, merchant, bank_source)` matching exactly — this is the primary case the PRD requires (§9.2a).
- **Same transaction appearing in overlapping exports:** same key match, same outcome.
- **Legitimate same-day/same-amount separate purchases** (e.g., two identical $4.50 coffees): these will share `(date, amount, merchant, bank_source)` and be flagged as a false-positive duplicate under a naive unique constraint. **Resolution:** the `dedup_key` is not a bare tuple but `date + amount + merchant + bank_source + occurrence_index`, where `occurrence_index` is the row's 0-based position among identical-looking rows *within the same import batch or existing table, in file order*. Two genuinely identical purchases on the same statement occupy occurrence_index 0 and 1 respectively and are both kept; a true re-import of the same file reproduces the same occurrence_index sequence and every row collides. This is a pragmatic MVP heuristic, not a perfect one — flagged as a known limitation (§22.F item 2).
- **Merchant normalization / raw vs. normalized description:** dedup uses the *normalized* `merchant` (same normalization V1 already applies in `ingest.py:133-136`), not `raw_description`, so trivial formatting differences between two exports of the same statement don't defeat matching.
- **Near-duplicates** (e.g., amount off by a cent due to a bank rounding difference): explicitly **not** treated as duplicates in the MVP — only exact key matches are suppressed. Flagged as a known limitation, not solved here (over-engineering a fuzzy-match heuristic is not justified by the PRD).
- **Manually created transactions:** also assigned a `dedup_key` at creation time and checked against existing rows, so a user who manually re-enters a transaction they already imported gets the same duplicate protection (consistent with PRD §9.3's "same model" requirement extending to validation, not just categorization).
- **DB constraint vs. service-level detection:** the constraint is `UNIQUE(dedup_key)` at the DB level (belt), and the import/creation services also pre-check via `TransactionRepository.exists_by_dedup_key()` (suspenders) so a caught duplicate can be reported back to the user as "skipped as duplicate" (HTTP 200 with a count) rather than surfaced as a raw constraint-violation error.

### 4.5 Demo / Real Data Isolation

Evaluated options:

| Option | Verdict |
|---|---|
| A. Per-transaction `data_mode` | **Chosen for transactions**, combined with... |
| B. Separate demo tables | Rejected — doubles the schema (and every query) for no isolation benefit beyond what a flag gives; harder to keep dashboard/forecast queries consistent across both table sets. |
| C. `is_demo`/`data_mode` on import batches only | Rejected alone — demo data is not created via the import flow, and forecasts/holdings are not import-batch-scoped, so this alone can't cover the full demo surface the PRD requires (dashboard, forecast, portfolio). |
| D. Application-level data mode: `EMPTY \| DEMO \| REAL` | **Chosen**, as the authoritative gate, in combination with A. |
| E. Other | Not needed. |

**Chosen design:** a single-row `app_state` table stores the current mode (`EMPTY`, `DEMO`, or `REAL`), maintained by `DemoService` (§7.7) as the source of truth for "can I load demo data" / "am I allowed to import real data" decisions. In addition, `data_mode` is stamped on **every row** that demo-loading creates or that real usage creates — `transactions.data_mode`, `holdings.data_mode`, `forecast_runs.data_mode` — so that a query never has to trust the global flag alone to exclude demo data from a real aggregate (defense in depth: even a future bug in `app_state` tracking cannot make demo rows leak into a real dashboard query, because every read query filters `WHERE data_mode = 'real'`, or conversely `= 'demo'` when explicitly viewing demo state).

Price cache is **not** flagged per-row: cached prices are commodity market data (a stock's price is the same whether looked up for a demo or real holding), so demo holdings reuse the same `price_cache` table as real holdings — no isolation concern there.

**Mechanics:**
- **Demo load:** allowed only when `app_state.mode = 'EMPTY'`. `DemoService.load_demo()` inserts demo transactions/holdings/forecast run(s) all stamped `data_mode='demo'`, then sets `app_state.mode = 'DEMO'`.
- **Demo clear:** `DemoService.clear_demo()` deletes all rows across `transactions`, `holdings`, `forecast_runs`/`forecast_predictions`, `price_cache` entries used only by demo holdings, and `import_batches`/`staged_transactions` (there should be none, since demo bypasses import) where `data_mode='demo'`, then sets `app_state.mode = 'EMPTY'`.
- **Demo → real transition:** attempting a real import while `app_state.mode = 'DEMO'` returns a structured conflict (HTTP 409) asking for confirmation; on confirmation the client calls `DELETE /api/demo/clear` first (full reset per above, which also returns `app_state.mode` to `'EMPTY'`), then the real import proceeds normally per the **EMPTY → REAL transition rule** below.
- **Real → demo rejected:** `load_demo()` checks `app_state.mode`; if it is `'REAL'`, returns HTTP 409 and does **not** offer a force option that deletes real data (PRD §9.2b, TRD constraint #5).
- **How pages know the mode:** `GET /api/health` and `GET /api/demo/status` both return the current `app_state.mode`; the frontend keeps this in a small global store (§9.7) so the Dashboard/Import/Portfolio pages can show a demo banner or the confirm-clear-demo modal without each page independently querying transactions to infer it.

#### 4.5.1 EMPTY is an application state, never a row `data_mode`

`app_state.mode` has three values (`EMPTY`, `DEMO`, `REAL`), but the row-level `data_mode` column on `transactions`/`holdings`/`forecast_runs` has only two possible values (`'demo'`, `'real'`) — there is no `'empty'` row value, because `EMPTY` describes the *absence* of rows, not a category of row. No query ever filters `WHERE data_mode = 'EMPTY'`; that condition can never match anything and would be a bug if written.

**Canonical read mapping**, used consistently by every repository/service that filters by mode:

| `app_state.mode` | What reads return |
|---|---|
| `EMPTY` | Valid, empty collections — `[]` for transaction/holding lists, `null`/`{status: "no_forecast_yet"}` for the forecast, `{total_spend_current: 0, ...}` shaped (not missing) dashboard summary. No `WHERE data_mode = ...` clause is applied, because there are no rows of either mode to exclude — the emptiness comes from the table genuinely having zero rows, not from a filter. |
| `DEMO` | `WHERE data_mode = 'demo'` on every relevant table. |
| `REAL` | `WHERE data_mode = 'real'` on every relevant table. |

**EMPTY → REAL transition — precise definition:** the transition happens at the moment the **first real row is durably committed**, specifically any one of:
- a TD import commit (`POST /api/imports/{batch_id}/confirm`) that persists **≥1** real transaction row, or
- a successful manual transaction creation (`POST /api/transactions`), or
- a successful holding creation (`POST /api/holdings`).

Each of these three write paths, immediately after its own durable insert succeeds (same DB transaction or the next statement within the same request), checks `app_state.mode` and — only if it is currently `EMPTY` — sets it to `'REAL'`. If `app_state.mode` is already `'REAL'`, this is a no-op (idempotent).

**If the attempted write fails before persistence** (validation error, 503 from a missing categorization model, a parse failure, a DB constraint violation): no row was durably created, so `app_state.mode` is not touched and remains `EMPTY`.

**If the write itself succeeds but a later, optional downstream action fails** (example: a manual transaction is created successfully, but the forecast-staleness hook that runs afterward throws): the mode transition already happened at the point of durable real-data creation and is **not rolled back** by the downstream failure. The rule is "mode follows successful durable real-data creation," not "mode follows the entire request succeeding end-to-end." This matters concretely for `IngestionService.commit_import()`: the transaction-insert step and the mode-transition check happen together as the "real work" of the request; the subsequent `ForecastService.mark_stale()` call (§10, §12.4) is downstream/optional and its failure does not un-transition `app_state.mode` back to `EMPTY`, nor does it prevent `ImportResult` from correctly reporting the transactions that were, in fact, persisted.

### 4.6 Forecast Runs and Predictions

Two tables:

**`forecast_runs`**: `id`, `generated_at`, `months_available`, `months_required` (constant 12 for MVP, stored for auditability), `is_stale` (BOOLEAN), `stale_reason` (TEXT, nullable), `data_mode`, `model_impl_version` (TEXT — a free-text tag like `"kmeans_v1+rf_v1"`, so a future model swap is visible in history without the schema needing to know what the tag means).

**`forecast_predictions`**: `id`, `forecast_run_id` (FK), `category`, `forecast_month` (`YYYY-MM`), `month_offset` (1/2/3), `predicted_amount` (REAL, nullable), `is_available` (BOOLEAN), `unavailable_reason` (TEXT, nullable).

**Uniqueness:** `UNIQUE(forecast_run_id, category, forecast_month)` — scoped to the run, not global, so multiple retained runs can each have their own prediction for, e.g., category=`Shopping`, forecast_month=`2026-10`, without conflict. This is the exact structure the authoring prompt flagged as necessary and explicitly warned against getting wrong.

**Staleness storage:** stored (`is_stale` column), not derived, because "what changed since generation" is a point-in-time fact that must survive later mutations being reversed (e.g., a transaction edited then edited back should not un-mark staleness — the forecast is still stale relative to what it was computed from). See §12.4 for the full mutation-list/orchestration reasoning; this section only fixes where the flag lives.

### 4.7 Forecast-vs-Actual

**Decision: deferred entirely from the MVP schema.** PRD §17 places forecast-vs-actual tracking post-MVP, and no MVP acceptance criterion (PRD §19) requires it. Including an empty, unused table now would be schema surface with no MVP purpose (§22.E orphan-table check). When this feature is built post-MVP, its key must be `UNIQUE(forecast_run_id, category, forecast_month)` or equivalent — explicitly **not** `UNIQUE(category, forecast_month)` — so it can evaluate multiple retained historical runs, per the authoring prompt's explicit warning. That requirement is recorded here for the future feature, not implemented now.

### 4.8 Holdings and Price Cache

**`holdings`**: `id`, `ticker`, `shares` (REAL), `avg_cost` (REAL), `data_mode`, `created_at`, `updated_at`. No `current_price`/`pnl` columns — those are derived at read time by joining `price_cache` (avoids storing a value that goes stale the instant it's written).

**`price_cache`**: `id`, `ticker` (UNIQUE), `current_price`, `fetched_at`. **Only the latest observation per ticker is stored** (V1's existing shape, `schema.sql:40-46`) — the PRD requires only "latest/last-known" (§9.7), and historical price series has no MVP consumer. If per-ticker history becomes useful later (e.g., a portfolio value trend chart), that's a new table, not a change to this one.

Opening Portfolio reads `holdings JOIN price_cache` regardless of `fetched_at` age — there is no TTL gate on the *read* path (§13.3). The TTL concept from V1 (`portfolio.py:21`, 1-hour) is retained **only inside `refresh_prices()`** as an optional optimization (skip re-fetching a ticker refreshed in the last hour even during an explicit refresh) — not as a gate on whether a price is *displayable*. This directly implements PRD §9.7/§13.3's "an old cached price should remain displayable."

### 4.9 Monthly Summary

**Decision: compute dashboard/monthly aggregates live from `transactions` via SQL, do not materialize.** V1's `monthly_summary` table exists because the batch pipeline computes it once per run and wants to display it later without re-running the pipeline. In V2, transactions are queried directly and cheaply (single-user, thousands of rows at most — not millions), so a live `GROUP BY` query is simpler, always consistent (no upsert-timing bugs), and avoids a second source of truth for numbers the dashboard needs. `monthly_summary` is **not carried into `schema_v2.sql`**. If a future version needs materialization for performance at much larger data volumes, that is an optimization to revisit with actual evidence — not assumed here (§19 non-goals implicitly cover premature optimization).

### 4.10 Indexes / Constraints

- `transactions`: PK `id`; `UNIQUE(dedup_key)`; FK `import_batch_id → import_batches(id)` ON DELETE SET NULL (deleting a batch record — which the MVP never does — should not cascade-delete real transactions); indexes on `date` (dashboard date-range queries), `data_mode` (every read filters by it), and `(data_mode, date)` composite for the common "real transactions in date range" query.
- `import_batches`: PK `id`; index on `status`.
- `staged_transactions`: PK `id`; FK `import_batch_id → import_batches(id)` ON DELETE CASCADE (deleting a batch's staging rows is exactly the cleanup step at confirm time).
- `forecast_runs`: PK `id`; index on `(data_mode, generated_at DESC)` for "get latest run" queries.
- `forecast_predictions`: PK `id`; FK `forecast_run_id → forecast_runs(id)` ON DELETE CASCADE (a run's predictions have no meaning without the run); `UNIQUE(forecast_run_id, category, forecast_month)` per §4.6.
- `holdings`: PK `id`; index on `data_mode`.
- `price_cache`: PK `id`; `UNIQUE(ticker)` (matches V1).
- `app_state`: single-row table, PK fixed at `id = 1` (CHECK constraint enforces exactly one row).

### 4.11 Complete DDL

**Source-of-truth note:** the DDL below is the content of `db/migrations/001_initial_v2.sql` — the single authoritative source for V2's initial schema (§4.12). No separate `schema_v2.sql` file exists in the V2 repository structure; that earlier draft's parallel file would have created two competing sources of truth for the same DDL, which is removed here in favor of the simpler single-source design.

```sql
-- db/migrations/001_initial_v2.sql

CREATE TABLE IF NOT EXISTS app_state (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    mode            TEXT NOT NULL CHECK (mode IN ('EMPTY','DEMO','REAL')) DEFAULT 'EMPTY',
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT OR IGNORE INTO app_state (id, mode) VALUES (1, 'EMPTY');

CREATE TABLE IF NOT EXISTS import_batches (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_source         TEXT NOT NULL,
    original_filename   TEXT,
    status              TEXT NOT NULL CHECK (status IN ('previewing','confirmed','failed')),
    data_mode           TEXT NOT NULL CHECK (data_mode IN ('demo','real')) DEFAULT 'real',
    rows_valid          INTEGER NOT NULL DEFAULT 0,
    rows_unparseable    INTEGER NOT NULL DEFAULT 0,
    rows_duplicate      INTEGER NOT NULL DEFAULT 0,
    rows_imported       INTEGER NOT NULL DEFAULT 0,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    confirmed_at        DATETIME
);
CREATE INDEX IF NOT EXISTS idx_import_batches_status ON import_batches(status);

CREATE TABLE IF NOT EXISTS staged_transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    import_batch_id     INTEGER NOT NULL REFERENCES import_batches(id) ON DELETE CASCADE,
    date                TEXT NOT NULL,
    raw_description     TEXT,
    merchant            TEXT NOT NULL,
    amount              REAL NOT NULL,
    predicted_category  TEXT,
    dedup_key           TEXT NOT NULL,
    is_duplicate        INTEGER NOT NULL DEFAULT 0,
    is_valid            INTEGER NOT NULL DEFAULT 1,
    invalid_reason      TEXT
);
CREATE INDEX IF NOT EXISTS idx_staged_txn_batch ON staged_transactions(import_batch_id);

CREATE TABLE IF NOT EXISTS transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    date                TEXT NOT NULL,
    raw_description     TEXT,
    merchant            TEXT NOT NULL,
    amount              REAL NOT NULL,
    bank_source         TEXT,
    predicted_category  TEXT NOT NULL,
    confirmed_category  TEXT,
    import_batch_id     INTEGER REFERENCES import_batches(id) ON DELETE SET NULL,
    data_mode           TEXT NOT NULL CHECK (data_mode IN ('demo','real')),
    dedup_key           TEXT NOT NULL,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dedup_key)
);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_transactions_data_mode ON transactions(data_mode);
CREATE INDEX IF NOT EXISTS idx_transactions_mode_date ON transactions(data_mode, date);

CREATE VIEW IF NOT EXISTS v_transactions_effective AS
SELECT
    t.*,
    COALESCE(t.confirmed_category, t.predicted_category) AS effective_category,
    (t.confirmed_category IS NOT NULL) AS is_manual_override
FROM transactions t;

CREATE TABLE IF NOT EXISTS forecast_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    months_available    INTEGER NOT NULL,
    months_required     INTEGER NOT NULL DEFAULT 12,
    is_stale            INTEGER NOT NULL DEFAULT 0,
    stale_reason        TEXT,
    data_mode           TEXT NOT NULL CHECK (data_mode IN ('demo','real')),
    model_impl_version  TEXT
);
CREATE INDEX IF NOT EXISTS idx_forecast_runs_mode_time ON forecast_runs(data_mode, generated_at DESC);

CREATE TABLE IF NOT EXISTS forecast_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_run_id     INTEGER NOT NULL REFERENCES forecast_runs(id) ON DELETE CASCADE,
    category            TEXT NOT NULL,
    forecast_month      TEXT NOT NULL,
    month_offset        INTEGER NOT NULL CHECK (month_offset IN (1,2,3)),
    predicted_amount    REAL,
    is_available        INTEGER NOT NULL DEFAULT 1,
    unavailable_reason  TEXT,
    UNIQUE (forecast_run_id, category, forecast_month)
);

CREATE TABLE IF NOT EXISTS holdings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT NOT NULL,
    shares              REAL NOT NULL CHECK (shares > 0),
    avg_cost            REAL NOT NULL CHECK (avg_cost >= 0),
    data_mode           TEXT NOT NULL CHECK (data_mode IN ('demo','real')),
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_holdings_data_mode ON holdings(data_mode);

CREATE TABLE IF NOT EXISTS price_cache (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT NOT NULL,
    current_price       REAL NOT NULL,
    fetched_at          DATETIME NOT NULL,
    UNIQUE (ticker)
);
```

### 4.12 Migration Strategy

- **`db/migrations/` is the single authoritative source of the V2 schema.** `001_initial_v2.sql` (== the DDL in §4.11) defines the full initial schema; `002_....sql` etc. for any later change. There is no separate `schema_v2.sql` file to keep in sync — removing that earlier duplication eliminates a source-of-truth risk where the two files could silently drift apart.
- **Tracking:** a `schema_migrations(version INTEGER PRIMARY KEY, applied_at DATETIME)` table records which numbered files have run.
- **Applying:** on FastAPI startup (lifespan), the backend checks `schema_migrations` for the highest applied version and runs any `.sql` files numerically greater than it, in order, each inside a transaction.
- **Rollback:** none for MVP — a local SQLite file backed by a single developer's machine does not need automated rollback; the practical rollback is restoring a file backup. A heavier framework (Alembic) is not justified for a handful of forward-only DDL files in a single-developer local MVP (§19).

---

## 5. REST API Contracts

All responses use the error envelope in §15 on failure. All list endpoints implicitly filter to `data_mode = current app_state.mode`'s relevant value (real endpoints show real data; demo-mode viewing is the same UI, just showing rows currently in `'demo'` mode — there's only ever one active mode at a time per §4.5).

### 5.1 Health
`GET /api/health` → `{db: "ok", categorization_model: "loaded"|"missing"|"error", data_mode: "EMPTY"|"DEMO"|"REAL"}`. No row counts, no file paths, no stack traces — enough to diagnose, not enough to leak internals.

### 5.2 Demo
- `POST /api/demo/load` — 200 + summary on success; 409 if `mode == 'REAL'`.
- `DELETE /api/demo/clear` — 200 on success (idempotent: 200 even if already empty); full reset per §4.5.
- `GET /api/demo/status` → `{mode, can_load_demo: bool}`.
- Confirmation semantics for demo→real: handled client-side (§5.3 import flow) — `POST /api/imports` on a real bank while `mode == 'DEMO'` returns 409 with `{error: "demo_conflict", message: "..."}`; frontend shows the confirm dialog, then calls `DELETE /api/demo/clear` followed by retrying the import.

### 5.3 Imports
- `POST /api/imports` (multipart file + bank field, MVP: bank is always `"TD"`, sent for forward compatibility not user choice) — parses, stages, returns `ImportPreview`. Does not commit. Returns 409 if `mode == 'DEMO'` (see above).
- `POST /api/imports/{batch_id}/confirm` — commits valid non-duplicate staged rows, returns `ImportResult`. Idempotent: calling confirm twice on an already-`confirmed` batch returns the original result (200) rather than re-inserting or erroring, since nothing was actually attempted twice.
- `GET /api/imports`, `GET /api/imports/{batch_id}` — list/detail, for the user's own import history (PRD §12 Import Batch).

*PRD basis:* §11.2, §11.3, §9.2a.

### 5.4 Transactions
- `GET /api/transactions?date_from=&date_to=&category=&search=&sort=&page=&page_size=` — reads `v_transactions_effective`.
- `POST /api/transactions` — manual creation; calls `CategorizationService` before persisting (constraint #6); 503 if model unavailable.
- `GET /api/transactions/{id}`, `PATCH /api/transactions/{id}`, `DELETE /api/transactions/{id}`.

*PRD basis:* §11.4, §11.5, §11.6.

### 5.5 Categories
`GET /api/categories` → fixed array of 8 strings from `config.CATEGORIES`. *PRD basis:* §9.4.

### 5.6 Forecasts
- `GET /api/forecasts/latest` — DB read only, never trains. 200 with `ForecastRunResponse`, or 200 with `{status: "no_forecast_yet"}` if none exists.
- `GET /api/forecasts/status` — 200 always; body is `{status: "ready"|"cold_start", months_available, months_required}` plus staleness info if a run exists. Cold start is HTTP 200 per constraint #3, never 422/500.
- `POST /api/forecasts/run` — the **only** endpoint that trains/generates; creates a new retained run. 200 on success, 422 if cold-start (insufficient months) prevents generation entirely — this is a deliberate exception to "cold start is never an error status": *attempting to generate* when ineligible is a rejected action (422, similar to any other invalid-state write), whereas *checking status/viewing* cold-start is a normal read (200). This distinction is called out explicitly in §15.

*PRD basis:* §9.6, §11.8, constraints #1–#3.

### 5.7 Holdings
- `GET /api/holdings` — reads cache only, never fetches.
- `POST /api/holdings`, `PATCH /api/holdings/{id}`, `DELETE /api/holdings/{id}`.
- `POST /api/holdings/refresh-prices` — the only endpoint that calls `yfinance`.

*PRD basis:* §9.7, §11.9, constraint #8.

### 5.8 Dashboard
`GET /api/dashboard/summary` — pure DB reads across transactions/forecasts/holdings; no ML, no network. Returns current-vs-previous-calendar-month totals, category breakdown, trend, recent transactions, forecast summary (if a run exists), portfolio summary (if holdings exist), and `data_mode`.

*PRD basis:* §11.7.

---

## 6. Pydantic / API Data Schemas

Representative shapes (fields only; exact Pydantic syntax is an implementation detail for the Build Plan):

- **TransactionCreate**: `date`, `merchant`, `amount`, `confirmed_category?` (optional — if provided, becomes `confirmed_category` immediately; `predicted_category` is still computed and stored per §9.3/constraint #6).
- **TransactionUpdate**: any of `date`, `merchant`, `amount`, `confirmed_category` (partial update).
- **TransactionResponse**: `id`, `date`, `merchant`, `raw_description`, `amount`, `bank_source`, `predicted_category`, `confirmed_category`, `effective_category` (computed, always present), `is_manual_override` (computed, from the view), `created_at`, `updated_at`. `is_demo` is **not** exposed per-transaction in the response — the frontend gets the mode once from `/api/health` or `/api/demo/status`, since every transaction currently visible shares the same mode by construction (§4.5), so a per-row flag would be redundant information.
- **ImportPreview**: `batch_id`, `rows_valid`, `rows_unparseable`, `rows_duplicate`, `date_range: {from, to}`, `sample_rows` (first ~10), `status`.
- **ImportResult**: `batch_id`, `rows_imported`, `rows_skipped_unparseable`, `rows_skipped_duplicate`, `status`.
- **ForecastStatusResponse**: `status` (`"ready"|"cold_start"|"no_forecast_yet"`), `months_available`, `months_required`, `latest_run_id?`, `is_stale?`.
- **ForecastRunResponse**: `run_id`, `generated_at`, `is_stale`, `stale_reason?`, `months_available`, `predictions: ForecastPrediction[]`.
- **ForecastPrediction**: `category`, `forecast_month`, `month_offset`, `predicted_amount?`, `is_available`, `unavailable_reason?`.
- **HoldingCreate/Update**: `ticker`, `shares`, `avg_cost`.
- **HoldingResponse**: `id`, `ticker`, `shares`, `avg_cost`, `current_price?`, `current_value?`, `pnl?`, `price_last_updated?`, `price_fetch_error?`.
- **DashboardSummary**: `period: {current, previous}`, `total_spend_current`, `total_spend_previous`, `change_pct`, `category_breakdown`, `spending_trend`, `recent_transactions`, `forecast_summary?`, `portfolio_summary?`, `data_mode`.
- **HealthResponse**: `db`, `categorization_model`, `data_mode`.
- **ErrorResponse** (§15): `error`, `message`, `details?`.

Fields the persistence design rejected (`cluster_id`, per-transaction `is_demo`) are not present in any schema above.

---

## 7. Service Layer

**Rule enforced throughout:** repositories never decide that another domain entity should change; only services orchestrate across domains.

### 7.1 IngestionService
`parse_and_stage(file_bytes, bank) -> ImportPreview`, `commit_import(batch_id) -> ImportResult`. Coordinates: `pipeline.ingest.load_and_clean_from_bytes`, dedup check (via `TransactionRepository.exists_by_dedup_key`), `CategorizationService.predict_batch`, `ImportBatchRepository`/staged-row writes, and — on successful commit — calls `ForecastService`'s staleness hook (not forecast generation). Does not import/call `pipeline.forecast` at all. Immediately after durably inserting ≥1 real transaction row, checks `AppStateRepository.get_mode()` and transitions `EMPTY → REAL` if applicable (§4.5.1); this transition is not undone if the subsequent staleness-hook call fails.

### 7.2 TransactionService
`create_manual(data) -> Transaction`, `update(id, data) -> Transaction`, `delete(id)`. Every manual create runs through `CategorizationService` first (constraint #6). Immediately after a successful `create_manual` insert, checks and transitions `EMPTY → REAL` per §4.5.1 (a failed creation, e.g. a 503 from a missing model, leaves the mode untouched). Determines whether a given mutation is forecast-relevant and, if so, calls `ForecastService.mark_stale(reason)`:

| Mutation | Marks latest forecast stale? | Why |
|---|---|---|
| Manual transaction creation | Yes | Changes aggregate monthly totals used by forecasting. |
| Deletion | Yes | Same reason. |
| Amount edit | Yes | Changes monthly totals. |
| Date edit | Yes | Can move a transaction into a different month bucket. |
| Confirmed/effective category change | Yes | Changes which category's monthly total the amount counts toward. |
| Merchant edit | **No** | Merchant text does not feed forecast aggregation (`aggregate_monthly` groups by `month`/`category` only, `forecast.py:22-50`) — only categorization uses merchant, and correcting a merchant string does not, by itself, change the effective category. **Decision, not re-prediction:** editing merchant text does **not** trigger re-categorization either — the PRD's manual-correction workflow (§9.3) is the only sanctioned path to change a category; silently re-predicting on a merchant edit could overwrite a user's prior confirmed category without their intent, which the PRD explicitly guards against ("must not silently overwrite original model predictions"). |

Successful import commit (via `IngestionService`, not directly) also triggers the same staleness hook.

### 7.3 CategorizationService
Loaded once at FastAPI startup (lifespan hook), not per-request. Interface: `predict(transaction: {merchant, amount, date}) -> {predicted_category}` and a batch variant `predict_batch(rows)`. Internally, on startup, loads `kmeans_model.pkl` via `joblib.load` **once** and keeps `kmeans`, `scaler`, `vectorizer`, `cluster_to_category` in memory; `predict()` calls `features.build_feature_matrix(..., fit=False)` + `kmeans.predict` + mapping directly — i.e., it reimplements `predict_categories()`'s three inner steps against the cached artifact rather than calling `predict_categories()` itself (which would reload from disk every time, `cluster.py:160-162`). Status: `loaded`/`missing`/`error`, reported via `/api/health`. No confidence score is fabricated — K-Means/majority-vote mapping does not produce a defensible per-prediction confidence, so none is returned (constraint honored per §11.2 of the prompt).

### 7.4 ForecastService
`check_status() -> ForecastStatusResponse` (DB/aggregation read only — counts distinct months in real transactions, compares to `months_required=12`, checks latest run's `is_stale`). `get_latest() -> ForecastRunResponse | None` (DB read only). `run_forecast() -> ForecastRunResponse` (the only method that touches `pipeline.forecast`; internally aggregates, fits, predicts — see §12 — and persists a new run via `ForecastRepository`). `mark_stale(reason)` (called by other services, flips `is_stale` on the current latest run only — see §12.4 for why only latest). No public `train()`/`predict()` split is exposed — the stable boundary is read-latest vs. explicitly-generate, per the authoring prompt's explicit instruction.

### 7.5 PortfolioService
`get_holdings_with_prices() -> HoldingResponse[]` (reads `holdings` JOIN `price_cache`, never calls `yfinance`). `refresh_prices() -> {refreshed: [...], failed: [...]}` (calls `pipeline.portfolio.fetch_price` per ticker, updates cache, tolerates per-ticker failure). CRUD (`create_holding`, `update_holding`, `delete_holding`) delegates to `HoldingRepository` directly (simple enough not to need separate orchestration), except that `create_holding` also checks and transitions `EMPTY → REAL` per §4.5.1 immediately after a successful durable insert.

### 7.6 DashboardService
`get_summary() -> DashboardSummary`. Reads `TransactionRepository`, `ForecastRepository`, `HoldingRepository`/`PriceCacheRepository` only. No ML calls, no `yfinance` calls — this is explicitly checked in §22.H.

### 7.7 DemoService
`load_demo()`, `clear_demo()`, `status()`. Enforces the state-machine in §4.5: `load_demo()` raises a domain conflict (mapped to HTTP 409 by the route) if `app_state.mode == 'REAL'`; never deletes real data under any parameter.

---

## 8. Repository Layer

Repositories contain SQL only — no ML calls, no `yfinance` calls, no cross-domain orchestration (e.g., `TransactionRepository` never itself decides to touch `forecast_runs`).

- **TransactionRepository**: `create`, `get`, `list(filters)`, `update`, `delete`, `exists_by_dedup_key(key) -> bool`, `count_distinct_months(data_mode) -> int`, `aggregate_by_month_category(data_mode, date_range)`.
- **ImportBatchRepository**: `create_preview(...)`, `get(id)`, `list()`, `update_status(id, status, counts)`.
- **StagedTransactionRepository**: `bulk_create(rows)`, `list_for_batch(batch_id)`, `delete_for_batch(batch_id)`.
- **ForecastRepository**: `create_run(...)`, `save_predictions(run_id, predictions)`, `get_latest_run(data_mode)`, `get_run(run_id)`, `mark_run_stale(run_id, reason)`.
- **HoldingRepository**: `create`, `get`, `list(data_mode)`, `update`, `delete`.
- **PriceCacheRepository**: `get_last_known(ticker)`, `upsert_latest(ticker, price, fetched_at)`.
- **AppStateRepository**: `get_mode()`, `set_mode(mode)`.

---

## 9. Frontend Architecture

### 9.1 Setup
Standard Vite + React + TypeScript scaffold (`npm create vite@latest -- --template react-ts`). `vite.config.ts` configures the dev proxy (`/api` → `http://localhost:8000`). No packages are installed as part of writing this TRD.

### 9.2 Server State
**Recommendation: TanStack Query**, with plain `useState`/`useContext` for the small amount of genuinely global UI state (the current `data_mode` banner). Rationale: the app is read-heavy (dashboard, transaction list, forecast, holdings) with a handful of mutations that need cache invalidation (e.g., editing a transaction should refetch the dashboard) — this is exactly TanStack Query's core use case, without the boilerplate Redux Toolkit/RTK Query would add for an app this size.

### 9.3 API Client
A single typed `apiClient` module wraps `fetch`, parses the `ErrorResponse` envelope (§15) on non-2xx, and exposes typed functions (`getTransactions()`, `createTransaction()`, ...) that TanStack Query hooks call. Loading/error states come from Query's own `isLoading`/`isError`. Mutations call `queryClient.invalidateQueries` for affected keys (e.g., editing a transaction invalidates `["transactions"]` and `["dashboard"]`).

### 9.4 TypeScript Types
Hand-written TS interfaces in `frontend/src/types/` mirror the Pydantic schemas in §6 (e.g., `TransactionResponse`, `ForecastRunResponse`). Given the schema surface is small (roughly a dozen types) and stable once frozen, manual duplication is acceptable for the MVP; introducing OpenAPI-codegen tooling now would be infrastructure ahead of a demonstrated pain point (§19 — avoid unnecessary tooling).

### 9.5 Pages
As enumerated in the prompt — Dashboard, Transactions, Import, Forecast, Portfolio — each with the component breakdown listed in the prompt's §9.5 (SpendingOverview, TransactionTable, FileUpload/ImportPreview, ForecastChart/ColdStartState/StaleWarning, HoldingsTable/RefreshPricesButton, etc.), all reading from the API client in §9.3. No additional pages are introduced beyond the PRD's five (§11.1).

### 9.6 Routing
A lightweight router (React Router) with routes `/` (redirects to `/dashboard`), `/dashboard`, `/transactions`, `/import`, `/forecast`, `/portfolio` — matching PRD §11.1's five sections exactly, no more.

### 9.7 Demo State
A small `AppStateContext` fetches `/api/health` (or `/api/demo/status`) once on load and on relevant mutations, exposing `mode` to the rest of the app. Banner/UI behavior by mode (corrected from an earlier `mode !== 'REAL'` shorthand, which would incorrectly show a "demo" banner during the `EMPTY` state):

| `mode` | UI behavior |
|---|---|
| `DEMO` | Persistent, clearly visible "Demo Data" banner shown on every page. |
| `EMPTY` | No demo banner. Onboarding/empty-state UI only (the offer to import real data or load demo data, per PRD §10a) — `EMPTY` is not demo data and must not be labeled as such. |
| `REAL` | Normal app, no banner. |

The context also gates the Import page's confirm-clear-demo modal (triggered by a 409 from `POST /api/imports`, which only occurs when `mode === 'DEMO'`).

### 9.8 PRD UX Obligations

These are UX requirements the PRD imposes (§13 Validation & Error Behavior; §9.1–§9.9 principles) that the frontend design in §9.1–§9.7 must satisfy; this subsection collects them explicitly rather than leaving them implicit in prose elsewhere.

- **Predicted vs. corrected category must be visually distinguishable** in the Transactions UI — e.g., a "predicted" badge/style vs. a "confirmed" badge/style on the category chip, not merely inferable from a tooltip (PRD §13).
- **Effective category is what dashboard/forecast calculations use** — the Transactions UI should make clear that a corrected category becomes what "counts" everywhere else, consistent with `effective_category` (§4.1, §6).
- **Loading/progress states are required for:**
  - CSV preview and import confirmation (§10).
  - Explicit forecast generation/refresh (`POST /api/forecasts/run`).
  - Explicit portfolio price refresh (`POST /api/holdings/refresh-prices`).
- **Confirmation is required before destructive actions:**
  - Deleting a transaction.
  - Deleting a holding.
  - Clearing demo data (`DELETE /api/demo/clear`, whether user-initiated directly or as part of the demo→real flow, §4.5).
- **Demo data must remain clearly labeled while active** — the §9.7 banner is the primary mechanism; no page may show demo-mode data without it.
- **Portfolio UI must disclose that prices come from an external market-data source**, and must show each holding's last-updated timestamp (`price_last_updated`, §6) wherever a price is shown, including when it is stale/old (§13.3 — an old price is still displayed, with its age visible, not hidden).
- **Empty states are deliberate product states, not error screens** — the empty-collection responses defined in §4.5.1's read mapping (`EMPTY` mode) must render as an intentional "nothing here yet, here's what to do" UI on every page (Dashboard, Transactions, Forecast, Portfolio), never as a caught exception or blank crash state (PRD §11.12).

This subsection does not redesign the frontend beyond stating these obligations — page/component structure remains as defined in §9.5.

---

## 10. Import Pipeline Design

| Step | Owning service | Input → Output | Failure mode | Notes |
|---|---|---|---|---|
| Upload received | `IngestionService` (via route) | file bytes, bank='TD' | 400 if not a parseable CSV at all | If `app_state.mode == 'DEMO'`, short-circuits to 409 before parsing (§5.3). |
| Parse/normalize | `pipeline.ingest.load_and_clean_from_bytes` | bytes → cleaned DataFrame | per-row: unparseable dates dropped + counted (existing `ingest.py:122-127` behavior) | Whole-file failure (wrong file type entirely) → 400, not a 200 with all rows invalid. |
| Row validation | `IngestionService` | DataFrame → valid/invalid split | invalid rows counted, not fatal | Total-import does not fail because of some bad rows; only total-file parse failure is fatal. |
| Duplicate analysis | `IngestionService` + `TransactionRepository.exists_by_dedup_key` | valid rows → duplicate/non-duplicate split | none (this step cannot fail, only classify) | See §4.4. |
| Categorization | `CategorizationService.predict_batch` | non-duplicate valid rows → predicted_category per row | 503 if model unavailable — **preview itself still succeeds but reports that categorization could not run**, since preview is read-only and shouldn't block on the model; **confirm is where 503 is enforced** (constraint #6) since that's the actual write. | |
| Preview stored | `StagedTransactionRepository.bulk_create` | rows + flags → `staged_transactions` | DB error → 500 | Atomic per batch (single transaction). |
| Return ImportPreview | route | — | — | |
| **USER CONFIRMS** | — | — | — | |
| Re-validate/re-check | `IngestionService.commit_import` | staged rows → re-checked against **current** `transactions` (not just the stale preview flags) | if model became unavailable since preview, confirm returns 503 and commits nothing (all-or-nothing for the batch) | Repeated confirmation on an already-`confirmed` batch returns the original `ImportResult` (200), no re-insert (§5.3). |
| Persist valid, non-duplicate rows | `TransactionRepository.create` (bulk) | staged rows → `transactions` | DB constraint violation on a race (two tabs) → treated as a duplicate skip, not an error | Single DB transaction: either the whole batch's valid rows commit or none do. |
| Complete ImportBatch | `ImportBatchRepository.update_status` | → `status='confirmed'`, counts finalized | — | |
| Mark relevant forecast stale | `ForecastService.mark_stale` (called by `IngestionService`, not by the repository) | → `forecast_runs.is_stale = true` if a run exists | no-op if no run exists yet | **No forecast generation call anywhere in this flow.** |
| Return ImportResult | route | — | — | UI may show "your forecast may be stale — refresh it" per PRD wording, but this is a message only. |
| Staged rows cleaned up | `StagedTransactionRepository.delete_for_batch` | — | — | Cascade also handles this if the batch itself were ever deleted (not an MVP action). |

**Server restart between preview and confirm:** `staged_transactions` is durable SQLite, not memory — confirm works identically after a restart. **Repeated confirmation:** idempotent, per above.

---

## 11. Categorization Integration Design

### 11.1 Model Lifecycle
Loaded once in FastAPI's `lifespan` context manager at process startup (`backend/main.py`), stored on `app.state.categorization_service`. If `kmeans_model.pkl` is missing at startup, the service is constructed in a `missing` state (does not crash the whole app — `/api/health` reports it, dashboard/transactions/import for **existing** data remain fully readable) rather than raising during startup. Logged at `WARNING` on missing, `ERROR` on a load exception (corrupt file), `INFO` on successful load.

### 11.2 Stable Prediction Contract
`predict(transaction) -> {predicted_category: str}`. No confidence field — not fabricated for K-Means (§7.3 reasoning). This is the entire contract any future classifier must satisfy.

### 11.3 Import / Manual Creation
Both paths require a successful prediction before any `transactions` row is written. If the service is `missing`/`error` at confirm-time or manual-create-time: return HTTP 503 with a structured message (`{error: "categorization_unavailable", message: "..."}`); **no row is written with `predicted_category = NULL`.** Existing data remains fully readable throughout (constraint #6, honored literally).

### 11.4 Replacement Boundary
A future supervised classifier implements the same `predict(transaction) -> {predicted_category}` contract inside a new `CategorizationService` implementation. Because `TransactionResponse`, the REST routes, the repository schema, and the React components all depend only on `predicted_category`/`confirmed_category`/`effective_category` as plain strings — never on K-Means-specific concepts like cluster IDs or the majority-vote mapping — swapping the implementation requires no change outside `backend/services/categorization_service.py` and the model-loading step in `main.py`'s lifespan hook.

---

## 12. Forecast Integration Design

### 12.1 V1 Problem (inspected)
`fit_and_forecast()` (`forecast.py:252-364`) combines, in one call: aggregation (`aggregate_monthly`), full walk-forward validation across **every** historical month (`walk_forward_validate`, refitting a fresh `RandomForestRegressor` per test month — potentially a dozen or more fits), conditional `GridSearchCV` over a 3×3×3 parameter grid with `TimeSeriesSplit(n_splits=3)` if MAPE > 15% (`forecast.py:281-299` — up to 27 parameter combinations × 3 folds = 81 additional fits), a final fit, and prediction generation. This is unambiguously too slow and too heavy for a synchronous button click.

### 12.2 V2 Contract
`check_status()`, `get_latest()`, `run_forecast()` as specified in §7.4/§5.6. `get_latest()` and `check_status()` never call any of `pipeline.forecast`'s fitting code.

### 12.3 User-Path Performance
**Recommendation:** `run_forecast()` calls a new, leaner `train_and_predict(monthly_df)` (extracted from `fit_and_forecast()`, §3) that performs **aggregation → one fit with fixed default hyperparameters (`n_estimators=100, max_depth=10, min_samples_leaf=5`, the same defaults `forecast.py:276-279` already uses before any GridSearchCV) → prediction generation**, and **does not** run `walk_forward_validate()` or `GridSearchCV` on the user path. A single `RandomForestRegressor.fit()` on a few hundred monthly-aggregated rows (at most `8 categories × N months`) is expected to complete in well under a second — this is not flagged as a performance risk. Full walk-forward MAPE evaluation and GridSearchCV tuning remain available as an **offline** diagnostic (the existing `__main__` block / a future admin-only script), which is where the forthcoming ML Specification's evaluation work belongs — not the interactive path. If, once implemented, `train_and_predict` is still measurably slow, that must be flagged rather than solved by introducing a task queue (§19) — no evidence of that today, so no such infrastructure is added.

### 12.4 Staleness
Mutations evaluated (mirrors §7.2's table): manual creation, deletion, amount edit, date edit, confirmed/effective category change, and successful real import — all mark the current run stale. Merchant edit does not (§7.2 reasoning). **Orchestration lives in services** (`TransactionService`, `IngestionService` call `ForecastService.mark_stale()`); `ForecastRepository.mark_run_stale()` itself only executes the UPDATE it's told to make — it never inspects a transaction change and decides on its own to flip the flag (§7 rule, §22.H check).

**Only the latest non-stale run is marked stale** — older, already-superseded runs are historical record and their staleness state at time of superseding is not retroactively changed. Staleness is **stored**, not derived, per §4.6's reasoning (a later reversal of the triggering change should not silently un-stale a forecast that was, at one point, correctly marked).

### 12.5 Cold Start
Overall eligibility: `months_available >= 12` (`TransactionRepository.count_distinct_months`), matching V1's `aggregate_monthly` raise threshold (`forecast.py:46-48`) exactly, per constraint #8/PRD §21. Cold start surfaces as HTTP 200 with `{status: "cold_start", months_available, months_required: 12}` from `GET /api/forecasts/status` — never an exception. Per-category sparse history: `build_forecast_features`'s existing `dropna` on rolling/lag columns (`forecast.py:128`) already naturally excludes a too-sparse category from the trainable feature set; `run_forecast()` maps any category absent from the resulting predictions to `is_available: false, unavailable_reason: "insufficient_history"` rather than fabricating a zero. The exact statistical rule for what counts as "sufficient" per category beyond V1's existing 6-month rolling-window `dropna` behavior is **deferred to the ML Specification** — not safely derivable from V1 alone (flagged in §22.M).

### 12.6 Run Retention
Every successful `run_forecast()` call inserts a new `forecast_runs` row (never updates one in place); `get_latest()` reads the max `generated_at` for the current `data_mode`. Historical runs are never overwritten or deleted (except by `DemoService.clear_demo()` for demo-mode runs specifically).

---

## 13. Portfolio Design

### 13.1 Holdings
Plain CRUD via `HoldingRepository`, no V1 `session_id` concept anywhere in the V2 schema (§4.8).

### 13.2 Price Fetch Separation
`GET /api/holdings` → `PortfolioService.get_holdings_with_prices()` → `HoldingRepository.list()` JOIN `PriceCacheRepository.get_last_known()` — zero calls into `pipeline.portfolio.fetch_price` or `yfinance`. `POST /api/holdings/refresh-prices` → `PortfolioService.refresh_prices()` → calls `pipeline.portfolio.fetch_price()` per ticker (which is itself cache-first, but the *cache being warm* is irrelevant here since this path is explicitly invoked to seek freshness) and updates `price_cache`.

### 13.3 Cache Semantics
An old cached price displays with its `fetched_at` timestamp (exposed to the frontend as `price_last_updated`) regardless of age — no age-based hiding. **V1's 1-hour TTL's remaining role:** kept only as an internal optimization inside `refresh_prices()` — if a ticker's cached price is under an hour old, `refresh_prices()` may skip re-calling `yfinance` for it and just return the cached value as "already fresh," reducing unnecessary API calls on a refresh click that covers many tickers. This is a performance courtesy, not a correctness gate, and the PRD does not require it — implementers may omit it without violating any acceptance criterion. It is not exposed as an `is_stale_price` flag in the MVP response schema (§6) since the PRD does not require distinguishing "cached but young" from "cached but old" in the UI — only that a price is displayable and its timestamp is visible.

### 13.4 Failure
`refresh_prices()` iterates tickers independently; a `fetch_price()` failure for one ticker (already returns `None` on any error, never raises — `portfolio.py:72-74`) is recorded as `{ticker, error: "price_fetch_failed"}` in the response's `failed` list, while other tickers still refresh normally. The failed ticker's holding still returns its last-known cached price (if any) on the next `GET /api/holdings` — never breaking the whole page. No raw `yfinance` exception text is passed to the client.

---

## 14. Demo Data Design

Based on §4.5's Option A+D combination (not Option C alone, per the prompt's explicit caution).

### 14.1 Demo Dataset
A new script, `backend/scripts/seed_v2_demo_data.py`, is written rather than reusing `db/seed_synthetic_data.py` directly, because the latter writes to V1's schema via `db/database.py` (different tables, different `session_id` semantics, no `data_mode`). It **reuses the data-generation patterns** from `seed_synthetic_data.py` (merchant lists per category, amount ranges, December/summer multipliers — `seed_synthetic_data.py:32-59`) but calls V2 repositories and stamps every row `data_mode='demo'`. It must populate: enough transactions across ≥12 months (to make the Forecast page immediately demonstrable, not cold-start), at least one holding with a cached price, and one `forecast_runs`/`forecast_predictions` set (so the Forecast page shows a populated result on first demo load without requiring the user to click "Generate" — this is what makes the demo "immediately visible and demonstrable" per PRD §10a).

### 14.2 Load Demo
`POST /api/demo/load` → `DemoService.load_demo()`: checks `app_state.mode`; if `'REAL'`, raises a conflict the route maps to 409 (no force option — constraint #5). If `'EMPTY'`, runs the seed script's logic, then sets `mode='DEMO'`.

### 14.3 Clear Demo
`DELETE /api/demo/clear` → deletes all `data_mode='demo'` rows across `transactions`, `holdings`, `forecast_runs` (cascades to `forecast_predictions`), and any `price_cache` rows exclusively used by demo holdings (a ticker also held for real is left alone — price data isn't demo/real-specific, §4.5). Sets `mode='EMPTY'`.

### 14.4 Demo → Real
Detected as a 409 from `POST /api/imports` while `mode='DEMO'` (§5.2/§5.3). Frontend confirms with the user, then calls `DELETE /api/demo/clear`, then retries the original import call. Each step is its own request/transaction; there is no single cross-service DB transaction spanning "clear demo + import," since the two are user-visible, separately confirmable steps rather than one atomic operation — acceptable because clearing demo data is itself idempotent and safe to retry if the subsequent import fails for an unrelated reason.

---

## 15. Error Handling

Envelope: `{"error": "stable_snake_case_code", "message": "human-readable", "details": {}}`.

| Status | Used for |
|---|---|
| 200 | Success, including cold-start (`GET /api/forecasts/status`) and idempotent repeats (re-confirming an import). |
| 400 | Malformed request — e.g., uploaded file isn't a CSV at all. |
| 404 | Entity not found (transaction id, holding id, batch id). |
| 409 | State conflict — demo/real mutual exclusivity violations. |
| 422 | Domain validation failure (invalid holding fields, invalid manual-transaction fields) **and** an explicit `POST /api/forecasts/run` attempted during cold-start (a rejected write, distinct from the 200 status read at `/api/forecasts/status` — see §5.6). FastAPI's own request-schema validation (missing required field, wrong type) also surfaces as 422; the response body is normalized into the same `{error, message, details}` envelope via a FastAPI exception handler, so the frontend has one shape to parse regardless of source. |
| 503 | Categorization model unavailable for a prediction-dependent write (constraint #6). |
| 500 | Unexpected internal error — logged server-side with full detail, returned to the client with a generic message and no stack trace. |

Import partial-row errors and market-data partial failures are **not** top-level HTTP errors — they are structured fields inside a 200 response (`rows_skipped_unparseable`, `failed: [...]` on refresh) since the overall request succeeded even though part of the underlying data had issues.

---

## 16. Configuration / Dependencies

**Config values:** `V2_DB_PATH` (default `plaincents_v2.db`, separate from V1's `plaincents.db` — see §18.2), `FRONTEND_ORIGIN` (for CORS, default `http://localhost:5173`), `KMEANS_MODEL_PATH` (imported from root `config.py`, not duplicated), `LOG_LEVEL`.

`.env.example`:
```
V2_DB_PATH=./plaincents_v2.db
FRONTEND_ORIGIN=http://localhost:5173
LOG_LEVEL=INFO
```

No secrets are needed for the MVP (no auth, no third-party API keys beyond `yfinance`'s keyless public endpoints).

**config.py evolution:** `backend/config.py` imports `CATEGORIES` and `BANK_DATE_FORMATS` from root `config.py` rather than redefining them, so the taxonomy stays single-sourced; it does not modify root `config.py`, so V1's `main.py` continues to import unchanged values.

**Dependency manifests** (to be created in the Build Plan, not here): `backend/requirements.txt` pinned (addressing the reality check's reproducibility gap — V1 currently has no pinned root `requirements.txt` per the audit); `frontend/package.json` via standard Vite scaffolding. No packages are installed by this document.

---

## 17. Testing Strategy

### 17.1 Unit Tests (services)
Ingestion parsing/dedup, categorization boundary (including missing-model 503 path), forecast cold-start detection, forecast staleness decision table (§7.2/§12.4), portfolio cached-price read-vs-refresh separation, demo state transitions (load/clear/conflict).

### 17.2 Repository Tests
Transaction CRUD + `UNIQUE(dedup_key)` constraint behavior, import batch status transitions, forecast run retention + `UNIQUE(forecast_run_id, category, forecast_month)`, stale-marking persistence, holdings CRUD, price-cache upsert, `app_state` mode transitions.

### 17.3 API Integration Tests
Exactly the list in the authoring prompt's §17.3 — TD upload→preview→confirm→transactions; duplicate re-import; manual create; category correction; forecast becomes stale after a relevant mutation; forecast-page read never generates; cold-start returns 200; explicit generation creates a new run; a second generation retains the first; demo load/clear; real→demo rejected (409); demo→real confirm/reset/import; portfolio GET makes no network call (assert via a mocked/blocked `yfinance` client); explicit refresh does call it; missing model returns 503 and blocks the write cleanly.

### 17.4 TD Parser Fixture
A representative **de-identified** TD CSV fixture (synthetic values in TD's real column layout, not a real customer's data) is required before TD import can be called "verified" per PRD §9.2/§11.3 and TRD constraint #4. Fixture set includes: a clean valid file, a file with some unparseable date rows, a file with an unrecognized-column layout (to test the 400 path), and a file containing rows that duplicate another fixture file (to test dedup). **This TRD does not claim TD is currently verified** — no such fixture or test exists in the repository today; creating and passing against it is Build Plan work, gating the "TD verified end-to-end" MVP acceptance criterion.

### 17.5 ML Test Boundary
Application-level tests use a small, checked-in test model artifact (a `kmeans_model.pkl` fit on a tiny fixture dataset) or a fake `CategorizationService` implementation returning deterministic categories — never retraining the production model artifact as part of a test run. Scientific evaluation (accuracy, MAPE benchmarking) is the forthcoming ML Specification's responsibility, not these tests'.

---

## 18. V1 Compatibility / Migration Path

### 18.1 V1 Preservation
Verified against actual repository structure: `python main.py` (exists, `main.py:1`), `python -m pipeline.cluster` (module has `if __name__ == "__main__"`, `cluster.py:176`), `python -m pipeline.forecast` (`forecast.py:367`), `python db/seed_synthetic_data.py` (`seed_synthetic_data.py:228`). `viz/report.py` and `viz/powerbi_export.py` also have `__main__` blocks but are invoked as part of `main.py`'s pipeline (`main.py:141-149`) rather than typically run standalone — both are still runnable directly (`report.py:287`, `powerbi_export.py:118`). All five commands are preserved unmodified; V2 backend code does not import from `db/database.py`, `db/schema.sql`, or any V1 `main.py` code path.

### 18.2 V1 vs V2 Database
**Chosen: separate `plaincents_v2.db`.** Reusing `plaincents.db` would require either (a) adding V2 tables alongside V1's incompatible `transactions` schema (name collision — V1's `transactions` table has no `dedup_key`/`data_mode`/`effective_category` view, so the table cannot serve both schemas), or (b) migrating V1's schema in place, which would break `main.py`'s existing inserts/queries (§18.1's preserved commands). A separate file avoids all of this: V1's batch pipeline keeps writing to `plaincents.db` exactly as today, V2 reads/writes only `plaincents_v2.db`, and there is no risk of V1's synthetic/batch data contaminating V2's demo/real distinction (which V1's schema has no concept of at all). Verified, not assumed, per the prompt's instruction — the schema incompatibility above is the concrete reason, not a default assumption.

### 18.3 Configuration Compatibility
Root `config.py` is unmodified and continues to serve V1's `main.py`/`pipeline/*` imports exactly as today. `backend/config.py` is new and additive, importing shared constants (`CATEGORIES`, `BANK_DATE_FORMATS`, `KMEANS_MODEL_PATH`) from root `config.py` rather than duplicating or overriding them, so a change to the shared taxonomy in one place is seen by both V1 and V2.

### 18.4 Deprecation
`main.py` is not deleted or modified during V2 implementation. Recommendation: retain it as the permanent V1 batch/demo-generation utility (useful for regenerating `viz/report.py`'s PDF or `viz/powerbi_export.py`'s CSVs against V1's schema, since those remain V1-only per PRD §17) until an explicit future decision — outside this TRD's scope — chooses to mark it legacy or deprecate it after V2 reaches feature parity with V1's reporting outputs.

---

## 19. Technical Non-Goals

The MVP does **not** require: Redis, Celery, task queues, WebSockets, authentication, JWT, OAuth, multi-user tenancy, PostgreSQL, Docker, Kubernetes, cloud deployment, microservices, PDF integration, PowerBI integration, LLMs, vector databases, automatic model retraining, automatic forecast generation, automatic portfolio refresh, streaming responses, GraphQL, event buses/Kafka.

**No genuine technical blocker was found in this TRD that requires any of the above.** The one place a scheduling-like mechanism was briefly considered — expiring stale `previewing` import batches (§4.3) — was deliberately left as a non-requirement rather than justifying a background job, since no PRD acceptance criterion needs it and single-user usage makes an orphaned preview low-cost.

---

## 20. Security / Privacy Boundaries

- Uploaded CSV bytes are held only in memory/`staged_transactions` for the life of the import batch; no uploaded file is written to an arbitrary filesystem path (avoids path-traversal surface entirely — there is no user-controlled filename used to construct a save path).
- Upload size limit: a reasonable cap (e.g., 10 MB) rejected with 400 before parsing, to avoid a pathological file consuming memory — not a claimed security control, just basic robustness.
- CSV parsing uses `pandas.read_csv` over an in-memory buffer, never `eval`/formula execution — no CSV-injection-into-spreadsheet-app concern applies since PlainCents never re-exports these values into a spreadsheet that a user might open with formulas enabled (PDF/PowerBI export is post-MVP, §17 PRD).
- Logs never contain full transaction rows or uploaded file contents (§21).
- External network requests are limited to `yfinance` ticker lookups triggered only by explicit "Refresh Prices" — no other outbound call exists anywhere in the backend.
- CORS allows only the configured local frontend origin (§1.7) — no wildcard.
- No enterprise security/compliance claim (SOC2, encryption-at-rest certification, audit logging platform) is made or implied — proportional to a local single-user MVP per PRD §15.

---

## 21. Observability / Logging

Logged: application startup, migration application (which numbered files ran), categorization model load outcome (loaded/missing/error), import batch result counts (not row contents), forecast run start/success/failure (with run id and months_available, not underlying transaction data), price-refresh success/failure counts per attempt (not raw API payloads).

Never logged: full transaction histories, uploaded CSV contents, individual merchant/amount/date values outside of aggregate counts. No external monitoring stack (Sentry, Datadog, etc.) is required — Python's standard `logging` to stdout/a local file is sufficient for a local single-user MVP.

---

## 22. Key Technical Decisions Summary

| Decision | Alternatives considered | Chosen | Why | PRD/repo basis | Consequences |
|---|---|---|---|---|---|
| SQLite | PostgreSQL | SQLite | Single-user, local-first, zero ops overhead; V1 precedent | PRD §7, §9.9 | Single-writer limitation documented, not solved (§1.8) |
| Separate V2 DB file | Reuse `plaincents.db` | `plaincents_v2.db` | V1 schema structurally incompatible (no dedup/data_mode/effective_category) | Reality check §18.2 | Two DB files to keep straight during dev; documented in §18.2 |
| Migration mechanism | Alembic, none | Numbered SQL files + `schema_migrations` table | Proportional to single-dev local MVP | §19 (no unjustified tooling) | Manual ordering discipline required |
| Schema source of truth | Separate `schema_v2.sql` + migrations, migrations only | `db/migrations/` only — `001_initial_v2.sql` is the full initial DDL, no parallel `schema_v2.sql` | A second file describing the same schema is a drift risk with no benefit | Amendment pass (this freeze) | Simpler file layout; §4.11's DDL block is sourced from the migration file, not a separate reference file |
| `EMPTY` state semantics | Treat `EMPTY` as a row-level `data_mode`, treat it as a pure application state | Application-only state (`app_state.mode`); row `data_mode` remains `'demo'`/`'real'` only | Prevents a nonsensical `WHERE data_mode = 'EMPTY'` filter; keeps "no data yet" cleanly distinct from "demo data" | PRD §10a, §11.12 | Every mode-aware read path must branch on `app_state.mode` before deciding whether to filter by row `data_mode` at all (§4.5.1) |
| Effective category | Stored column, generated column, view | SQL VIEW (`v_transactions_effective`) | Usable directly in aggregate SQL; no migration if rule changes | PRD §12 | One extra JOIN-free view to maintain |
| Manual-override flag | Stored `is_manual_override` column | Derived (`confirmed_category IS NOT NULL`) | Avoids a second source of truth that can drift | Prompt's explicit "don't add merely because mentioned" instruction | None — trivial to derive |
| Duplicate strategy | Hash-based, app-only, DB constraint | `UNIQUE(dedup_key)` with occurrence-index heuristic | Covers exact re-import while allowing legitimate same-day/amount purchases | PRD §9.2a | Known limitation: doesn't catch near-duplicates (flagged) |
| Import preview staging | Staging table, serialized blob, filesystem, client-held | Staging table (`staged_transactions`) | Survives restart, supports live re-check at confirm, no blob/filesystem lifecycle | PRD §11.2 (preview→confirm flow) | One extra table + cleanup step |
| Demo isolation | Per-row flag only, separate tables, batch-only flag, app-mode only | Per-row `data_mode` + app-level `app_state.mode` gate | Defense in depth; every read query filters by mode regardless of gate correctness | PRD §9.2b, constraint #5 | Every demo-touching table needs the `data_mode` column |
| Forecast run persistence | Overwrite latest, append-only runs | Append-only `forecast_runs`/`forecast_predictions`, run-scoped uniqueness | PRD requires retained history; run-scoped key avoids the prompt's flagged uniqueness trap | PRD §17 (forecast-vs-actual, future), constraint #2 | More rows over time; no MVP pruning implemented |
| Forecast staleness | Derived, stored | Stored boolean on `forecast_runs`, only latest run flippable | A later-reversed change shouldn't silently un-stale a forecast | PRD §9.6, constraint #2 | Orchestration must live in services, not repositories (enforced, §7 rule) |
| Forecast-vs-actual | Include now, defer | Deferred entirely | No MVP consumer; avoids orphan schema | PRD §17 | Future feature must use run-scoped key when built (documented) |
| Holdings/price cache | Latest-only, historical series | Latest-only `price_cache` (matches V1) | No MVP consumer of price history | PRD §9.7 | Revisit if a price-trend chart is ever requested |
| Dashboard aggregation | Live query, materialized `monthly_summary` | Live SQL aggregation | Simpler, always consistent, MVP data volume is small | PRD §11.7 | V1's `monthly_summary` concept is not carried into V2 schema |
| Frontend server-state | TanStack Query, SWR, Redux Toolkit | TanStack Query + local state | Matches app's read-heavy/few-mutations shape without Redux boilerplate | — (pure technical choice) | — |
| Categorization boundary | Direct `predict_categories()` calls, service wrapper | `CategorizationService` wrapping V1 logic, loaded once at startup | Avoids per-request disk reload; keeps contract model-agnostic | PRD §9.3, Non-negotiable ML rule | K-Means specifics never leak past this service |
| Forecasting boundary | Reuse `fit_and_forecast()` directly on user path | New `train_and_predict()` (fit-only, no walk-forward/GridSearch) for the interactive path; full `fit_and_forecast()` kept for offline evaluation | V1's function is too slow/heavy for a synchronous click | PRD §9.6, constraint #1, Non-negotiable ML rule | Two forecast code paths (interactive vs. offline) to keep in sync conceptually |
| V1 compatibility | Delete/modify `main.py`, leave untouched | Untouched, preserved indefinitely pending future decision | PRD/prompt explicitly forbid deletion during V2 build | §18.4 | Two parallel systems exist during the build; acceptable per PRD §17 |

---

## Required Self-Audit

### A. PRD Traceability

| TRD decision | Frozen PRD section |
|---|---|
| Manual-only forecast generation/refresh | PRD §9.6, §11.8 |
| Forecast run retention | PRD §12 (Forecast Run concept), §17 (post-MVP forecast-vs-actual implies retained history) |
| Cold start as HTTP 200 structured state | PRD §11.8, §21 |
| TD-only MVP bank | PRD §9.2, §11.3, §7 |
| Demo/real mutual exclusivity, asymmetric load/clear | PRD §9.2b, §10a |
| Predicted/confirmed/effective category | PRD §9.3, §12 |
| Manual creation uses same categorization | PRD §9.3 |
| Fixed 8-category taxonomy | PRD §9.4 |
| Manual-only portfolio price refresh | PRD §9.7, §11.9 |
| PDF/PowerBI excluded from MVP | PRD §11.10, §17 |
| No auth/multi-user | PRD §11 (non-goals), §18 |
| Import duplicate detection required | PRD §9.2a, §11.2 |
| Calendar-month dashboard default | PRD §11.7 |

**Decisions with no direct PRD line but technically necessary:** the `app_state` table and `staged_transactions` table are not named anywhere in the PRD (correctly — the PRD stays implementation-agnostic per its own rules), but both are necessary mechanisms to satisfy PRD requirements that *are* explicit (demo/real exclusivity; upload→preview→confirm). Flagged here rather than silently introduced, per the authoring prompt's instruction.

### B. PRD Acceptance-Criteria Coverage (PRD §19)

| Acceptance criterion | Endpoint(s) | Service(s) | Repo/schema | Frontend |
|---|---|---|---|---|
| TD upload → transactions with predicted category | `POST /api/imports`, `POST /api/imports/{id}/confirm` | `IngestionService`, `CategorizationService` | `staged_transactions`, `transactions` | Import page |
| Edit transaction, persists | `PATCH /api/transactions/{id}` | `TransactionService` | `transactions` | Transactions page |
| Delete transaction | `DELETE /api/transactions/{id}` | `TransactionService` | `transactions` | Transactions page |
| Category correction reflected in dashboard, prediction retained | `PATCH /api/transactions/{id}`, `GET /api/dashboard/summary` | `TransactionService`, `DashboardService` | `v_transactions_effective` | Transactions + Dashboard |
| Forecast generate, reopen without retrain | `POST /api/forecasts/run`, `GET /api/forecasts/latest` | `ForecastService` | `forecast_runs`/`forecast_predictions` | Forecast page |
| Cold-start message | `GET /api/forecasts/status` | `ForecastService` | `TransactionRepository.count_distinct_months` | Forecast page |
| Holding CRUD + P&L, no auto network | `GET/POST/PATCH/DELETE /api/holdings` | `PortfolioService` | `holdings`, `price_cache` | Portfolio page |
| Explicit refresh vs. page-open | `POST /api/holdings/refresh-prices` vs. `GET /api/holdings` | `PortfolioService` | `price_cache` | Portfolio page |
| Demo offer, populated on load | `POST /api/demo/load` | `DemoService` | all `data_mode='demo'` rows | Onboarding/empty state |
| Demo→real conflict confirmation | `POST /api/imports` (409), `DELETE /api/demo/clear` | `IngestionService`, `DemoService` | `app_state` | Import page modal |
| Duplicate import skipped + reported | `POST /api/imports/{id}/confirm` | `IngestionService` | `transactions.dedup_key` UNIQUE | Import result screen |
| Forecast marked stale after mutation | any transaction mutation endpoint | `TransactionService`/`IngestionService` → `ForecastService.mark_stale` | `forecast_runs.is_stale` | Forecast page banner |
| No unhandled error on empty screens | all `GET` endpoints | respective services (return empty-but-valid structures, not errors) | — | Empty-state components |

All confirmed implementable using this TRD's design.

### C. Reality-Check Alignment

| Reality-check finding | Addressed in TRD |
|---|---|
| `ingest.py` expects filesystem path | §3 (REFACTOR — bytes-based entry point added) |
| `predict_categories()` reloads pkl per call | §7.3, §11.1 (loaded once at startup) |
| `fit_and_forecast()` retrains every call | §12.1–§12.3 (leaner user-path function; heavy path moved offline) |
| Forecast requires ≥12 months | §12.5 (kept as MVP threshold, surfaced as cold-start) |
| `portfolio.py` mixes fetch + persist | §3, §13.2 (separated) |
| `database.py` append-only, no UPDATE/DELETE, unfiltered reads | §8 (new repository layer with full CRUD and explicit filters) |
| `session_id` batch semantics | §4 (replaced with `import_batch_id`/`forecast_run_id`/`data_mode`, no `session_id` in V2 schema) |
| V1 DB reads mix sessions | §4.5 (`data_mode` filtering on every query) |
| Need true CRUD persistence model | §4, §8 |
| PDF/PowerBI adaptable later | §3, §17 (LEGACY/PRESERVE, untouched) |
| V1 `main.py` should not become V2 API architecture | §1, §2, §18.4 (new `backend/` app; `main.py` preserved but not extended) |

### D. V1 Reuse Accuracy

Every classification in §3 was made after reading the actual file content in this session (line numbers cited throughout). **No REUSE AS-IS classification was made without inspection** — `features.py` was read in full and confirmed pure/stateless before being marked REUSE AS-IS. No uncertain classifications remain; where a function is REUSE AS-IS but wrapped rather than called directly (e.g., `predict_categories()`'s inner steps), this is called out explicitly rather than glossed over.

### E. Schema Completeness

| PRD §12 concept | Table/column/derived/deferred |
|---|---|
| Transaction | `transactions` table |
| Import Batch | `import_batches` table |
| Predicted Category | `transactions.predicted_category` |
| Confirmed Category | `transactions.confirmed_category` |
| Effective Category | Derived — `v_transactions_effective` view |
| Forecast | `forecast_predictions` rows |
| Forecast Run | `forecast_runs` table |
| Holding | `holdings` table |

No orphan tables: every table in §4.11 is justified by a PRD concept, a required mechanism (`app_state`, `staged_transactions`), or explicit product need (`price_cache`, matching PRD §9.7). `forecast_vs_actual` is explicitly **not** created (§4.7) — deferred, not an orphan.

### F. Schema Invariant Check

1. **Same TD CSV imported twice** — Second confirm's rows all collide on `dedup_key` (same date/amount/merchant/bank_source/occurrence_index sequence) → all reported as duplicates, zero re-inserted. **Works.**
2. **Two legitimate same-day/same-amount purchases** — Differ by `occurrence_index` (0 and 1) → both persist as distinct rows. **Works**, with the known heuristic limitation noted in §4.4 (a true accidental duplicate presented as the 2nd of two identical-looking real purchases in the *same* file would still be correctly kept, since dedup only fires against *already-persisted* transactions or *within-batch* exact repeats — cross-batch legitimate repeats are the harder case and are accepted as a documented limitation).
3. **Manually created transaction** — Goes through `TransactionService.create_manual` → `CategorizationService.predict()` → `dedup_key` computed and checked → persisted with `import_batch_id = NULL`. **Works.**
4. **Predicted category later corrected** — `PATCH` sets `confirmed_category`; `predicted_category` column is never written to again. `v_transactions_effective` immediately reflects the correction. **Works.**
5. **Confirmed category later cleared** — `PATCH` with `confirmed_category = NULL` (an explicit "clear correction" action) restores `effective_category` to the original `predicted_category` automatically via the view's `COALESCE`. **Works** — no separate "revert" logic needed.
6. **Demo → real transition** — Detected via 409 on import attempt while `mode='DEMO'`; explicit clear-then-import sequence (§14.4). **Works**, contingent on the frontend actually performing both steps in sequence (a single-request atomic "clear and import" is not implemented — acceptable per §14.4's reasoning).
7. **Attempted real → demo transition** — `load_demo()` checks `mode='REAL'` → 409, no data touched. **Works.**
8. **Two forecast runs both predicting the same category/month** — `UNIQUE(forecast_run_id, category, forecast_month)` scopes uniqueness to the run, so run A's `(Shopping, 2026-10)` and run B's `(Shopping, 2026-10)` coexist as distinct rows under different `forecast_run_id`s. **Works** — this is the exact trap the prompt warned about, and the schema in §4.11 avoids it.
9. **Transaction deletion after forecast** — Delete triggers `ForecastService.mark_stale()` via `TransactionService`; the forecast row itself is untouched (historical record preserved), only flagged stale. **Works.**
10. **Stale cached stock price** — `GET /api/holdings` returns it regardless of `fetched_at` age, with the timestamp exposed; no gate. **Works** as designed (§13.3).
11. **Missing categorization artifact** — Existing transactions remain fully readable (`GET` endpoints never touch `CategorizationService`); new writes (`POST /api/transactions`, import confirm) return 503, no `predicted_category = NULL` row is ever written. **Works.**
12. **Application restart during an import preview** — `staged_transactions` is durable SQLite; confirm after restart re-reads the same staged rows and re-checks duplicates live against current `transactions`. **Works.**
13. **First real write while `app_state.mode = EMPTY`, then a failed write attempt before any real row exists** — a failed manual-creation attempt (e.g., 503 from missing model) leaves `mode` at `EMPTY` (§4.5.1); a subsequent *successful* creation, edit, or import commit correctly transitions `EMPTY → REAL` on that first durable success, and downstream optional-step failures (e.g., a staleness-hook error) do not revert the transition. **Works.**

All twelve invariants hold under the proposed design; none required a design change during this check.

### G. API Completeness

Every MVP action in PRD §16 has a corresponding endpoint in §5. **Technical-support-only endpoints** (not directly product-facing, but necessary plumbing): `GET /api/health` (diagnostic, not a user-facing feature per se, though its `data_mode` field feeds the demo banner), `GET /api/imports`/`GET /api/imports/{batch_id}` (supports the Import Batch visibility PRD §12 mentions as "give the user visibility," which is product-facing but secondary to the core confirm flow).

### H. Layering Check

- Routes contain no SQL — confirmed by design (§1.2, §5); routes call exactly one service method each.
- Routes do not call ML modules directly — confirmed (§1.2; all `pipeline.*` calls happen inside `backend/services/*`).
- Repositories contain no business orchestration — confirmed (§8's descriptions are pure CRUD/query methods; no repository method name implies cross-table decision-making).
- Repositories do not call ML or `yfinance` — confirmed by §8's method list (no repository imports `pipeline.cluster`, `pipeline.forecast`, or `yfinance`).
- Services coordinate cross-domain behavior — confirmed (§7's staleness orchestration explicitly lives in `TransactionService`/`IngestionService`, not `ForecastRepository`).
- Dashboard reads persistence only — confirmed (§7.6, §22.C row "DashboardService... No ML. No yfinance.").
- Forecast reads do not generate — confirmed (§7.4, §12.2: `get_latest`/`check_status` never touch `pipeline.forecast`).
- Portfolio reads do not fetch market prices — confirmed (§7.5, §13.2).

### I. ML Agnosticism

- API exposes only `predicted_category`/`confirmed_category`/`effective_category` as plain strings — no cluster IDs, no K-Means-specific fields in any schema in §6. Confirmed.
- React types (§9.4) mirror those same string-only schemas — no K-Means dependency. Confirmed.
- Repositories store `predicted_category` as a plain TEXT column — no K-Means dependency. Confirmed.
- Future categorizer replacement requires changing only `backend/services/categorization_service.py`'s internals (§11.4) — confirmed, no other layer references K-Means by name except code comments/docstrings explaining the *current* implementation.
- API exposes no Random-Forest-specific concepts — `ForecastPrediction`/`ForecastRunResponse` (§6) contain only category/month/amount/availability fields; `model_impl_version` is a free-text audit tag, not a structured RF-specific field. Confirmed.
- React does not depend on Random Forest — confirmed, same schemas.
- Forecast persistence does not hardcode RF — `forecast_predictions` stores only the prediction output shape, not RF hyperparameters or tree structure. Confirmed.
- Future forecaster replacement requires changing only `pipeline.forecast`'s `train_and_predict()` implementation and `ForecastService.run_forecast()`'s internals — no schema or route change. Confirmed.

### J. Interactive Performance Check

- **CSV preview:** parsing + dedup-check + categorization for a typical statement (tens to low hundreds of rows) — expected sub-second to low-single-digit-seconds; not flagged as a risk.
- **Import confirmation:** bulk insert of the same row count — expected fast; not flagged.
- **Forecast generation:** per §12.3, redesigned to a single RF fit on a small aggregated table (categories × months) — expected sub-second; the *original* `fit_and_forecast()` (walk-forward + conditional GridSearchCV) **would** have been a real risk (potentially many seconds to tens of seconds) and is explicitly excluded from the user path for exactly this reason. This is the one place a genuine performance concern was found and it is solved by scope reduction (moving the expensive work offline), not by adding queue infrastructure.
- **Portfolio refresh:** one `yfinance` network call per ticker, sequential in the current `pipeline.portfolio.fetch_price` design; for a small personal holdings list (a handful of tickers) this is acceptable synchronously (each call is typically ~1 second per V1's own documented behavior in the reality check). Not flagged as needing async infrastructure at MVP scale, though a Build Plan implementer should be aware sequential calls could add up if a user holds many tickers — noted, not solved, since no evidence of that scale exists.

### K. Technical Non-Goals Check

Reviewed against §19's list — nothing in this TRD introduces Redis, Celery, task queues, WebSockets, auth, PostgreSQL, Docker/Kubernetes, microservices, PDF/PowerBI integration, LLMs, vector databases, automatic retraining/forecast generation/portfolio refresh, streaming responses, GraphQL, or event buses. Confirmed clean.

### L. Scope Check

Nothing in this TRD expands beyond PRD §16's MVP list. Every endpoint, service, and table traces to an MVP feature (§22.A/§22.B). The one candidate for scope-creep scrutiny — `import_batches`/`staged_transactions` visibility endpoints (§5.3's `GET /api/imports`) — is justified by PRD §12's Import Batch concept, which the PRD itself lists as an MVP-relevant concept, not an addition by this TRD.

### M. Open Technical Questions

1. **Exact per-category "sufficient history" statistical rule beyond the existing 6-month rolling-window `dropna`** — **DEFERRED TO ML SPECIFICATION** (§12.5).
2. **Whether the occurrence-index dedup heuristic (§4.4) needs strengthening (e.g., to handle a legitimate duplicate purchase split across two separate CSV exports)** — **NON-BLOCKING**; may be resolved during implementation with real TD fixture data, or deferred further if it doesn't surface in practice.
3. **Exact CSV upload size limit value** — **NON-BLOCKING**; any reasonable default (e.g., 10 MB) satisfies the PRD; can be tuned during implementation.
4. **Whether `refresh_prices()`'s internal TTL optimization (§13.3) is worth implementing at MVP or can be skipped entirely (always re-fetch on explicit refresh)** — **NON-BLOCKING**; PRD does not require it either way.
5. **Whether the packaged "single process serves built frontend" run mode (§1.7) is actually needed for the MVP demo, or whether two dev servers are an acceptable reviewer experience** — **NON-BLOCKING**; a product/demo-logistics call, not a architectural blocker either way.

No item above is marked **BLOCKING** — the TRD is believed sufficient to proceed to the Build Plan as written.

---

*No V2 code, migrations, or production files were created or modified in the production of this document. Only `docs/V2_TRD.md` was written.*
