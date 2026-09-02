# PlainCents V2 — Build Plan

**Status:** FROZEN — V2 Build Plan
**Traces to:** `docs/V2_PRD.md` (FROZEN), `docs/V2_TRD.md` (FROZEN), `docs/V2_ML_SPEC.md` (FROZEN), `docs/V2_PREBUILD_REALITY_CHECK.md`
**Purpose:** Convert the frozen PRD (what), TRD (how/contracts), and ML Spec (how ML earns final status) into an ordered, testable, phase-by-phase implementation sequence. This document does not implement anything, does not create migrations/code, and is not frozen by its own authoring — it is reviewed before freeze like the three documents before it.

---

## 0. Repository Reality Check (verified this session, grounds Phase 0)

- `requirements.txt` **exists** but is corrupted: it is UTF-16-encoded (Windows `pip freeze` artifact), which renders as garbled spaced characters to normal text tools and will not `pip install -r` correctly as-is. This must be fixed in Phase 0 — not assumed away.
- Installed versions differ from what's pinned: `pandas==3.0.1`, `scikit-learn==1.9.0` actually installed vs. `pandas==3.0.1`/`scikit-learn==1.8.0` in the (corrupted) file — Phase 0 re-pins from the actual working environment, not from the stale file.
- `models/kmeans_model.pkl` and `models/rf_model.pkl` exist on disk today, but `.gitignore` has a blanket `*.pkl` rule — meaning any committed test-fixture model artifact (§4 of the ordering constraints) needs an explicit `.gitignore` exception, not an assumption that "models already work this way."
- No `backend/` or `frontend/` directory exists yet — Phase 0 starts from a clean slate for V2, exactly as the TRD's repo structure (§2) describes.
- Python 3.14 is the active interpreter — noted for Phase 0's dependency-compatibility check (FastAPI/Pydantic version selection must target this, not assumed from memory).

---

## 1. Non-Negotiable Build Philosophy (carried from the authoring prompt, restated for traceability)

1. **App track bootstraps on V1's K-Means/RF** behind the TRD's frozen service boundaries — this is not a scientific acceptance of either model (ML Spec §0/§22).
2. **ML runs as a parallel track** (ML-A–E) that does not block ordinary app phases unless a frozen acceptance gate genuinely requires it.
3. **One phase per Claude Code session** — every phase below is scoped to be implementable, testable, and stoppable in one sitting, per the template in §14.
4. **Do not rewrite V1 unnecessarily** — TRD §3's REUSE/WRAP/REFACTOR/LEGACY classifications are the ceiling on what changes; V1's `main.py` and CLI commands (TRD §18.1) must keep working throughout.
5. **No architecture drift** — the TRD §19 non-goals list (Redis, Celery, auth, PostgreSQL, Docker, microservices, LLMs, etc.) applies to every phase below without exception.

---

## 2. Additional Implementation-Ordering Constraints (resolved explicitly, not left implicit)

### 2.1 Phase 3 ↔ Phase 4 categorization dependency — **Option A chosen**

**Decision: Option A — minimal `CategorizationService` bootstrap in Phase 3.**

Phase 3 (Transaction CRUD) introduces the minimum `CategorizationService` needed to satisfy manual-create's frozen requirement (TRD §7.3, §11.1–§11.3): artifact loading at FastAPI startup, `predict(transaction) -> {predicted_category}`, and graceful `missing`/`error` status reporting (503 on a prediction-dependent write). Phase 4 (TD Import) then **reuses the same service instance** for bulk prediction (`predict_batch`) rather than building a second categorization path — the TRD's contract (§7.3) is singular and this Build Plan does not create two competing implementations.

**Why Option A, not Option B:** the TRD explicitly scopes `CategorizationService` as a single, once-loaded, startup-lifecycle object (§11.1) — splitting "manual predict" and "bulk predict" into separately-built services in two different phases would risk exactly the divergence the TRD's model-agnostic boundary (§11.4) is designed to prevent. Building the minimal service once, early, and extending it is materially less risky than building it twice and reconciling later. Phase 3's acceptance gate therefore **does** include working manual-create categorization (not deferred), and Phase 4 explicitly reuses — never reimplements — the Phase 3 service.

### 2.2 Interactive forecast path — exact function inventory

The TRD (§12.1–§12.3) and ML Spec (§11) require a lean, non-blocking interactive path. This Build Plan fixes the following inventory (Phase 7 implements it):

| Function | Classification | Used by |
|---|---|---|
| `pipeline.forecast.aggregate_monthly()` | **Interactive/runtime** — reused as-is | `ForecastService.run_forecast()` |
| `pipeline.forecast.build_forecast_features()` | **Interactive/runtime** — reused as-is | `ForecastService.run_forecast()` |
| **NEW** `pipeline.forecast.train_and_predict(monthly_df)` | **Interactive/runtime** — new, added in Phase 7 | `ForecastService.run_forecast()`; performs exactly one `RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)` fit (TRD §12.3's fixed defaults — the same configuration ML Spec §11's frozen bake-off evaluates RF at, per §2.2 and ML Spec §11) and generates the 3-horizon predictions. **No walk-forward loop, no GridSearchCV, no diagnostic comparison inside this function.** |
| `pipeline.forecast.walk_forward_validate()` | **Offline evaluation** — untouched, reused only by ML-B scripts | ML-B categorization/forecast bake-off scripts (not the API) |
| `pipeline.forecast.fit_and_forecast()` | **Legacy compatibility** — untouched, remains V1's own entry point | V1's `main.py` only; **never called from any V2 route or service** |
| V1 `__main__` diagnostic block (K-Means vs. heuristic-label comparison) | **Legacy compatibility** — untouched | `python -m pipeline.forecast` only |

**Binding rule for Phase 7's acceptance gate:** a code-search/import-graph check confirming `fit_and_forecast`, `walk_forward_validate`, and `GridSearchCV` do not appear anywhere in `backend/services/forecast_service.py`'s call chain is a required verification step (§8, Phase 7).

### 2.3 Reviewer/packaged run mode

**Decision, per TRD §1.7:** development uses two servers (Vite dev + FastAPI `--reload`, proxied). The **reviewer/demo run mode** builds the Vite app to static files and serves them from FastAPI via `StaticFiles` mounted at `/`, with the API under `/api/*` — one process, one port, one command. This is planned explicitly in **Phase 10** (polish), not invented ad hoc at the end. The final `README.md` (Phase 10 deliverable) documents both modes side by side under clearly separate headings ("Development" vs. "Run the demo"). No cloud deployment is introduced at any point.

### 2.4 Phase 0 fixtures / model bootstrap

**A. TD CSV fixture strategy:** a small set of hand-authored, de-identified, TD-shaped CSV fixtures is created and **committed** in Phase 0 under `tests/fixtures/td_csv/` — covering a clean valid file, a file with unparseable date rows, an unrecognized-column-layout file, and a file containing rows that duplicate another fixture. These fixtures are explicitly labeled in a fixture-local README as **synthetic, TD-format-shaped test data**, not verified real-export samples — PRD §9.2/§11.3's "not verified until a representative real export is checked" caveat is preserved; these fixtures make the *parser* testable, they do not themselves constitute the real-export verification the MVP acceptance criterion requires (that verification is a manual step tracked in Phase 4's acceptance gate, performed against the product builder's own real TD export outside the repository).

**B. Categorizer test-artifact strategy — deterministic bootstrap command chosen over a checked-in binary.** Phase 0 adds a small, deterministic script (`tests/fixtures/build_test_kmeans_model.py`) that fits a tiny K-Means artifact on a small fixed synthetic sample (reusing `pipeline.cluster.fit_and_evaluate`'s existing logic, fixed `random_state`) and saves it to `tests/fixtures/kmeans_model_test.pkl`. **Reasoning over a checked-in binary:** a committed `.pkl` is opaque, hard to diff/review, and risks silently going stale relative to `pipeline/features.py`/`pipeline/cluster.py` if either changes; a deterministic build script is transparent, reviewable as code, and reproducible by CI or any developer with one command (`python tests/fixtures/build_test_kmeans_model.py`), which the frozen ML Spec's reproducibility principle (§19) already favors. Phase 0's test setup documentation states this command explicitly — no developer is expected to manually reconstruct an unknown artifact.

**`.gitignore` correction (Phase 0 task):** the blanket `*.pkl` rule currently would also exclude `tests/fixtures/kmeans_model_test.pkl` if it were ever committed instead of generated — since the bootstrap-script approach is chosen, the test artifact is **not** committed either; it is generated on demand by the fixture script and remains gitignored consistently with production artifacts. Phase 0 documents this distinction (generated production artifacts vs. generated-on-demand test artifacts) explicitly in `tests/fixtures/README.md` so the rule's intent is clear rather than accidental.

### 2.5 Demo functionality sequencing

**Decision:** Phase 5 (frontend foundation) builds only the **UI hooks** — the `AppStateContext` (TRD §9.7), the demo banner component, and the EMPTY/DEMO/REAL conditional rendering — wired against a `GET /api/demo/status` endpoint that Phase 2 (App State foundation) already makes minimally functional (reading `app_state.mode`, always returning `EMPTY` until Phase 9 exists). **`POST /api/demo/load` and `DELETE /api/demo/clear` are not functionally implemented until Phase 9.** No phase before Phase 9 may claim a working demo-load acceptance gate; Phase 5's acceptance gate explicitly states "banner and empty-state UI render correctly against a mocked/stubbed mode value" rather than "demo loading works end-to-end." Phase 9 alone is responsible for: deterministic demo seed, sufficient (≥12 months) demo transaction history, a prebuilt demo forecast run (so entering demo mode never waits on a live fit merely to render), at least one demo holding with a cached price, persistent DEMO labeling, atomic full reset, and DEMO/REAL mutual exclusion — all per TRD §14 and PRD §10a.

---

## 3. Frontend Design System — Implementation Decision

**Stack:** React + TypeScript + Vite (TRD §9.1) + **Tailwind CSS** + **shadcn/ui** + **Lucide React** (icons) + `clsx` + `tailwind-merge`. **Charting: Recharts** (lightweight, React-idiomatic, sufficient for the dashboard/forecast chart needs in PRD §11.7/§11.8; no TRD conflict — TRD §9 left this open). **Motion: GSAP, used selectively** for entrance/reveal/count-up/modal-transition polish only — never as product logic, never large-scale (no Three.js/Vanta/Lenis/particle backgrounds/3D/parallax, per the authoring prompt's explicit exclusion list). GSAP is **optional polish, not an acceptance-gate requirement** — if time-constrained, Phases 5/6/10 may ship without it with no acceptance-criterion impact, since no PRD criterion mentions animation.

**Where established:** the full design system (Tailwind config, shadcn primitives, icon usage convention, `clsx`/`tailwind-merge` utility, chosen chart library) is installed and its **reusable primitives** (buttons, cards, badges, modals/confirmation dialogs, loading skeletons, empty-state component, toast/feedback pattern) are built **once, in Phase 5**, then reused verbatim in Phases 6–9. No separate "UI redesign phase" exists later — Phase 10 only polishes what Phases 5–9 already built with this system.

**UI/UX principles enforced across every frontend phase** (PRD §13, TRD §9.8): predicted-vs-corrected category visually distinguishable; destructive actions require confirmation; demo mode clearly labeled; cached/external price freshness visibly communicated; desktop-first responsive layout (tablet-acceptable, mobile not required by PRD); clear loading/empty/error states on every screen.

---

## 4. Database / Transaction Safety — Implementation Notes

1. V2 uses `plaincents_v2.db`, a separate file from V1's `plaincents.db` (TRD §18.2) — Phase 1 creates it; V1's file is never opened by V2 code.
2. `db/migrations/*.sql` is the **sole** schema source of truth (TRD §4.12, frozen amendment) — Phase 1 has no parallel `schema_v2.sql`.
3. Migration bootstrap (Phase 1) creates `schema_migrations` itself as the very first statement of `001_initial_v2.sql`, guarded with `CREATE TABLE IF NOT EXISTS`, so a fresh DB and a partially-migrated DB both converge correctly.
4. **`PRAGMA foreign_keys = ON`** must be set on **every** connection (SQLite does not persist this pragma across connections) — Phase 1's connection helper sets it immediately after opening, and a repository test (§8, Phase 1) asserts a foreign-key violation is actually rejected, not silently allowed.
5. **Unit-of-work / explicit transactions required for:** import confirmation (staged rows → `transactions` + `import_batches.status` update, TRD §10), forecast run persistence (`forecast_runs` + all its `forecast_predictions` rows, TRD §4.6), and full demo reset (deleting across `transactions`/`holdings`/`forecast_runs`/`price_cache` + resetting `app_state.mode`, TRD §14.3). Phase 1 builds a small `with transaction(conn):` context-manager helper (commit on success, rollback on any exception) that Phases 3/4/7/9 all reuse — not three bespoke implementations.
6. `updated_at` is set explicitly by application code on every `UPDATE` (not relied upon as an SQLite trigger, since SQLite has no built-in auto-update-timestamp mechanism) — Phase 1's repository tests assert `updated_at` actually changes on an update and does not change on a no-op read.
7. **Monetary value policy:** all amounts stored as SQLite `REAL`, rounded to 2 decimal places at the point of insertion/computation (matching V1's existing `round(x, 2)` convention throughout `forecast.py`/`portfolio.py`) — Phase 1 documents this once; every later phase that writes a monetary value follows it rather than re-deciding rounding behavior per feature.

No schema redesign — this section only adds implementation discipline around the TRD's already-frozen DDL (TRD §4.11).

---

## 5. App State / Demo / Real Data — Implementation Notes

Frozen semantics (TRD §4.5, §4.5.1) are implemented exactly as specified: `EMPTY` is never a row-level `data_mode`; the canonical read mapping (`EMPTY`→valid empty collections with no filter clause, `DEMO`/`REAL`→row filters) is implemented once in a shared repository helper (Phase 2) and reused by every repository that reads mode-scoped data (Phases 3, 4, 7, 8). The `EMPTY → REAL` transition (first durable real write: import commit, manual transaction creation, or holding creation) is implemented as a small shared `AppStateService.maybe_transition_to_real()` call invoked from the three respective services immediately after their own durable insert succeeds (TRD §4.5.1) — not duplicated three times with slightly different logic. Full demo reset is atomic (§4 above). Demo/real mutual exclusion is enforced server-side (409 responses, TRD §5.2/§5.3) regardless of what the frontend shows.

---

## 6. Dependency Graph

```
Phase 0 (foundation)
    ↓
Phase 1 (DB + repositories)
    ↓
Phase 2 (FastAPI + app state)
    ↓
Phase 3 (Transaction CRUD + CategorizationService bootstrap)
    ↓
Phase 4 (TD Import, reusing Phase 3's CategorizationService)
    ↓
Phase 5 (Frontend foundation + Import + Transactions UI)
    ↓
    ├── Phase 6 (Dashboard UI)         — requires Phase 3/4's transaction read APIs
    ├── Phase 7 (Forecast service + UI) — requires Phase 3/4's effective-category data
    └── Phase 8 (Portfolio CRUD + UI)   — independent of 6/7, only requires Phase 2's app-state/DB foundation
    ↓ (6, 7, 8 may proceed in any order once Phase 5 lands; they do not depend on each other)
Phase 9 (Demo/onboarding — requires 3,4,7,8's data shapes to seed against)
    ↓
Phase 10 (Integration/E2E/polish/packaged run mode)
```

**Legitimate overlap, not invented for appearance:** Phases 6, 7, and 8 are mutually independent once Phase 5's frontend shell and API client exist — each only needs its own backend API (dashboard summary, forecast status/run, holdings CRUD respectively), none needs the other two's UI. A team of one still benefits from this because it clarifies that reordering 6/7/8 (e.g., building Portfolio before Dashboard) does not violate any dependency. **ML-B may run fully in parallel with Phases 1–10** — it needs only the frozen ML Spec and, eventually, real/independent data, not the running application (ML Spec §22).

---

## 7. Milestones

| Milestone | Definition of "done" |
|---|---|
| **M0 — V2 Foundation Ready** | Phase 0 complete: dependencies pinned and installable, V2 folder skeleton exists, test fixtures/bootstrap script work, V1 commands still run unmodified. |
| **M1 — Core Data/CRUD Ready** | Phases 1–3 complete: V2 schema live via migrations, repositories tested, FastAPI app running with app-state endpoints, transaction CRUD (including manual-create categorization) working end-to-end via API. |
| **M2 — TD Import Working** | Phase 4 complete: TD CSV upload → preview → confirm → persisted transactions with predicted categories, dedup working, TD fixture tests passing. |
| **M3 — Interactive App Core Working** | Phase 5 complete: React app renders, Import and Transactions pages fully functional against the real API, design system primitives established. |
| **M4 — Dashboard + Forecast + Portfolio Working** | Phases 6–8 complete: all three feature areas functional against real backend data, forecast generation/staleness working, portfolio refresh working. |
| **M5 — App-Demonstrable MVP** | Phases 9–10 complete: demo onboarding fully functional, all PRD §19 acceptance criteria pass manually and via automated tests, packaged run mode documented and working. **This milestone does not require any ML-B/C/D work.** |
| **M6 — Scientific ML Acceptance Complete** | ML Spec §20's categorizer and forecaster acceptance gates both pass (ML-B, ML-C complete), winning implementations integrated behind unchanged service contracts (ML-D complete), regression tests re-passed. |
| **M7 — Final Internship-Ready V2** | M5 and M6 both achieved; ML-E's reproducibility/claim-gate checklist passed; README, resume claims, and any demo materials reflect only verified, evidence-backed numbers (ML Spec §21). |

---

## 8. Application Phases

Format per the authoring prompt's required 12 elements, condensed where a field would otherwise repeat verbatim across phases (V1-files-untouched and anti-scope-creep lists especially share common items, stated once in §1/§4/§5 above and referenced, not re-typed 11 times in full).

---

### PHASE 0 — Repo / Dependency / V2 Foundation

**1. Purpose:** Establish a working, reproducible V2 foundation (dependencies, folder skeleton, test fixtures) without implementing any business feature.

**2. Frozen requirements traced:** TRD §2 (repo structure), §16 (config/dependencies), §17.4/§17.5 (fixture/ML-boundary testing), §18.1 (V1 preservation); ML Spec §19 (reproducibility).

**3. Prerequisites:** none (first phase).

**4. Files expected to create:** `backend/` (empty package skeleton: `__init__.py` files only, no logic), `frontend/` (not yet scaffolded — Vite init happens in Phase 5, but the empty directory and a placeholder `.gitkeep` may be created here for structure clarity), `requirements.txt` (re-written, UTF-8, pinned to actually-installed versions plus new backend deps: `fastapi`, `uvicorn`, `pydantic` at versions compatible with the installed Python 3.14), `tests/fixtures/td_csv/*.csv` (four fixtures per §2.4.A), `tests/fixtures/td_csv/README.md`, `tests/fixtures/build_test_kmeans_model.py`, `tests/fixtures/README.md` (per §2.4.B), `.env.example` (TRD §16), `db/migrations/` (empty directory, populated in Phase 1).

**5. Files expected to modify:** `.gitignore` (add explicit test-fixture exceptions/clarifying comments per §2.4.B; do **not** remove the blanket `*.pkl` rule for production artifacts).

**6. V1 files that must remain working/untouched:** `main.py`, `config.py`, `pipeline/*`, `db/database.py`, `db/schema.sql`, `db/seed_synthetic_data.py`, `viz/*` — none are touched in this phase; `python main.py` must still run after this phase (manual verification, below).

**7. Implementation tasks:**
   1. Diagnose and fix `requirements.txt`'s encoding (re-save as UTF-8 plain text).
   2. Re-pin `requirements.txt` against actually-installed package versions (`pip freeze` in the working environment), adding `fastapi`, `uvicorn[standard]`, `pydantic` at versions compatible with Python 3.14 (verify compatibility before pinning — do not guess versions).
   3. Create `backend/` package skeleton (no logic yet).
   4. Author the four TD CSV fixtures + fixture README (§2.4.A), explicitly labeled synthetic/TD-format-shaped.
   5. Author `build_test_kmeans_model.py`, verify it runs and produces a loadable artifact.
   6. Update `.gitignore` per §2.4.B.
   7. Create `.env.example` (TRD §16: `V2_DB_PATH`, `FRONTEND_ORIGIN`, `LOG_LEVEL`).
   8. Document, in a short `backend/README.md` (or a section of the root README), the local dev setup steps this phase enables.

**8. Tests to add/run:** a single smoke test asserting `build_test_kmeans_model.py` produces a file `joblib.load`-able with the expected keys (`kmeans`, `scaler`, `vectorizer`, `cluster_to_category`); a smoke test that all four TD fixtures are valid UTF-8 CSV files parseable by `pandas.read_csv`.

**9. Manual verification:** `python main.py` still runs to completion against V1's schema; `pip install -r requirements.txt` succeeds in a clean virtualenv without errors; `python tests/fixtures/build_test_kmeans_model.py` succeeds.

**10. Acceptance gate:** all of §8's tests pass; §9's manual checks pass; no `backend/` file contains any business logic yet (skeleton only).

**11. STOP/handoff format:** per §14's template.

**12. Do NOT do in this phase:** write any FastAPI route, any repository, any schema DDL, any React code, or touch any V1 pipeline/db file's contents.

---

### PHASE 1 — V2 Database + Repository Foundation

**1. Purpose:** Stand up the complete V2 schema via migrations and a tested repository layer, with no business/API logic yet.

**2. Frozen requirements traced:** TRD §4 (entire persistence design), §4.11 (DDL), §4.12 (migration strategy), §8 (repository layer), §22 decision table (migration source of truth, EMPTY semantics).

**3. Prerequisites:** Phase 0 complete (M0).

**4. Files expected to create:** `db/migrations/001_initial_v2.sql` (TRD §4.11's full DDL verbatim), `backend/db/connection.py` (connection helper: opens `plaincents_v2.db`, sets `PRAGMA foreign_keys = ON`, runs pending migrations), `backend/db/migration_runner.py`, `backend/db/unit_of_work.py` (the `transaction(conn)` context manager, §4 above), `backend/repositories/{transaction,import_batch,staged_transaction,forecast,holding,price_cache,app_state}_repository.py` (TRD §8's list), `tests/backend/repositories/test_*.py` per repository.

**5. Files expected to modify:** none outside `backend/`/`db/migrations/`.

**6. V1 files untouched:** `db/database.py`, `db/schema.sql` (TRD §18.1/§18.2 — V2 never imports these).

**7. Implementation tasks:**
   1. Write `001_initial_v2.sql` exactly per TRD §4.11 (all tables, the `v_transactions_effective` view, all indexes).
   2. Implement the migration runner (creates/checks `schema_migrations`, applies pending files in order, each in its own transaction).
   3. Implement the connection helper with `foreign_keys=ON` enforced per-connection.
   4. Implement the unit-of-work helper.
   5. Implement each repository per TRD §8's method signatures, using the app-state read-mapping (§5 above) consistently.
   6. Write repository tests (§8 below) against a temporary/in-memory or fixture SQLite file, never against `plaincents_v2.db` directly in tests.

**8. Tests to add/run — Database/repository tests:** migrations apply cleanly from an empty DB; running migrations twice is idempotent (no duplicate application); `schema_migrations` correctly tracks applied versions; foreign keys are actually enforced (attempt an invalid FK insert, assert it's rejected); `transactions.dedup_key` UNIQUE constraint rejects a duplicate; `forecast_predictions`'s `UNIQUE(forecast_run_id, category, forecast_month)` allows the same `(category, forecast_month)` across two different `forecast_run_id`s but rejects a repeat within one run; `updated_at` changes on update, not on read; each repository's CRUD methods round-trip correctly; the EMPTY/DEMO/REAL read-mapping helper returns an empty list (no filter) when mode is EMPTY and filters correctly for DEMO/REAL.

**9. Manual verification:** open `plaincents_v2.db` in a SQLite browser after running migrations; confirm all tables/views/indexes exist as specified in TRD §4.11.

**10. Acceptance gate:** all repository tests pass; migration runner is idempotent (verified by running it twice in a test); no repository method contains business orchestration (spot-check against TRD §8's "repositories don't decide cross-table changes" rule).

**11. STOP/handoff format:** per §14.

**12. Do NOT do in this phase:** any FastAPI route, any service-layer orchestration (staleness marking, EMPTY→REAL transition logic itself — only the read-mapping helper, not the transition trigger), any React code.

---

### PHASE 2 — FastAPI Foundation + App State

**1. Purpose:** Stand up the FastAPI app, error-handling convention, CORS/dev proxy setup, and the app-state (`EMPTY`/`DEMO`/`REAL`) service — the thin backbone every later phase's routes attach to.

**2. Frozen requirements traced:** TRD §1.7 (dev topology), §5.1 (health), §5.2 (demo status only — not load/clear, per §2.5 above), §15 (error envelope), §16 (config).

**3. Prerequisites:** Phase 1 complete.

**4. Files expected to create:** `backend/main.py` (FastAPI app + lifespan hook — model loading is stubbed/deferred to Phase 3, but the lifespan hook structure is created here), `backend/config.py`, `backend/api/routes/health.py`, `backend/api/routes/demo.py` (status-only endpoint for now, per §2.5), `backend/schemas/common.py` (`ErrorResponse`), `backend/services/app_state_service.py`, `backend/api/error_handlers.py` (maps domain exceptions → TRD §15's envelope/status codes), `tests/backend/api/test_health.py`, `tests/backend/api/test_demo_status.py`.

**5. Files expected to modify:** `backend/db/connection.py` (wire into FastAPI dependency injection).

**6. V1 files untouched:** all.

**7. Implementation tasks:**
   1. Create the FastAPI app with a lifespan context manager (model loading deferred/no-op until Phase 3, but the hook exists so Phase 3 only adds to it, not restructures it).
   2. Implement CORS middleware restricted to the Vite dev origin (TRD §1.7).
   3. Implement the TRD §15 error envelope + exception handlers (400/404/409/422/503/500 mapping).
   4. Implement `GET /api/health` (db status, categorization_model status — reports `"missing"` until Phase 3 wires the real service, data_mode from `AppStateService`).
   5. Implement `AppStateService.get_mode()` / a stub `maybe_transition_to_real()` (used for real starting in Phase 3).
   6. Implement `GET /api/demo/status` (functional now — reads real `app_state.mode`); stub `POST /api/demo/load`/`DELETE /api/demo/clear` to return a clear "not yet implemented" 501 rather than pretending to work (explicitly, so no later phase mistakes this for functional).

**8. Tests to add/run — API integration tests:** `GET /api/health` returns 200 with correct shape; `GET /api/demo/status` returns `EMPTY` on a fresh DB; error-handler unit tests for each mapped status code using a deliberately-raised domain exception.

**9. Manual verification:** `uvicorn backend.main:app --reload` starts without error; hitting `/api/health` and `/api/demo/status` in a browser/curl returns expected JSON.

**10. Acceptance gate:** both endpoints pass tests; error envelope is consistent across a deliberately-triggered 404 and a deliberately-triggered 500.

**11. STOP/handoff format:** per §14.

**12. Do NOT do in this phase:** implement `POST /api/demo/load`/`clear` for real (§2.5); implement transaction/import/forecast/holding routes; write any React code.

---

### PHASE 3 — Transaction CRUD + CategorizationService Bootstrap

**1. Purpose:** Full transaction CRUD via API, including manual-create categorization — the minimum viable `CategorizationService` per §2.1's Option A.

**2. Frozen requirements traced:** TRD §4.1 (transaction schema/view), §5.4 (routes), §6 (schemas), §7.2 (TransactionService), §7.3 (CategorizationService), §11.1–§11.3 (model lifecycle/contract/manual-create); PRD §9.3, §11.4–§11.6.

**3. Prerequisites:** Phases 1–2 complete (M1).

**4. Files expected to create:** `backend/services/categorization_service.py`, `backend/services/transaction_service.py`, `backend/api/routes/transactions.py`, `backend/schemas/transaction.py` (`TransactionCreate`/`Update`/`Response`), `tests/backend/services/test_categorization_service.py`, `tests/backend/services/test_transaction_service.py`, `tests/backend/api/test_transactions.py`.

**5. Files expected to modify:** `backend/main.py` (wire `CategorizationService` into the lifespan hook — loads `tests/fixtures/kmeans_model_test.pkl` in test config, `models/kmeans_model.pkl` in real config), `backend/api/routes/health.py` (report real model status now).

**6. V1 files untouched:** `pipeline/cluster.py`, `pipeline/features.py` (wrapped, not modified, per TRD §3).

**7. Implementation tasks:**
   1. Implement `CategorizationService`: startup load (once), `predict(transaction)`, `predict_batch(rows)`, status reporting (`loaded`/`missing`/`error`) — internal logic reimplements `predict_categories()`'s three steps against the cached artifact (TRD §7.3), never calling `predict_categories()` itself.
   2. Wire model-missing handling: `predict()`/`predict_batch()` raise a domain exception mapped to 503 by Phase 2's error handler.
   3. Implement `TransactionService.create_manual()`: calls `CategorizationService.predict()` first, computes `dedup_key`, persists via `TransactionRepository`, then calls `AppStateService.maybe_transition_to_real()` (TRD §4.5.1) — failure of the categorization call aborts before any row is written (no `predicted_category=NULL` fallback).
   4. Implement `TransactionService.update()`/`delete()`, including the forecast-staleness mutation table (TRD §7.2) — calling a stubbed `ForecastService.mark_stale()` no-op until Phase 7 exists (documented as a deliberate temporary no-op, replaced when Phase 7 lands).
   5. Implement routes: `GET /api/transactions` (filters: date range, category, search, sort, pagination), `POST`, `GET/{id}`, `PATCH/{id}`, `DELETE/{id}`.
   6. Implement `TransactionResponse` including computed `effective_category`/`is_manual_override` (read from `v_transactions_effective`).

**8. Tests to add/run:** unit tests for `CategorizationService` (loaded/missing/error paths, using the Phase 0 test artifact); unit tests for `TransactionService`'s staleness-mutation table (merchant edit does NOT mark stale; amount/date/category/create/delete DO — stubbed forecast call is asserted-called or asserted-not-called per row); API tests for full CRUD, including a 503 on missing-model manual-create, and effective-category correctness after a correction.

**9. Manual verification:** via `curl`/API docs (FastAPI's auto `/docs`), create a transaction manually, confirm `predicted_category` is set; correct its category; confirm `effective_category` updates and `predicted_category` is unchanged.

**10. Acceptance gate:** all tests pass; a manual transaction cannot be created when the test model artifact is intentionally misconfigured to be missing (503 verified); no transaction is ever persisted with a NULL `predicted_category`.

**11. STOP/handoff format:** per §14.

**12. Do NOT do in this phase:** implement bulk TD import (Phase 4 reuses this service, does not duplicate it); implement real forecast staleness (stub only, documented); write React code.

---

### PHASE 4 — TD Import + Categorization Reuse

**1. Purpose:** Complete TD CSV upload → preview → confirm flow, reusing Phase 3's `CategorizationService`.

**2. Frozen requirements traced:** TRD §4.2–§4.4 (import batches, staging, dedup), §5.3 (routes), §7.1 (IngestionService), §10 (full import pipeline design); PRD §9.2/§11.2/§11.3.

**3. Prerequisites:** Phase 3 complete (M1), TD fixtures from Phase 0.

**4. Files expected to create:** `backend/services/ingestion_service.py`, `backend/api/routes/imports.py`, `backend/schemas/import_.py` (`ImportPreview`/`ImportResult`), `tests/backend/services/test_ingestion_service.py`, `tests/backend/api/test_imports.py`.

**5. Files expected to modify:** `pipeline/ingest.py` (**additive only** — add `load_and_clean_from_bytes()`, per TRD §3; `load_and_clean(csv_path, bank)` is not touched).

**6. V1 files untouched:** `pipeline/ingest.py`'s existing function signature and behavior (verified by re-running V1's existing manual test: `python main.py` still works against `data/raw/synthetic_24mo.csv`).

**7. Implementation tasks:**
   1. Add `load_and_clean_from_bytes(file_bytes, bank)` to `pipeline/ingest.py`, delegating to the same column-mapping/date/merchant logic as the existing path.
   2. Implement `IngestionService.parse_and_stage()`: parse bytes → validate → dedup-check (`TransactionRepository.exists_by_dedup_key`) → `CategorizationService.predict_batch()` (reused from Phase 3, not reimplemented) → `StagedTransactionRepository.bulk_create()` → return `ImportPreview`.
   3. Implement `IngestionService.commit_import()`: re-validate against current `transactions` (live re-check, not trusting stale preview flags), persist via unit-of-work (Phase 1's helper), call `AppStateService.maybe_transition_to_real()`, call the (still-stubbed) forecast-staleness hook, mark batch `confirmed`, clean up staged rows.
   4. Implement routes: `POST /api/imports`, `POST /api/imports/{batch_id}/confirm`, `GET /api/imports`, `GET /api/imports/{batch_id}`.
   5. Implement 409 short-circuit when `app_state.mode == 'DEMO'` (demo-conflict path, functional now even though `POST /api/demo/load` itself isn't yet — the conflict check only reads `app_state.mode`, it doesn't need demo-loading to exist).

**8. Tests to add/run:** parser tests against all four Phase 0 TD fixtures (clean, unparseable-dates, wrong-format, duplicate-containing); dedup tests (exact re-import skipped and counted; two same-day/same-amount legitimate rows both kept per TRD §4.4's occurrence-index heuristic); idempotent-confirm test (confirming twice returns the same result, no double-insert); model-unavailable tests — when the categorization artifact is unavailable, `POST /api/imports` preview must still return HTTP 200 with a clear categorization-unavailable indication (TRD §10: preview is read-only and must not block on the model), while `POST /api/imports/{batch_id}/confirm` remains prediction-dependent and must return 503 without committing any transactions; atomicity test (a forced mid-batch failure leaves no partial rows).

**9. Manual verification:** upload each Phase 0 fixture through `/docs`, inspect the preview and confirm results; **perform the real-TD-export verification** (per §2.4.A, outside the repo, using the product builder's own real TD statement) and record the outcome (pass/fail/adjustments needed) in this phase's handoff — this is the step that actually allows PRD §11.3's "TD verified end-to-end" criterion to be marked complete.

**10. Acceptance gate:** all fixture tests pass; dedup/atomicity/idempotency tests pass; the manual real-TD-export check has been performed and its outcome recorded (if it reveals format assumptions were wrong, this phase's scope explicitly includes fixing `BANK_COLUMNS`/`BANK_DATE_FORMATS` for TD only — not RBC/Scotiabank/others).

**11. STOP/handoff format:** per §14.

**12. Do NOT do in this phase:** build a second categorization implementation; add RBC/Scotiabank/BMO/CIBC/National Bank parsing; write React code.

---

### PHASE 5 — Frontend Foundation + Import + Transactions UI

**1. Purpose:** Stand up the React application, the full design system (§3), and the first two functional pages (Import, Transactions) against the real Phase 3/4 APIs.

**2. Frozen requirements traced:** TRD §9 (entire frontend architecture); PRD §11.1–§11.6, §13 (UX requirements).

**3. Prerequisites:** Phases 3–4 complete (M2).

**4. Files expected to create:** `frontend/` full Vite+React+TS scaffold; `frontend/tailwind.config.ts`; shadcn/ui component installs under `frontend/src/components/ui/`; `frontend/src/components/layout/{AppShell,Sidebar,TopNav}.tsx`; `frontend/src/api/client.ts` (typed fetch wrapper + TRD §15 error parsing); `frontend/src/types/*.ts` (mirroring backend schemas, TRD §9.4); `frontend/src/context/AppStateContext.tsx` (§2.5); `frontend/src/components/shared/{LoadingState,EmptyState,DemoBanner,ConfirmDialog,Toast}.tsx`; `frontend/src/pages/{Import,Transactions}.tsx` + their component breakdowns (TRD §9.5); `frontend/vite.config.ts` (dev proxy to `:8000`); `frontend/src/router.tsx`.

**5. Files expected to modify:** none in `backend/`.

**6. V1 files untouched:** all.

**7. Implementation tasks:**
   1. Scaffold Vite+React+TS; install Tailwind, shadcn/ui, Lucide React, `clsx`, `tailwind-merge`, Recharts, TanStack Query, React Router.
   2. Configure the dev proxy (`/api` → `localhost:8000`).
   3. Build the design-system primitives (§3) once: buttons, cards, badges, modal/confirm dialog, loading skeleton, empty-state component, toast pattern.
   4. Build `AppShell`/`Sidebar`/`TopNav` and the five-route `router.tsx` (`/dashboard` placeholder stub, `/transactions`, `/import`, `/forecast` placeholder stub, `/portfolio` placeholder stub — only Import/Transactions are functional this phase; the other three render a simple "coming in Phase N" placeholder, not a broken route).
   5. Build `AppStateContext` (fetches `/api/demo/status`), the `DemoBanner` (rendered per the corrected mode table: DEMO→banner, EMPTY→none, REAL→none).
   6. Build the Import page: file upload, preview display, confirm action, result display, loading states, demo-conflict modal (functional now, since Phase 4's 409 exists — even though demo *loading* isn't built yet, the *conflict* path is fully testable by manually setting `app_state.mode` to `DEMO` in the test DB).
   7. Build the Transactions page: table, filters, search, sort, pagination, create/edit/delete with confirm dialogs, predicted-vs-confirmed category visual distinction (a badge style difference, per TRD §9.8).
   8. Add restrained GSAP entrance animation to at most one or two elements (e.g., transaction table row entrance) — optional, skippable without affecting the acceptance gate.

**8. Tests to add/run — Frontend tests (lightweight tooling, e.g. Vitest + React Testing Library):** Transaction table renders correctly with mock data; category-correction flow updates the UI optimistically/after refetch; delete requires confirmation (dialog must appear, action only fires on confirm); Import page shows preview then result; empty states render when API returns empty collections.

**9. Manual verification:** run both dev servers, click through the entire Import → Transactions flow against Phase 4's real backend using a Phase 0 fixture file; verify predicted/confirmed badges look visually distinct; verify demo-conflict modal appears when `app_state.mode` is manually set to `DEMO` in the DB.

**10. Acceptance gate:** frontend tests pass; manual click-through of Import + Transactions succeeds end-to-end against the real backend; Dashboard/Forecast/Portfolio routes render a placeholder without crashing.

**11. STOP/handoff format:** per §14.

**12. Do NOT do in this phase:** implement `POST /api/demo/load` (still Phase 9); build Dashboard/Forecast/Portfolio's real UI (placeholders only); add Three.js/Vanta/Lenis/particle effects.

---

### PHASE 6 — Overview Dashboard

**1. Purpose:** Functional dashboard using real backend data — current vs. previous calendar month, category breakdown, trend, recent transactions.

**2. Frozen requirements traced:** TRD §5.8, §7.6, §6 (`DashboardSummary`); PRD §11.7.

**3. Prerequisites:** Phase 5 complete (M3); reads Phase 3/4's transaction data.

**4. Files expected to create:** `backend/services/dashboard_service.py`, `backend/api/routes/dashboard.py`, `backend/schemas/dashboard.py`, `frontend/src/pages/Dashboard.tsx` + components (`SpendingOverview`, `CategoryBreakdown`, `SpendingTrend`, `RecentTransactions`), `tests/backend/services/test_dashboard_service.py`, `tests/backend/api/test_dashboard.py`, frontend component tests.

**5. Files expected to modify:** `frontend/src/router.tsx` (replace placeholder with real page).

**6. V1 files untouched:** all (this phase does not touch V1's `monthly_summary` concept — TRD §4.9 already decided against carrying it into V2; this phase computes live from `transactions`).

**7. Implementation tasks:**
   1. Implement `DashboardService.get_summary()`: live SQL aggregation for current/previous calendar month, category breakdown, trend, recent transactions — reads only `TransactionRepository`/`ForecastRepository`/`HoldingRepository` (no ML, no `yfinance`, per TRD §7.6).
   2. Implement `GET /api/dashboard/summary`.
   3. Build the Dashboard page with Recharts for the trend/breakdown visuals, using the design-system primitives from Phase 5 (no new component library introduced).
   4. Wire the empty state (no transactions → onboarding message, not blank charts, per PRD §11.12).
   5. Optional restrained GSAP: one-time card/chart entrance.

**8. Tests to add/run:** `DashboardService` unit tests for current/previous month math (including month-boundary edge cases — e.g., first day of month); API test for the summary shape; frontend test for the empty-state rendering.

**9. Manual verification:** import the Phase 0 fixtures, confirm dashboard totals match a manual calculation.

**10. Acceptance gate:** all tests pass; dashboard never calls `pipeline.forecast`/`pipeline.portfolio`/`yfinance` (verified by inspection/import-graph check, per TRD §22.H).

**11. STOP/handoff format:** per §14.

**12. Do NOT do in this phase:** add forecast/portfolio summary cards yet if their backing services don't exist (they land in Phases 7/8; Dashboard's `forecast_summary?`/`portfolio_summary?` fields are optional per TRD §6 and simply absent until those phases exist — no placeholder fake data).

---

### PHASE 7 — Forecast Service + Forecast UI

**1. Purpose:** Implement the lean interactive forecast path (§2.2) and its UI — generation, viewing, staleness, cold-start, per-horizon display.

**2. Frozen requirements traced:** TRD §4.6 (schema), §5.6 (routes), §7.4 (ForecastService), §12 (forecast integration design); ML Spec §11.1 (multi-step strategy — V1 last-known-history approach shipped as the bootstrap, per ML-A), §13.1 (horizon reporting, informing what the UI must display, not what ML-B evaluates); PRD §9.6/§11.8.

**3. Prerequisites:** Phase 3/4 complete (effective-category transaction data exists); Phase 5's frontend foundation.

**4. Files expected to create:** `backend/services/forecast_service.py`, `backend/api/routes/forecasts.py`, `backend/schemas/forecast.py`, `frontend/src/pages/Forecast.tsx` + components (`ForecastChart`, `CategoryForecastList`, `ColdStartState`, `StaleWarning`, `RunForecastButton`, `ForecastMetadata`), `tests/backend/services/test_forecast_service.py`, `tests/backend/api/test_forecasts.py`, frontend tests.

**5. Files expected to modify:** `pipeline/forecast.py` (**additive only** — add `train_and_predict(monthly_df)` per §2.2; `fit_and_forecast`/`walk_forward_validate` untouched); `backend/services/transaction_service.py` and `backend/services/ingestion_service.py` (replace the Phase 3/4 stubbed staleness no-op with a real call to `ForecastService.mark_stale()`).

**6. V1 files untouched:** `fit_and_forecast()`, `walk_forward_validate()`, `GridSearchCV` usage, and the `__main__` diagnostic block — all remain byte-for-byte as V1 shipped them; `python -m pipeline.forecast` still runs unchanged.

**7. Implementation tasks:**
   1. Add `train_and_predict(monthly_df)` to `pipeline/forecast.py`: aggregate (reused `aggregate_monthly`) → one `RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)` fit → generate +1/+2/+3 predictions using the existing last-known-history feature construction (V1's current multi-step approach, shipped as the ML-A bootstrap per ML Spec §11.1 — **not yet the scientifically-evaluated final strategy**, which is ML-C/D work). **This function must not call `walk_forward_validate` or `GridSearchCV` — verified by the import-graph check in §2.2.**
   2. Implement `ForecastService.check_status()` (DB-only: `TransactionRepository.count_distinct_months`, latest run's `is_stale`), `get_latest()` (DB-only read), `run_forecast()` (the only method touching `pipeline.forecast`; persists via the Phase 1 unit-of-work helper), `mark_stale(reason)` (flips `is_stale` on the latest non-stale run only, per TRD §12.4).
   3. Set `forecast_runs.model_impl_version = "rf_v1_default_hparams"` (ML Spec §18's naming convention) on every run created by this bootstrap implementation.
   4. Implement per-category `is_available`/`unavailable_reason` (categories that don't survive `build_forecast_features`'s rolling/lag `dropna` are marked unavailable, not fabricated as zero — TRD §12.5).
   5. Implement routes: `GET /api/forecasts/latest`, `GET /api/forecasts/status`, `POST /api/forecasts/run` (422 if cold-start, per TRD §15).
   6. Wire the real staleness hook into `TransactionService`/`IngestionService` (replacing Phase 3/4's stub).
   7. Build the Forecast UI: cold-start message, stale warning banner, manual "Generate/Refresh Forecast" button (loading state while running), per-category +1/+2/+3 display, "generated at" timestamp, per-category unavailable messaging.

**8. Tests to add/run:** unit test asserting `train_and_predict` never calls `walk_forward_validate`/`GridSearchCV` (mock/patch-based); staleness-decision-table tests (already partly covered in Phase 3, extended to confirm the real hook fires); cold-start test (< 12 months → 200 status, structured payload); run-retention test (two consecutive `run_forecast()` calls create two distinct `forecast_runs` rows, both queryable); API test that `GET /api/forecasts/latest`/`status` never trigger any fit (verified via a call-count assertion on `train_and_predict`); frontend tests for cold-start/stale/loading states.

**9. Manual verification:** with the Phase 0/4 fixture data imported (must have ≥12 months for a non-cold-start test), click "Generate Forecast," observe the loading state resolve quickly (sub-second to low-single-digit-seconds per TRD §12.3/ML Spec's expectation — flag if it is not), reload the page, confirm no re-fit occurs; edit a transaction's category, confirm the forecast now shows "stale."

**10. Acceptance gate:** all tests pass; the import-graph/call-count checks confirming no walk-forward/GridSearchCV on the interactive path both pass; staleness correctly triggers per the TRD §7.2/§12.4 mutation table.

**11. STOP/handoff format:** per §14.

**12. Do NOT do in this phase:** run any ML-B bake-off experiment; change the RF hyperparameters from the TRD-frozen defaults; implement recursive/direct multi-step strategies (ML-B/C work, not app-track work) — V1's last-known-history approach ships as-is for the MVP, explicitly labeled as the ML-A bootstrap, not a scientifically final choice.

---

### PHASE 8 — Portfolio CRUD + Price Refresh

**1. Purpose:** Persistent holdings CRUD with manual-only price refresh, separated from read.

**2. Frozen requirements traced:** TRD §4.8, §5.7, §7.5, §13; PRD §9.7/§11.9.

**3. Prerequisites:** Phase 2 (DB/app-state foundation); independent of Phases 6–7.

**4. Files expected to create:** `backend/services/portfolio_service.py`, `backend/api/routes/holdings.py`, `backend/schemas/holding.py`, `frontend/src/pages/Portfolio.tsx` + components (`HoldingsTable`, `AddHoldingForm`, `EditHolding`, `DeleteConfirm`, `RefreshPricesButton`, `PriceStatus`), `tests/backend/services/test_portfolio_service.py`, `tests/backend/api/test_holdings.py`, frontend tests.

**5. Files expected to modify:** `pipeline/portfolio.py` is **not modified** — `fetch_price()` is reused as-is (TRD §3); no signature change needed since the separation happens at the service layer, not inside `pipeline/portfolio.py` itself.

**6. V1 files untouched:** `pipeline/portfolio.py` entirely; `build_portfolio()` is never called by V2 code (its DB-insert-inside-fetch coupling is bypassed, per TRD §3).

**7. Implementation tasks:**
   1. Implement `PortfolioService.get_holdings_with_prices()`: `HoldingRepository.list()` JOIN `PriceCacheRepository.get_last_known()` — zero calls into `fetch_price`/`yfinance`.
   2. Implement `PortfolioService.refresh_prices()`: calls `pipeline.portfolio.fetch_price()` per ticker, tolerates per-ticker failure, updates cache.
   3. Implement CRUD (`create_holding` — includes the `EMPTY → REAL` transition check per §2.1's pattern — `update_holding`, `delete_holding`).
   4. Implement routes: `GET/POST/PATCH/DELETE /api/holdings`, `POST /api/holdings/refresh-prices`.
   5. Build the Portfolio UI: holdings table with last-known price + `price_last_updated` timestamp always visible, explicit "Refresh Prices" button (loading state, per-ticker failure surfaced without breaking the whole page), add/edit/delete with confirmation on delete, empty state when no holdings exist.

**8. Tests to add/run:** unit test asserting `GET /api/holdings` never calls `fetch_price`/`yfinance` (mock/patch-based call-count assertion, mirroring Phase 7's forecast check); refresh test with a simulated per-ticker failure (one ticker fails, others succeed, response reflects both); CRUD tests; ticker validation test (positive shares, non-negative avg_cost).

**9. Manual verification:** add a holding, confirm it shows with no price initially (or a "never refreshed" state), click Refresh Prices, confirm price/timestamp populate; edit shares, confirm P&L recalculates.

**10. Acceptance gate:** all tests pass; the no-network-on-GET assertion passes; a simulated single-ticker failure does not 500 the whole endpoint.

**11. STOP/handoff format:** per §14.

**12. Do NOT do in this phase:** add automatic/scheduled price refresh; add price history charting (TRD §4.8 explicitly defers this).

---

### PHASE 9 — Demo / Onboarding

**1. Purpose:** Fully functional demo load/clear, EMPTY onboarding UI, and the complete DEMO/REAL mutual-exclusion flow — the first phase where `POST /api/demo/load`/`DELETE /api/demo/clear` become real (§2.5).

**2. Frozen requirements traced:** TRD §4.5 (entire demo isolation design), §5.2, §7.7 (DemoService), §14; PRD §10a.

**3. Prerequisites:** Phases 3, 4, 7, 8 complete (M4) — demo seeding needs transaction/forecast/holding data shapes to populate.

**4. Files expected to create:** `backend/services/demo_service.py`, `backend/scripts/seed_v2_demo_data.py` (TRD §14.1 — reuses V1 seed script's *data-generation patterns*, not its code, calling V2 repositories and stamping `data_mode='demo'`), full implementation of `backend/api/routes/demo.py`'s load/clear endpoints, `frontend/src/components/OnboardingEmptyState.tsx`, `tests/backend/services/test_demo_service.py`, `tests/backend/api/test_demo.py`, frontend tests.

**5. Files expected to modify:** `frontend/src/context/AppStateContext.tsx` (wire real load/clear actions, replacing Phase 5's UI-hooks-only wiring); `backend/api/routes/demo.py` (replace Phase 2's 501 stubs).

**6. V1 files untouched:** `db/seed_synthetic_data.py` remains as V1's own standalone script (TRD §3, §18.1) — the new `backend/scripts/seed_v2_demo_data.py` is a separate file, not a modification of it.

**7. Implementation tasks:**
   1. Write `seed_v2_demo_data.py`: generates ≥12 months of demo transactions across all 8 categories (reusing V1's merchant/amount-range/seasonal-multiplier patterns), at least one demo holding with a seeded `price_cache` entry, and — critically — **a prebuilt `forecast_runs`/`forecast_predictions` set** so the Forecast page shows populated results immediately on demo load without requiring the user to click "Generate" (per the authoring prompt's explicit requirement). All rows stamped `data_mode='demo'`.
   2. Implement `DemoService.load_demo()`: reject with 409 if `mode == 'REAL'`; otherwise run the seed script's logic inside a unit-of-work transaction, set `mode = 'DEMO'`.
   3. Implement `DemoService.clear_demo()`: atomic deletion across all `data_mode='demo'` rows (transactions, holdings, forecast_runs+predictions, demo-exclusive price_cache entries) inside one unit-of-work transaction, set `mode = 'EMPTY'`.
   4. Wire `POST /api/imports`'s existing 409-on-DEMO path (already built in Phase 4) to the now-real demo state.
   5. Build the onboarding empty-state UI (offers "Import real data" vs. "Load demo data," clearly distinct, per PRD §10a) and the demo-clear confirmation dialog (destructive-action confirmation, per TRD §9.8).
   6. Verify the demo→real sequence end-to-end: load demo → attempt real import → 409 → confirm → demo cleared → import proceeds → mode becomes REAL.

**8. Tests to add/run:** `load_demo()` rejected with 409 when real data exists (no data deleted); `clear_demo()` removes all and only demo-flagged rows across every table, leaves any real data untouched if present; full demo→real sequence integration test; demo forecast is present immediately after load (no generation call needed); EMPTY/DEMO/REAL banner-rendering tests (already partly covered in Phase 5, now against the real endpoint).

**9. Manual verification:** on a fresh DB, click "Load Demo Data," confirm Dashboard/Forecast/Portfolio all show populated, clearly-labeled demo content; attempt a real import, confirm the conflict prompt, confirm, verify demo is fully gone and the real import proceeds.

**10. Acceptance gate:** all tests pass; demo data never appears unlabeled anywhere in the UI; real data is never deleted by any demo-load attempt; full reset is atomic (a forced mid-reset failure test leaves no partial demo remnants).

**11. STOP/handoff format:** per §14.

**12. Do NOT do in this phase:** allow `load_demo()` a "force" option that deletes real data under any circumstance; present demo forecast/accuracy numbers as ML evidence anywhere in the UI or docs.

---

### PHASE 10 — Integration / UX / E2E / MVP Polish

**1. Purpose:** Cross-feature regression protection, packaged reviewer run mode, accessibility/responsive polish, and final README — the phase that closes out M5 (App-Demonstrable MVP).

**2. Frozen requirements traced:** TRD §1.7 (packaged run mode, §2.3 above), §17.3 (API integration test list), §20 (security/privacy proportionality); PRD §19 (acceptance criteria — final verification pass), §20 (success criteria).

**3. Prerequisites:** Phases 0–9 complete (M4).

**4. Files expected to create:** `tests/e2e/*.spec.ts` (Playwright — the specific high-value flows listed in §9 below), `backend/main.py` additions for `StaticFiles` mount (packaged mode), `README.md` (rewritten — development vs. reviewer run mode, per §2.3), `docs/screenshots/` (optional, for README embedding).

**5. Files expected to modify:** any file with a discovered cross-feature regression bug found during this phase's integration pass (documented per-fix in the handoff, not a blanket rewrite).

**6. V1 files untouched:** verified one final time — `python main.py`, `python -m pipeline.cluster`, `python -m pipeline.forecast`, `python db/seed_synthetic_data.py`, `python viz/report.py`, `python viz/powerbi_export.py` all still run (TRD §18.1's five commands).

**7. Implementation tasks:**
   1. Run the full backend/frontend test suites together, fix any cross-phase regression found (e.g., a Phase 7 staleness hook interaction with Phase 4's import that wasn't caught in isolation).
   2. Implement the packaged run mode: `vite build` output served via FastAPI `StaticFiles` at `/`, API under `/api/*`.
   3. Write the 4 Playwright E2E flows (§9).
   4. Accessibility basics pass (contrast, focus states, form labels — not a full WCAG audit, proportional to a personal project).
   5. Responsive review at common laptop/tablet widths (desktop-first, per TRD §9's UX principles — mobile not required).
   6. Write the final README: setup, development run mode, reviewer/demo run mode, data privacy notice (no real bank data committed), current MVP status, explicit note that ML acceptance (M6) is a separate, ongoing track from the app (M5).
   7. Verify all five V1 commands from §6 still work.

**8. Tests to add/run — E2E (Playwright), exactly four flows, no more:**
   - Clean app → load demo → verify populated → reset demo → verify EMPTY again.
   - TD import: upload fixture → preview → confirm → transactions visible with predicted categories.
   - Edit a transaction's category → verify forecast becomes marked stale.
   - Create a holding → explicit refresh → verify price/timestamp populate.

**9. Manual verification:** run the packaged single-process mode end-to-end as a fresh reviewer would (no dev-mode setup); walk through every PRD §19 acceptance criterion manually once, checking each off.

**10. Acceptance gate:** all four E2E flows pass; every PRD §19 acceptance criterion passes manual verification; the packaged run mode works standalone; all five V1 commands still work; README accurately describes both run modes.

**11. STOP/handoff format:** per §14 — this phase's handoff additionally states explicitly whether **M5 (App-Demonstrable MVP)** is achieved.

**12. Do NOT do in this phase:** run or claim any ML Spec §20 acceptance gate result (that's the ML track, M6/M7, entirely separate); add cloud deployment; expand the E2E suite beyond the four listed flows without a specific justified gap found during this phase.

---

## 9. Parallel ML Track

### ML-A — App Bootstrap
**When it happens:** inside Phases 3 (categorization) and 7 (forecasting) above — not a separate calendar phase, a label for work already embedded in the app track.
**Deliverable:** current K-Means/RF integrated behind `CategorizationService`/`ForecastService`. **No claim of scientific finality** — every artifact/log/UI string referencing these models is neutral ("current implementation"), never "final" or "validated."

### ML-B — Scientific Evaluation
**When it happens:** in parallel with any/all app-track phases, starting as soon as ML Spec §3.2's data acquisition begins (no app-track dependency).
**Tasks (ML Spec §3, §5, §6, §11, §12):**
1. Acquire the independent categorization evaluation dataset (Tier A/B/C per ML Spec §3.2) — this is the item most likely to take real calendar time and has no fixed start-blocking dependency on app phases.
2. Freeze the merchant-grouped, category-stratified TRAIN/VALIDATION/TEST split (ML Spec §6) and version it (§18/§19).
3. Implement K-Means's TRAIN-only fitting/mapping evaluation harness (ML Spec §6.1).
4. Implement and run the frozen 3-candidate categorization bake-off (K-Means, Logistic Regression, Linear SVM).
5. Implement the explicit calendar-month-boundary expanding-window evaluation harness (ML Spec §12) — reusing `walk_forward_validate`'s proven-correct loop structure, extended across a hyperparameter grid.
6. Implement and run the frozen 4-candidate forecast bake-off (naive, seasonal naive, RF at the TRD-frozen interactive hyperparameters, Ridge), evaluated at +1/+2/+3 individually and combined (ML Spec §13.1).
7. Run the multi-step strategy experiment (ML Spec §11.1: current V1 approach vs. recursive vs., if warranted, direct).
8. Run the history-length (§15) and per-category sparsity (§16) experiments.
9. Complete structured error analysis for categorization (§8).

### ML-C — Model Selection
**When it happens:** after ML-B produces results.
**Tasks:** apply ML Spec §7's primary metric (macro F1) for categorization and §14's rule for forecasting; explicitly document the outcome even if it is "the current baseline wins" or "the naive baseline wins" — per ML Spec §0/§14, this is a legitimate, expected-possible outcome, not a failure of the process.

### ML-D — Final Integration
**When it happens:** only if ML-C selects something other than the current bootstrap implementation.
**Tasks:** swap the winning implementation behind the **unchanged** `CategorizationService`/`ForecastService` contracts (TRD §11.4/§12.2 — no route/schema/frontend change required by design); rerun the full Phase 1–10 backend/API/E2E regression suite; update `model_impl_version`/artifact metadata (ML Spec §18).

### ML-E — Final Reproducibility / Claim Gate (Build-Plan bookkeeping only, per the authoring prompt's explicit constraint)
**When it happens:** after ML-D (or after ML-C if ML-C kept the bootstrap implementation and ML-D was a no-op).
**Tasks:** reproduce ML-B/C's final reported numbers from a clean environment (ML Spec §19); assemble the final evaluation report/artifacts; walk the ML Spec §20 acceptance-gate checklists item by item and confirm every box; verify every resume/interview claim against ML Spec §21's policy before it is used anywhere. **This does not reinterpret or add new model-development work** — it is a verification/documentation checkpoint only, consistent with the authoring prompt's explicit constraint on ML-E's meaning.

---

## 10. MVP Acceptance Matrix

| PRD §19 Acceptance Criterion | Build Plan Phase | Component | Automated Test | Manual Verification |
|---|---|---|---|---|
| Upload TD CSV → transactions with predicted category | Phase 4 | `IngestionService`, `CategorizationService` | Phase 4 fixture/dedup tests | Phase 4, Phase 10 |
| Edit transaction, persists | Phase 3 | `TransactionService` | Phase 3 API tests | Phase 5 |
| Delete transaction, disappears everywhere | Phase 3 | `TransactionService` | Phase 3 API tests | Phase 5 |
| Category correction reflected in dashboard, prediction retained | Phases 3, 6 | `TransactionService`, `DashboardService` | Phase 3 + Phase 6 tests | Phase 6 |
| Forecast generate, reopen without retrain | Phase 7 | `ForecastService` | Phase 7 call-count test | Phase 7 |
| Cold-start message (not error) | Phase 7 | `ForecastService.check_status` | Phase 7 cold-start test | Phase 7 |
| Holding CRUD + P&L, no auto network | Phase 8 | `PortfolioService` | Phase 8 no-network-on-GET test | Phase 8 |
| Explicit refresh vs. page-open | Phase 8 | `PortfolioService` | Phase 8 refresh test | Phase 8 |
| Demo offer, populated on load | Phase 9 | `DemoService` | Phase 9 seed/load test | Phase 9, Phase 10 E2E |
| Demo→real conflict confirmation | Phases 4, 9 | `IngestionService`, `DemoService` | Phase 9 sequence test | Phase 9, Phase 10 E2E |
| Duplicate import skipped + reported | Phase 4 | `IngestionService` | Phase 4 dedup tests | Phase 4 |
| Forecast marked stale after mutation | Phases 3, 4, 7 | `TransactionService`/`IngestionService` → `ForecastService` | Phase 7 staleness tests | Phase 7, Phase 10 E2E |
| No unhandled error on empty screens | Phases 3, 4, 6, 7, 8, 9 | all `GET` endpoints/pages | per-phase empty-state tests | Phase 10 |

**Major TRD invariants → phases/tests:**

| TRD Invariant | Phase | Test |
|---|---|---|
| Same TD CSV imported twice → all skipped | Phase 4 | Dedup test |
| Two legitimate same-day/amount purchases both kept | Phase 4 | Dedup occurrence-index test |
| Manually created transaction categorized | Phase 3 | Manual-create test |
| Predicted category corrected, original retained | Phase 3 | Effective-category test |
| Confirmed category cleared, reverts to predicted | Phase 3 | Effective-category test |
| Demo → real transition | Phase 9 | Full sequence test |
| Real → demo rejected | Phase 9 | 409 test |
| Two forecast runs, same category/month, no conflict | Phase 7 | Run-retention test |
| Transaction deletion after forecast marks stale, doesn't alter history | Phase 7 | Staleness test |
| Stale cached price still displayable | Phase 8 | Portfolio read test |
| Missing categorization artifact → 503, existing data readable | Phase 3 | Model-missing test |
| Restart during import preview | Phase 4 | (documented as durable-by-design via SQLite; explicit restart-simulation test optional per time) |

**ML Spec §20 acceptance gates → ML phases:** every categorizer gate item → ML-B (execution) + ML-C (documented decision); every forecaster gate item → ML-B (execution, including the +1/+2/+3 and calendar-boundary requirements) + ML-C (documented decision); artifact/version metadata → ML-D; reproducibility/claim verification → ML-E.

---

## 11. Risk Register

| # | Risk | Probability/Impact (qualitative) | Mitigation | Addressed in |
|---|---|---|---|---|
| 1 | V1's append-only DB semantics leak into V2 CRUD | Low probability (separate DB/repo layer built from scratch) / High impact if it happened | V2 never imports `db/database.py`; repositories built fresh against `schema_v2` tables | Phase 1 |
| 2 | V1 forecast code (`fit_and_forecast`/walk-forward/GridSearchCV) accidentally runs on a user request | Medium probability if not explicitly guarded / High impact (unusable UX) | `train_and_predict()` is a separate new function; import-graph/call-count test enforces separation | Phase 7 |
| 3 | Missing/stale K-Means artifact | Medium probability (artifact is gitignored, easy to forget) | Startup status reporting, 503 on prediction-dependent writes, test-artifact bootstrap script | Phase 0, 3 |
| 4 | Real TD CSV differs from current fixture/format assumptions | Medium probability (V1's TD format was never verified against a real export) | Explicit manual verification step in Phase 4's acceptance gate; scope limited to fixing TD-only assumptions if needed | Phase 4 |
| 5 | Dedup heuristic false positive/negative | Medium probability (occurrence-index heuristic is a known limitation, TRD §4.4) | Documented limitation, not silently ignored; tests cover the two known edge cases explicitly | Phase 4 |
| 6 | Import confirmation fails midway | Low probability / High impact if unguarded | Unit-of-work transaction wraps the whole commit | Phase 1, 4 |
| 7 | Demo reset partial failure | Low probability / High impact (could leave demo/real data mixed) | Unit-of-work transaction wraps the whole reset; explicit test for a forced mid-reset failure | Phase 1, 9 |
| 8 | SQLite connection/foreign-key mistakes | Medium probability (FK pragma is per-connection, easy to forget) | Foreign keys set on every connection at the helper level, tested explicitly | Phase 1 |
| 9 | `yfinance` partial outage/rate limits | Medium probability (external dependency) / Low impact (per-ticker failure isolated) | Per-ticker try/except already in V1's `fetch_price`, reused as-is; failure surfaced without breaking the page | Phase 8 |
| 10 | Frontend scope/polish consuming too much time | Medium probability (design-system ambition is real) / Medium impact | GSAP/motion explicitly optional; design system built once in Phase 5, reused not rebuilt | Phase 5, 10 |
| 11 | ML evaluation dataset unavailable/too small | Medium-high probability (no dataset exists yet, acquisition is nontrivial) / blocks M6, not M5 | Explicitly non-blocking for app track (ML Spec §24); M5 achievable without it | ML-B |
| 12 | +2/+3 forecast performance weak | Medium probability (known V1 limitation, unevaluated) / Medium impact on scientific claims, none on app functionality | ML Spec §11.1/§14 explicitly plan for this outcome; shipping the simpler/naive result at weak horizons is an acceptable, documented outcome | ML-B, ML-C |
| 13 | V1 compatibility broken by refactors | Low probability (refactors are additive-only per TRD §3 classifications) / High impact if it happened | Every phase's "V1 files untouched" section is a checked item; Phase 10 re-verifies all five V1 commands | Every phase, Phase 10 |
| 14 | Resume claims outpace scientific evidence | Medium probability (temptation exists) / reputational impact | ML Spec §21's explicit disallowed-claims list; ML-E is the dedicated verification gate before any claim is used | ML-E |

---

## 12. Testing Strategy Summary

- **Backend unit tests:** every service and repository, added in the phase that creates them (never deferred to a "testing phase").
- **API integration tests:** every route, added in the phase that creates it.
- **Database tests:** migrations, constraints, atomicity — Phase 1, extended as new tables' invariants are exercised in later phases.
- **Frontend tests:** lightweight (Vitest + React Testing Library), component/flow-level, added per-phase from Phase 5 onward.
- **E2E:** exactly four Playwright flows, all in Phase 10, chosen for maximum regression value (demo cycle, import cycle, staleness cycle, portfolio refresh cycle) — deliberately not exhaustive.
- **ML tests:** application-boundary tests (Phase 3/7) use the Phase 0 deterministic test artifact, never retraining production models; scientific evaluation tests/scripts belong to ML-B, outside the app-track test suite entirely.

---

## 13. Time Estimates (ranges, not commitments)

| Phase | Estimate |
|---|---|
| Phase 0 | 2–4 hours |
| Phase 1 | 4–7 hours |
| Phase 2 | 2–4 hours |
| Phase 3 | 5–8 hours |
| Phase 4 | 6–10 hours (includes real-TD-export verification time, which is variable) |
| Phase 5 | 8–14 hours (design system + two pages) |
| Phase 6 | 4–6 hours |
| Phase 7 | 6–10 hours |
| Phase 8 | 4–6 hours |
| Phase 9 | 5–8 hours |
| Phase 10 | 6–10 hours |
| **App track total (M5)** | **~52–87 hours** |
| ML-B | Highly variable — dataset acquisition dominates; days to weeks depending on Tier A/B/C path chosen (ML Spec §3.2) |
| ML-C, ML-D, ML-E | 4–10 hours combined, once ML-B's results exist |

These ranges exist to aid scheduling only — architecture correctness is never traded for hitting a number.

---

## 14. Implementation Session Template (for future use — not executed now)

```
Implement PlainCents V2 Build Plan Phase [N] ONLY.

Read:
- docs/V2_PRD.md
- docs/V2_TRD.md
- docs/V2_ML_SPEC.md
- docs/V2_BUILD_PLAN.md

Follow Phase [N] exactly.

Before modifying code:
1. Inspect all files Phase [N] expects to touch.
2. Confirm prerequisites from prior phases are actually satisfied (don't assume).
3. Report any blocking contradiction with the frozen docs before proceeding.

Then implement ONLY Phase [N].

Run every Phase [N] automated test and manual-verification command
that can be run in this environment.

Do not begin Phase [N+1].

At the end, respond with:
- files created
- files modified
- commands run
- test results (pass/fail counts)
- deviations from the plan, if any, and why
- unresolved issues
- whether Phase [N+1] is unblocked

STOP.
```

---

## Final Cross-Document Self-Audit

**A. PRD traceability** — every §19 acceptance criterion and §16 MVP feature appears in §10's matrix, mapped to a specific phase. Confirmed complete.

**B. TRD traceability** — every schema table (§4.11), service (§7), route group (§5), and the twelve schema invariants (§22.F) all appear mapped to a phase in §8/§10. Confirmed complete.

**C. ML Spec traceability** — ML-A through ML-E all appear in §9 with their frozen bake-off sets restated verbatim (3 categorization, 4 forecast candidates, no additions); §20's acceptance-gate items are mapped to ML-B/C/E in §10's matrix.

**D. No architecture drift** — no phase introduces Redis, Celery, task queues, WebSockets, auth, PostgreSQL, Docker, microservices, LLMs, vector databases, automatic retraining, or automatic forecast/portfolio refresh; Three.js/Vanta/Lenis/particle effects are explicitly excluded in §3.

**E. V1 compatibility** — every phase's "V1 files untouched" field is populated and specific; Phase 10 re-verifies all five V1 commands as a dedicated acceptance-gate item.

**F. Dependency order** — §6's graph shows no phase depending on unbuilt work; Phase 3's `CategorizationService` bootstrap explicitly precedes and is reused by Phase 4 (§2.1), resolving the one real sequencing ambiguity the reviewers flagged.

**G. Atomicity** — import confirm (Phase 4), forecast run persistence (Phase 7), and full demo reset (Phase 9) each have an explicit unit-of-work requirement (§4) and a corresponding atomicity test in their phase.

**H. Frontend design system** — Tailwind + shadcn/ui + Lucide + Recharts are established once in Phase 5 and reused in Phases 6–9; GSAP is explicitly optional; Three.js/Vanta/Lenis are explicitly excluded (§3).

**I. Test coverage** — every phase includes backend/API/frontend tests as appropriate to its scope (§8's per-phase "Tests to add/run" fields); no phase defers all testing to Phase 10 — Phase 10 adds only E2E and regression re-verification.

**J. App vs. science** — M5 (App-Demonstrable) and M6 (Scientifically Final) are distinct milestones (§7) with M5 achievable entirely without any ML-B/C/D work (explicitly stated in Phase 10's acceptance gate and Risk #11's mitigation).

**K. Claim discipline** — no phase's acceptance gate or manual verification step requires or produces a resume-ready performance claim; ML-E is the single dedicated gate for that, per ML Spec §21.

**L. Phase executability** — every phase has prerequisites limited to prior phases' concrete deliverables, a bounded file list, and an explicit STOP condition (§14's template) — no phase requires "half the application" to exist first beyond its stated, minimal prerequisites.

**No blocking contradiction was found between this Build Plan and the three frozen documents.**

---

*No production code, migrations, frontend/backend source files, or model artifacts were created or modified in the production of this document. No dependencies were installed. Only `docs/V2_BUILD_PLAN.md` was created; `.gitignore`'s current state was inspected, not modified.*
