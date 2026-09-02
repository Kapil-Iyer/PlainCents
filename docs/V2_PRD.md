# PlainCents V2 — Product Requirements Document

**Status:** FROZEN — V2 Product Requirements
**Date:** 2026-09-01
**Scope of this document:** Product requirements only. No architecture, schema, or implementation decisions beyond what is required to state product behavior. Technical design belongs in the V2 TRD.

Technical implementation is governed by the forthcoming V2 TRD. Changes to frozen product scope require an explicit PRD amendment.

---

## 1. Document Purpose

This document defines **what PlainCents V2 is and what users can do with it**. It is the product-level source of truth that the V2 TRD, ML specification, DB schema, API contracts, and Build Plan must trace back to.

It is based on:
- The actual V1 codebase (`pipeline/`, `db/`, `viz/`, `main.py`, `config.py`)
- The V1 README
- The V1 PRD (`PRD_and_Buildplan/PlainCents_PRD.txt`) and Build Plan (`PRD_and_Buildplan/PlainCents_BuildPlan.txt`)
- `docs/TECHNICAL_WALKTHROUGH.md`
- `docs/V2_PREBUILD_REALITY_CHECK.md` (the forensic V1→V2 reusability audit)

This document does not describe how V2 will be built. It describes what V2 must do for a user.

---

## 2. Product Overview

PlainCents V1 is a Python batch pipeline: a bank CSV is dropped into `data/raw/`, run through `main.py`, and produces categorized transactions, a 3-month spending forecast, portfolio P&L, a SQLite warehouse, a static PDF report, and PowerBI CSV exports. There is no persistent CRUD data model, no UI, and no notion of a returning user — every run is a batch session identified by `session_id`.

PlainCents V2 turns this into an interactive, single-user, full-stack personal-finance application. A user imports their own bank transactions, reviews and corrects ML-assigned categories, manages their transaction history directly, views an interactive dashboard of their spending, generates and reviews a spending forecast, and tracks investment holdings with live P&L — all through a persistent application, not a one-shot script.

---

## 3. Problem Statement

Personal-finance transaction exports contain useful spending history, but that history requires cleaning, categorization, and aggregation before it becomes actionable. Doing this by hand is tedious and inconsistent; doing it once as a script does not give a user anything to return to, correct, or build on over time.

PlainCents V2 provides a local workflow for importing those transactions, correcting ML-assisted categories, analyzing spending, forecasting category-level expenses, and tracking investments in one persistent application.

V1 proved the ML pipeline (categorization + forecasting) works end-to-end on synthetic data but is not usable as a product: it has no persistence model a user can edit, no way to import their own data through anything but a filesystem drop, and no interface besides a PDF and PowerBI file regenerated from scratch on every run.

---

## 4. Product Vision

> "PlainCents V2 is a personal-finance analytics application that lets users import Canadian bank transactions, review and correct ML-assisted transaction categories, manage their transaction history through CRUD workflows, understand spending through an interactive dashboard, forecast future category-level spending, and track investment holdings."

V1: `CSV → batch pipeline → ML → SQLite → static reports`

V2: `User → interactive application → persistent financial data → ML / analytics`

V2 must feel like a coherent personal-finance product, not a React front end bolted onto `main.py`.

---

## 5. Goals

- Give a user a real, persistent home for their transaction history that they can search, edit, and correct over time.
- Preserve V1's two core ML capabilities — categorization and forecasting — as product features, not scripts.
- Make categorization corrections a first-class, low-friction user action, without ever silently discarding what the model originally predicted.
- Present a real dashboard that answers "what is happening with my money" without requiring PowerBI or a PDF.
- Turn portfolio holdings into data the user manages directly.
- Ship a narrow, coherent, demonstrable MVP rather than a wide, half-finished feature set.

## 6. Non-Goals (for this document's product scope)

- V2 does not aim to replace or outperform commercial budgeting apps on breadth of features.
- V2 does not aim to support every Canadian bank in the MVP.
- V2 does not aim to provide investment advice, tax advice, or automated trading.
- V2 does not aim to be multi-user or hosted for the general public in the MVP.

---

## 7. Target User

**MVP target user:** a single user (initially the product's own builder, and by extension an internship/portfolio-review audience) who holds a TD account and has a TD CSV export available, wants to understand and forecast their own spending by category, wants to correct auto-categorization rather than categorize everything by hand, and optionally tracks a small personal investment portfolio.

**Longer-term V2 target persona:** the same user profile, generalized to anyone holding an account at one of Canada's Big Six banks (TD, RBC, Scotiabank, BMO, CIBC, National Bank). This broader persona describes the product's direction, not a capability the MVP claims to deliver — see §9.2 and §17 for which banks are actually supported at MVP versus added later.

V2 is explicitly a **local-first, single-user** product for this MVP. It is not designed for multi-tenant or public deployment (see §6 Non-Goals and §18 Explicitly Out of Scope).

---

## 8. V1 → V2 Evolution

| Dimension | V1 (today, verified in code) | V2 (target) |
|---|---|---|
| Interface | None — CLI script (`main.py`) + generated PDF/CSV | Interactive web application |
| Transaction storage | Append-only insert per run, `session_id`-scoped, no update/delete (`db/database.py`) | Persistent, user-editable transactions with full CRUD |
| Categorization | K-Means cluster → majority-vote label, applied at pipeline time (`pipeline/cluster.py`) | Same predicted-category concept, exposed to the user, with an explicit user-correction workflow; categorizer implementation is not locked to K-Means going forward |
| Category truth | Single `category` column; whatever the model outputs is the record | Distinction between what the model predicted and what the user confirmed, with the confirmed value taking precedence |
| Import | Manual CSV drop into `data/raw/`, only TD/RBC/Scotiabank column mapping exists (`config.py` `BANK_COLUMNS`) | In-app import workflow; TD supported end-to-end in MVP; other Big Six banks added incrementally as real formats are verified |
| Forecast | `fit_and_forecast()` retrains a Random Forest and re-runs walk-forward validation on every call (`pipeline/forecast.py`) | Forecast generation is a manual user-triggered action in the MVP; viewing a forecast never triggers retraining |
| Portfolio | Holdings supplied via seed/config, inserted per run (`pipeline/portfolio.py`) | Holdings are persistent, user-managed CRUD entities |
| Reporting | PDF (`viz/report.py`) and PowerBI CSV export (`viz/powerbi_export.py`) are the only outputs | The in-app dashboard is the primary analytics surface; PDF/PowerBI become optional secondary exports |
| Cold start | Forecasting raises an exception below 12 months of history (`pipeline/forecast.py`) | Insufficient history is a normal, explained product state, not an error |

---

## 9. Product Principles

### 9.1 Transactions are first-class entities
Every transaction is persistent, user-visible data — not a byproduct of a batch run. Users can view, search, filter, sort, create, edit, delete, and correct the category of any transaction. V2's user-facing model does not expose V1's `session_id` batch-run concept as a product concept.

### 9.2 Bank import is real, but scoped honestly
Target long-term support is Canada's Big Six (TD, RBC, Scotiabank, BMO, CIBC, National Bank). V1 code today only has column-mapping conventions for TD, RBC, and Scotiabank, and none of these have been verified against real, representative bank exports as part of this audit — this document makes no claim that TD (or any bank) has already been verified. **TD is the only bank required to have a verified, end-to-end import path for the V2 MVP to be considered complete.** RBC, Scotiabank, and the remaining Big Six banks are expansion work. A bank may only be advertised as supported after a representative export has actually been obtained, its parsing verified, its output checked against the common transaction representation, and appropriate tests/fixtures added — not merely because column-mapping code exists for it. The product must normalize any supported bank's transactions into one common representation so downstream behavior (categorization, dashboard, forecasting) never varies by bank.

### 9.2a Duplicate imports are prevented, not silently repeated
If PlainCents detects transactions that appear to have already been imported, they must not be inserted again as duplicates. The import preview/result must report how many transactions were skipped as duplicates. The exact detection mechanism (matching fields, hashing, constraints) is a TRD decision; this document only requires that duplicates are surfaced and not silently duplicated.

### 9.2b Demo data and real data do not mix
Demo/synthetic data and real imported data are mutually exclusive for the MVP. If demo data is currently loaded and the user attempts to import real data, the product must explain that demo data needs to be cleared first and ask for confirmation before proceeding. Clearing demo data is a **full demo reset** — all demo transactions, demo forecasts, demo portfolio state, and related demo artifacts are removed together. Demo data must never silently mix into real financial analytics (dashboard totals, forecasts, portfolio values). The storage mechanism for enforcing this separation is a TRD decision.

### 9.3 Categorization is ML-assisted, not ML-owned
Users receive a **predicted category** and can accept or correct it. The system must always be able to distinguish the model's original prediction from the user's correction, and the **effective category** (what dashboards/forecasts use) is the user's correction when one exists, otherwise the prediction. Correcting one transaction must not trigger model retraining. Which classifier produces the prediction (K-Means, another clustering approach, a supervised classifier, etc.) is explicitly **not decided by this document** — the product only requires that a prediction exists and can be corrected.

This applies identically to manually created transactions: a transaction the user types in directly goes through the same categorization model as an imported one and receives a predicted category. There is no separate "quick-add" transaction type or path that skips prediction. If the user explicitly selects a category while creating it, the model's prediction is still preserved separately, and the user's selection becomes the confirmed/effective category — the same predicted/confirmed/effective relationship described above.

### 9.4 A fixed category taxonomy applies everywhere
The eight V1 categories (Food & Dining, Transport, Rent & Utilities, Entertainment, Healthcare, Shopping, Subscriptions, Other) are the **fixed V2 MVP taxonomy**. The taxonomy is bank-agnostic — no bank's CSV format may introduce bank-specific categories. Custom or user-defined categories are out of scope for the MVP but may be considered in a future version.

### 9.5 The dashboard is the product
The in-app dashboard, not PowerBI, is how a user understands their spending. It must be genuinely useful without ever opening PowerBI or a PDF.

### 9.6 Forecasting is a controlled operation, not a side effect
For the MVP, forecast generation and refresh are **manual only**. Opening or reloading the Forecast page must never itself trigger model training or forecast recomputation. Generating/refreshing a forecast is an explicit user action with a visible "as of" timestamp. Insufficient history is a defined, calm product state, not an error.

Existing forecasts are preserved once generated. If the transaction data or effective categories a forecast was based on change afterward (e.g., new imports, category corrections), the existing forecast is not silently recomputed or discarded — instead it is visibly marked **stale**, and the user may choose to manually refresh it. Refreshing a forecast creates a new **Forecast Run**; previous forecast runs are retained (supporting later forecast-vs-actual comparison), though the MVP UI primarily displays the latest run. How staleness is detected is a TRD-level decision; this document only requires that it is detected and shown.

### 9.7 Portfolio holdings are owned by the user
Holdings (ticker, shares, average cost) are entered, edited, and removed by the user, not supplied through configuration or a seed script, in the running product.

For the MVP, market-price refresh is **manual only**. Opening the Portfolio page displays the latest available/cached values without automatically triggering an external price request; the user explicitly chooses "Refresh Prices" to fetch current values.

### 9.8 Synthetic data is clearly synthetic
Synthetic data has a defined role (development, tests, demos, reproducible fixtures) and must never be presented as evidence of real-world model performance. Any accuracy/error metric shown to a user or reviewer must be labeled by its data source.

### 9.9 Local-first privacy by default
Imported transaction data is not transmitted anywhere beyond what is strictly needed for the feature that requires it (e.g., a market-data lookup for a ticker the user added). The user should always be able to tell when an external service is being used.

### 9.10 No unnecessary authentication surface
No account registration, login, OAuth, or multi-tenant handling is in scope for the MVP.

### 9.11 Scope discipline
No product requirement in this document should introduce chatbots/generative AI features, vector databases, social features, payment processing, message queues/microservices, real-time bank integrations (e.g., Plaid/Open Banking), automated investing, or automated financial/tax advice. Anything resembling these is explicitly Out of Scope (§18) unless a future-scope section says otherwise.

---

## 10. User Journeys

**J1 — First-time open, no data yet**
A new user opens PlainCents with an empty database. The application explains that there is no data yet and offers two paths: import a real TD statement, or load synthetic demo data to see the product populated (see §10a Demo/Onboarding State). The user is never shown a broken or blank dashboard with no explanation.

**J2 — Importing a statement**
A user with a TD CSV export uploads it through the application. The system parses it, shows a preview/summary of what will be imported (row count, date range, any rows it could not parse), and the user confirms the import. Imported transactions appear with a predicted category already assigned.

**J3 — Reviewing and correcting categories**
A user browses their transaction list, notices "Amazon" was predicted as Shopping but is really a subscription, and corrects it. The change is saved immediately and reflected everywhere the category is used (dashboard, forecast inputs), while the original prediction is not lost from the record.

**J4 — Managing transactions directly**
A user finds a transaction with a duplicate or wrong amount (e.g., a bank export quirk) and edits its merchant, date, amount, or category directly. A user removes a transaction that should not have been imported (e.g., a duplicate). Changes persist across sessions.

**J5 — Checking the dashboard**
A user opens PlainCents to answer "how much have I spent this month, and where?" They see current-period spending, a comparison to the prior period, a category breakdown, a spending trend, and recent transactions, without leaving the app.

**J6 — Generating and reviewing a forecast**
A user with enough history requests a spending forecast. The system computes it, shows predicted spend per category for the next three months, and shows when it was generated. Reopening the forecast page later shows the same result instantly, without recomputation, until the user explicitly refreshes it.

**J7 — Cold-start forecast**
A new user with only 2 months of imported data opens the Forecast page. Instead of an error, they see a clear explanation that more history is needed, and how much more.

**J8 — Managing portfolio holdings**
A user adds a holding (ticker, shares, average cost). They see current market price, current value, and profit/loss, and can refresh prices, edit, or remove a holding.

---

## 10a. Demo/Onboarding State

V2 must include a demo/onboarding state for new users with no imported data. When a user opens PlainCents with no transactions, the application should offer to load synthetic demo data so the dashboard, forecasts, and portfolio are immediately visible and demonstrable. This is a product requirement, not a testing convenience — it is what a reviewer or interviewer will see on first open, and the application must not present an empty, unimpressive shell as the first experience.

This document specifies only the user-facing behavior:
- The empty state must clearly offer "load demo data" as an option, distinct from real import.
- Data loaded this way must be clearly and persistently labeled as synthetic/demo data everywhere it is shown, and must be as easy to clear/reset as it was to load. Clearing demo data is a **full demo reset** (see §9.2b).
- Demo data and real imported data are **mutually exclusive** for the MVP. If demo data is loaded and the user attempts to import real data, the application explains that demo data must be cleared first and asks for confirmation before the import proceeds. Demo data must never silently mix into real financial analytics (dashboard totals, forecasts, portfolio values). The storage mechanism used to enforce this separation is a TRD decision.

---

## 11. Functional Requirements

### 11.1 Application Navigation
- The application provides a persistent way to move between: Dashboard, Transactions, Import, Forecast, Portfolio.
- The current section is always clear to the user.

### 11.2 CSV Import
- A user can upload a bank CSV export through the application (not by placing a file on the filesystem).
- The system identifies or lets the user select which bank format the file is in.
- Before committing, the user sees a summary of what will be imported (e.g., number of rows, date range), any rows that could not be parsed, and any transactions identified as duplicates of already-imported data.
- On confirmation, valid, non-duplicate rows are persisted as transactions with a predicted category; detected duplicates are skipped, not inserted again.
- The import result reports counts for: rows imported, rows skipped as unparseable, and rows skipped as duplicates.
- Unparseable rows are reported to the user, not silently dropped without explanation.
- If demo data is currently loaded, attempting a real import first prompts the user to confirm clearing demo data, per §10a; the two do not mix.

### 11.3 Bank Handling
- TD import must work end-to-end, verified against a representative real export, before the V2 MVP is considered complete.
- The application is designed so that adding a new bank's format does not change how transactions behave downstream (categorization, dashboard, forecasting, portfolio are bank-agnostic once a transaction exists).
- The application must not claim to support a bank whose format has not actually been verified against a real, representative export — this applies to TD as much as to any later bank.

### 11.4 Transactions
- A user can view a list of their transactions, searchable and filterable (e.g., by date range, category, merchant) and sortable.
- A user can create a transaction manually; it goes through the same categorization step as an imported transaction and receives a predicted category, per §9.3.
- A user can edit an existing transaction's merchant, date, amount, and category.
- A user can delete a transaction.
- Edits persist and are visible after the application is refreshed or reopened.

### 11.5 Categorization
- Every imported or manually created transaction receives a predicted category.
- The predicted category is visible to the user, distinguishable from a user-corrected category.
- Predictions are produced without requiring the user to wait through a visible, per-transaction training step.

### 11.6 Category Corrections
- A user can change a transaction's category at any time.
- Correcting a category takes effect immediately for that transaction everywhere it is used (dashboard, forecast inputs).
- Correcting a category does not delete or hide the model's original prediction.
- Correcting a category does not trigger model retraining.

### 11.7 Dashboard
- The dashboard's default period is the **current calendar month compared to the previous calendar month**. Custom date ranges are post-MVP (§17).
- The dashboard shows, at minimum: current-period spending total, comparison to the prior period, a category breakdown, a spending trend over time, and a list of recent transactions.
- The dashboard also surfaces a forecast summary and a portfolio summary if either has data.
- If there is no transaction data, the dashboard shows the empty/onboarding state described in §10a rather than blank charts or errors.

### 11.8 Forecasting
- A user can request generation of a spending forecast covering the next three months, per category. Generation and refresh are **manual only** for the MVP — nothing about opening, loading, or reloading the Forecast page triggers training or recomputation.
- The forecast view shows when it was last generated.
- If transaction data or effective categories underlying a generated forecast change afterward, the existing forecast remains visible but is marked **stale**; the user may explicitly refresh it to recompute. Refreshing creates a new forecast run; prior forecast runs are retained, though the MVP UI primarily displays the latest run.
- If there is insufficient overall transaction history to forecast at all, the user sees an explanation of what is missing (e.g., "3 of 12 months available") rather than an error.
- If overall history is sufficient but a specific category's history is too sparse to produce a valid forecast for it, that category's forecast is shown as unavailable rather than a fabricated or zero value.

### 11.9 Portfolio
- A user can add a holding (ticker, shares, average cost).
- A user can edit or delete an existing holding.
- For each holding, the user sees current market price, current value, and profit/loss, using the latest available/cached values.
- Market-price refresh is **manual only** for the MVP: opening the Portfolio page never itself triggers an external price request; the user explicitly chooses "Refresh Prices."
- If there are no holdings, the portfolio view explains this clearly rather than showing an error or blank chart.

### 11.10 Analytics / Reports
- The dashboard is the primary analytics surface and must stand alone, fully usable, without PDF or PowerBI.
- PDF report generation and PowerBI export are **entirely post-MVP** (§17) — not part of the MVP application. The existing V1 capabilities are not deleted merely because they are deferred; they remain available as a separate, non-integrated artifact until revisited post-MVP.

### 11.11 Data Persistence
- All user data (transactions, corrections, holdings, generated forecasts, demo-data state) persists across application restarts.
- Demo/synthetic data loaded per §10a is clearly distinguishable from real imported data at all times.

### 11.12 Error / Empty / Cold-Start States
- No core screen (Dashboard, Transactions, Import, Forecast, Portfolio) may show an unhandled error or a blank/broken layout when its underlying data is empty.
- Every empty state explains what is missing and what the user can do next (import data, load demo data, add a holding, wait for more history, etc.).
- Import errors (unparseable rows, unrecognized bank format) are reported to the user in specific, actionable terms.

---

## 12. Data / Product Concepts

These are conceptual entities the product reasons about. They are not a schema and contain no SQL, types, or storage decisions — those belong to the TRD/DB schema design.

- **Transaction** — A single financial event a user has (date, merchant, amount, and a category). It is persistent and directly editable by the user, not merely an artifact of an import run.
- **Import Batch** — The record of one CSV import action (what file, which bank, when, how many rows succeeded/failed/skipped as duplicates). Used to give the user visibility into their own import history, not exposed as a required concept for using the rest of the product.
- **Predicted Category** — The category the categorization system assigned to a transaction at the time it was created or imported.
- **Confirmed Category** — The category the user has explicitly set for a transaction, if they have corrected or confirmed it.
- **Effective Category** — The category actually used everywhere in the product (dashboard, forecast inputs, exports): the confirmed category if one exists, otherwise the predicted category.
- **Forecast** — A set of predicted spending amounts per category for the next three months, generated at a specific point in time. A forecast can become **stale** if the underlying transaction data or effective categories change after it was generated, without being recomputed automatically.
- **Forecast Run** — The record of one forecast-generation action (when it happened, what data it was based on), letting the user (and the product) know how fresh the current forecast is and enabling a later comparison of forecast vs. actual. Refreshing a forecast creates a new run; prior runs are retained, though the MVP UI primarily displays the latest.
- **Holding** — A user-managed investment position (ticker, shares, average cost) with derived current value and profit/loss.

---

## 13. UX Requirements

- The application must be usable start-to-finish (import → review → correct → view dashboard → forecast → portfolio) without needing to read documentation.
- Destructive actions (deleting a transaction, deleting a holding, clearing demo data) require a confirmation step.
- Any action that could take a noticeable amount of time (import processing, forecast generation, price refresh) gives the user a clear loading/in-progress indication.
- Any category the user sees as "predicted" vs. "corrected" must be visually distinguishable, not just inferable from tooltips or hidden state.
- Synthetic/demo data must be visually labeled wherever it appears, not just documented in a README.

---

## 14. Validation & Error Behavior

- Manually entered or edited transactions must be validated for plausible values (e.g., a date, a non-empty merchant, a numeric amount) before being saved; the user receives a specific, actionable error otherwise.
- Import files that cannot be parsed at all (wrong file type, completely unrecognized format) produce a clear, specific error rather than a silent failure or a generic crash message.
- Rows within an otherwise-valid import that cannot be parsed are reported individually (or as a clear summarized count with reasons), and the rest of the valid import still proceeds.
- Rows detected as duplicates of already-imported transactions are not inserted again; the import result reports how many were skipped for this reason.
- Attempting a real import while demo data is loaded does not proceed silently — the user is asked to confirm clearing demo data first. Clearing demo data is a full demo reset (see §9.2b).
- Holdings require a valid ticker, a positive share count, and a non-negative average cost before being saved.
- If a market-price lookup for a holding fails (e.g., network issue, unknown ticker), the holding still displays with its last known price (if any) and a clear indication that the price could not be refreshed, rather than breaking the portfolio view.

---

## 15. Privacy Requirements

- Imported transaction data is not transmitted to any external service beyond what a specific feature genuinely requires (e.g., a ticker price lookup for portfolio holdings).
- The user should be able to tell, in the product, when a feature depends on an external service (e.g., that portfolio prices come from an external market-data source).
- No enterprise-grade compliance program (SOC2, encryption-at-rest certification, audit logging infrastructure, etc.) is claimed or required for this MVP; this document does not invent such requirements.
- No user financial data is sold, shared, or used for any purpose other than operating the product for that user.

---

## 16. MVP Scope

The first shippable V2 includes:

1. Application shell with navigation across Dashboard, Transactions, Import, Forecast, Portfolio.
2. TD CSV import, verified end-to-end against a representative real export, with import preview, per-row error reporting, and duplicate-skip reporting.
3. Persistent transactions (not batch/session-scoped).
4. Full transaction CRUD (create, view/search/filter/sort, edit, delete), with manual creation going through the same categorization step as import.
5. ML-assisted categorization applied automatically to imported/created transactions.
6. Manual category correction, preserving the original prediction and computing an effective category.
7. A basic but real spending dashboard defaulting to current calendar month vs. previous calendar month (category breakdown, trend, recent transactions).
8. Forecast viewing and **manual-only** forecast generation/refresh, with a visible "generated at" timestamp, no retraining on page view, visible staleness marking when underlying data changes after generation, and retention of prior forecast runs (MVP UI primarily shows the latest run).
9. Defined cold-start behavior when there isn't enough overall history to forecast, and defined "unavailable" messaging for individual categories with too-sparse history.
10. Portfolio CRUD (add/edit/delete holdings) with current value and P&L from latest cached prices, and **manual-only** price refresh.
11. The demo/onboarding empty state described in §10a, offering synthetic demo data on first open with no real data present, mutually exclusive with real imported data.
12. Import duplicate detection: previously-imported transactions are not re-inserted, and skip counts are reported to the user.

PDF report generation and PowerBI export are **not** part of the MVP (see §17).

## 17. Post-MVP / V2.x Scope

Deferred to after the first working demo:

- RBC and Scotiabank import (formats exist in V1 code but are unverified against real exports).
- BMO, CIBC, and National Bank import (no verified formats exist yet).
- Richer analytics (e.g., merchant-level drill-down, custom date ranges beyond current/prior calendar month).
- PDF report generation as an in-app, on-demand feature (V1 capability is not deleted; it is adapted/re-integrated later, not rebuilt from scratch).
- PowerBI CSV export as an in-app, on-demand feature (same non-deletion principle applies).
- Categorization model bake-off / potential replacement of K-Means with another approach.
- Deeper forecast evaluation (e.g., forecast-vs-actual tracking surfaced to the user over time).
- Automatic or scheduled portfolio price refresh (MVP is manual-only, per §9.7/§11.9).
- Automatic or event-triggered forecast recomputation beyond staleness marking (MVP is manual-only, per §9.6/§11.8).
- Additional automated testing and hardening beyond MVP-level coverage.
- Deployment/hosting enhancements beyond local single-user use.

## 18. Explicitly Out of Scope

The following are not part of PlainCents V2 (MVP or post-MVP) unless a future decision reverses this explicitly:

- Multi-user accounts, authentication, OAuth, or any account registration flow.
- LLM chatbot or generative-AI features.
- Vector databases or semantic search.
- Social features (sharing, comments, feeds).
- Payment processing of any kind.
- Message queues / microservices architecture.
- Real-time bank integrations (Plaid, Open Banking, or equivalents).
- Automated investing or trade execution.
- Automated financial advice or tax advice.
- Public hosting / multi-tenant deployment.

---

## 19. Acceptance Criteria

- A user can upload a TD CSV export and, after confirming the import, see the resulting transactions in the Transactions list with a predicted category on each.
- A user can edit the merchant, date, amount, and confirmed category of an existing transaction and see the persisted changes after refreshing the application.
- A user can delete a transaction and it no longer appears in the Transactions list or in dashboard totals after refreshing.
- When a user corrects a transaction's category, the dashboard's category breakdown reflects the corrected (effective) category, and the transaction record still retains what the model originally predicted.
- A user with sufficient historical data can generate a forecast and later reopen the Forecast page without retraining the model merely by viewing the page.
- A user with fewer than the required months of history sees a specific cold-start message on the Forecast page (not an error page or a crash) stating how much history is available and how much is required.
- A user can add a holding, see its current value and P&L from cached/last-known prices without any automatic external price request, edit its shares/average cost, and delete it, with each state persisting across a refresh.
- A user can explicitly click "Refresh Prices" on the Portfolio page and see updated values; simply opening or reloading the page does not trigger a price fetch.
- A brand-new user with no data sees an explicit offer to load synthetic demo data, and after doing so sees a populated dashboard, forecast, and portfolio, all clearly labeled as demo/synthetic data.
- If a user with demo data loaded attempts to import a real CSV, they are prompted to confirm clearing demo data first, and the import does not proceed until they confirm.
- If a user re-imports a CSV containing transactions already present, those transactions are not duplicated, and the import result reports how many were skipped.
- If a user corrects transaction categories or imports new data after a forecast was generated, the existing forecast remains visible but is marked stale, and only an explicit refresh recomputes it. Refreshing creates a new forecast run; prior runs are retained.
- Clearing demo data removes all demo state in a full demo reset.
- No core screen throws an unhandled error or renders blank when its underlying data is empty.

---

## 20. Success Criteria

- The MVP can be demonstrated start-to-finish (fresh instance → demo data or real TD import → categorize/correct → dashboard → forecast → portfolio) in a single sitting without hitting an unhandled error.
- A reviewer opening the application for the first time, with no setup beyond running it, sees a populated, coherent product within the demo/onboarding flow — not an empty shell.
- Transaction, category-correction, forecast, and portfolio behavior all match the acceptance criteria in §19 when exercised manually.
- No claim in the product UI (metrics, accuracy figures, forecast confidence) mischaracterizes synthetic data as real-world performance.

---

## 21. Known Constraints

- The current categorization model artifact (`kmeans_model.pkl`) must exist before predictions can be produced; the product must handle its absence gracefully rather than crashing (exact mechanism is a TRD concern, but the product-level requirement is: no unhandled failure).
- Forecasting requires a minimum amount of historical data (V1 enforces at least 12 unique months); for the MVP this remains the eligibility threshold unless later technical validation demonstrates it must change, and V2 must treat falling short of it as a defined cold-start state, not an exception.
- Only TD is required to have a real, end-to-end-verified import path at MVP time; RBC/Scotiabank formats exist in V1 code but are unverified, and BMO/CIBC/National Bank formats do not exist yet. No bank, including TD, should be described as verified unless repository evidence actually demonstrates it.
- This is a single-user, local-first product for the MVP; no requirement in this document assumes multi-user isolation.
- Portfolio price data depends on an external market-data source; its availability is not guaranteed and the product must degrade gracefully (§14). Price refresh is manual-only for the MVP (§9.7/§11.9).
- Forecast generation is manual-only for the MVP (§9.6/§11.8); staleness detection logic is a TRD/ML-specification concern, not decided here.
- Import duplicate detection is required for the MVP (§9.2a/§11.2); the specific matching/hashing mechanism is a TRD concern.

---

## 22. Resolved Product Decisions

The following product decisions are **frozen** as of this document. Implementation details belong in the V2 TRD / ML specification.

1. **Demo data clearing** — Clearing demo data is a **full demo reset**. All demo transactions, demo forecasts, demo portfolio state, and related demo artifacts are removed together. There is no partial demo retention in the MVP.

2. **Forecast refresh** — Refreshing a forecast **creates a new Forecast Run**. Previous forecast runs are **retained** (supporting later forecast-vs-actual comparison per §17). The MVP UI primarily displays the latest run.

3. **Category taxonomy** — The eight-category taxonomy (§9.4) is **fixed for the V2 MVP**. Custom or user-defined categories are out of scope for the MVP but may be considered in a future version.

**Note on technical questions:** Items such as exact staleness-detection logic, deduplication matching strategy, database representation of predicted/confirmed/effective category, forecast-run storage schema, and the mechanism enforcing demo/real data separation are **not** product questions — they belong to the V2 TRD / ML specification.

---

*End of document. No V2 code was written or modified in the production of this document.*
