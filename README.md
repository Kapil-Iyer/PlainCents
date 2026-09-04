# PlainCents

**A local-first personal finance MVP.** Import a bank CSV (or load sample
demo data), get ML-assisted transaction categorization, a spending
dashboard, on-demand category-level forecasts, and simple portfolio
tracking — all running as a single app on your own machine.

**PlainCents V2 is the current application.** V1 (the original batch ML
pipeline) still lives in this repo for historical/regression reference —
see [V1 — historical predecessor](#v1--historical-predecessor) below.

---

## V2 architecture

```
React + TypeScript (Vite)  →  FastAPI  →  service / repository layers  →  SQLite
                                             ↑
                              ML: TF-IDF + Logistic Regression categorizer,
                              Naive spend forecaster — selected via an
                              evidence-based evaluation pipeline (see
                              reports/ml/), integrated behind V2's service
                              boundaries
```

- **Frontend:** `frontend/` — React 19 + TypeScript + Vite, Tailwind, a
  small shadcn/ui-style component set, Recharts, TanStack Query, React
  Router.
- **Backend:** `backend/` — FastAPI, with `api/routes` → `services` →
  `repositories` → SQLite (`plaincents_v2.db`, separate from V1's
  `plaincents.db`).
- **ML:** categorization is TF-IDF + Logistic Regression
  (`ml/categorization/candidates.py::TfidfLogRegCandidate`, merchant text
  only), forecasting is a Naive (lag-1) baseline
  (`ml/forecasting/baselines.py::naive_predict`) — both selected over
  several alternatives (K-Means/Linear SVM; Random Forest/Ridge/Seasonal
  Naive) through a merchant-grouped/temporal, leakage-safe evaluation
  documented in `reports/ml/`. `backend/services/categorization_service.py`
  and `pipeline/forecast.py::train_and_predict` are the production
  integration points; V1's original K-Means/Random Forest implementations
  remain in the repo untouched as historical/evaluation baselines, no longer
  what the running app uses.

Product surfaces: **Dashboard**, **Transactions**, **Import**, **Forecast**,
**Portfolio** — plus a demo/onboarding flow for a brand-new, empty install.

---

## MVP / product status

- **M5 — App-Demonstrable MVP: complete.** All five product surfaces work
  end-to-end, packaged reviewer mode works, and the acceptance criteria in
  the frozen V2 PRD (§19) pass under manual verification.
- **ML scientific evaluation and production integration: complete.** A
  leakage-safe evaluation (merchant-grouped splits for categorization,
  temporal expanding-window validation for forecasting) benchmarked
  multiple candidates per problem and selected TF-IDF + Logistic Regression
  (categorization) and Naive (forecasting), now integrated as the running
  app's production implementations. Full methodology, evidence, and exact
  numbers — with their evidence-tier caveats — are in `reports/ml/`,
  particularly `ML_E_FINAL_ML_REPORT.md` and `ML_E_CLAIM_MATRIX.json`.
  **In one sentence, with the caveat attached:** the selected categorizer
  scored 42.2% accuracy / 0.4405 macro F1 on a held-out slice of an
  independently curated (not real-world) benchmark; the selected forecaster
  scored 18.9% WAPE on a reserved period of a synthetic (not real-world)
  dataset — neither number should be read as real-world accuracy.
- **CSV import supports four Canadian bank export formats.** PlainCents
  currently supports transaction CSV imports for RBC, Scotiabank, TD, and
  CIBC. RBC and Scotiabank formats were validated against actual exports;
  TD support is project-verified, with headerless-format limitations
  disclosed; CIBC support is research-backed and fail-closed. BMO and
  National Bank support are coming soon. Do not read this as universal Big
  Six coverage or as a claim that every product/account variant from a
  supported bank is handled.

---

## Prerequisites

- Python 3.11+
- Node.js 20+ and npm
- (Windows/macOS/Linux — no Docker, no cloud account, no external database
  required; everything runs locally against SQLite.)

## Initial installation

```bash
# from the repo root
pip install -r requirements.txt
cp .env.example .env      # optional — defaults work out of the box
cd frontend && npm install && cd ..
```

If you already have a trained categorizer at `models/tfidf_logreg_v1.pkl`,
nothing else is needed. If not, run `python -m scripts.build_production_logreg_model`
once (see [Model artifact](#model-artifact) below) — or just use **Explore
demo**, which doesn't need it at all.

## Development workflow

Two servers, with hot reload:

```bash
# terminal 1 — backend (http://localhost:8000)
uvicorn backend.main:app --reload

# terminal 2 — frontend (http://localhost:5173), proxies /api to :8000
cd frontend && npm run dev
```

Open `http://localhost:5173`. Frontend tests: `cd frontend && npm test`.
Backend tests: from the repo root, first build the deterministic test
categorizer fixture (once per fresh clone), then run pytest:

```bash
python tests/fixtures/build_test_logreg_model.py
pytest
```

(`tests/fixtures/logreg_model_test.pkl` is gitignored — see
`tests/fixtures/README.md`. Production artifact:
`python -m scripts.build_production_logreg_model`.)

## Reviewer / demo launch (one command)

After the installation step above, this is the **one normal command** to
run PlainCents as a reviewer would — one process, one port, no dev tooling
required in a second terminal:

```bash
python -m backend.scripts.run_reviewer
```

This builds the frontend production bundle if it's missing or stale, then
starts the FastAPI app, which serves the built frontend directly alongside
the API — open **http://127.0.0.1:8000** and use the app normally,
including deep links like `/dashboard` or `/portfolio`. `/api/*` continues
to serve the API from the same process. Stop it with Ctrl+C.

(Equivalently, if you've already run `npm run build` in `frontend/`
yourself, `uvicorn backend.main:app` alone will serve the same packaged
mode — `run_reviewer.py` is just a convenience wrapper that makes sure the
build exists first.)

## Explore the demo

From an empty install (packaged mode or dev mode), the app's first screen
offers **Explore demo** / **Load demo data**: a deterministic, clearly
synthetic dataset (12 months of transactions, a prebuilt forecast, sample
portfolio holdings) that populates every screen so you can see the product
work without importing anything. Everything loaded this way is labeled
**Demo** throughout the UI, is mutually exclusive with real imported data,
and can be cleared at any time. A first-open recruiter/product walkthrough
is also shown alongside this screen — a presentation-only preview of the
five product areas that never touches app state (distinct from actually
loading the interactive demo).

---

## Model artifact

The categorizer needs a trained artifact at `models/tfidf_logreg_v1.pkl`
(gitignored — never committed) — TF-IDF + Logistic Regression, the model
selected by the evaluation in `reports/ml/` (see
[ML scientific evaluation](#mvp--product-status) above). To build it from
the frozen benchmark evidence committed in this repo:

```bash
python -m scripts.build_production_logreg_model
```

This fits on the frozen TRAIN partition only (never on the held-out
VALIDATION/FINAL_TEST rows scored during evaluation) and refuses to run if
the ML-C selection record doesn't name this model as selected. See
`reports/ml/ML_E_REPRODUCIBILITY.md` for the full reproducibility workflow.

If this file is missing, the app still runs: `/api/health` reports the
categorizer as unavailable, real CSV import is blocked with a clear message
until a model is present, and **Explore demo** is unaffected (its data is
pre-labeled, not run through the categorizer).

Forecasting has no equivalent artifact — the selected Naive model is
stateless code, recomputed fresh on every forecast run
(`pipeline/forecast.py::train_and_predict`), nothing to build ahead of time.

---

## Data privacy

- No real bank data is committed to this repo. `data/raw/`, `plaincents.db`,
  and `plaincents_v2.db` are all gitignored.
- The app makes exactly one kind of outbound network call: fetching a
  current price from Yahoo Finance (`yfinance`), and only when you
  explicitly click **Refresh Prices** on the Portfolio page. Simply opening
  or reloading the Portfolio page never makes a network request — prices
  shown are the last cached values.
- Everything else — import, categorization, dashboard, forecast — runs
  entirely against your local SQLite database.

---

## Testing

| Suite | Command | Count |
|---|---|---|
| Backend (pytest) | `pytest` | 270 tests |
| Frontend (Vitest) | `cd frontend && npm test` | 39 tests |
| E2E (Playwright) | `npm run e2e:install` once, then `npm run e2e` | 4 flows |

E2E setup, from the repo root (separate from `frontend/`'s own `npm install`):

```bash
npm install          # repo root — installs @playwright/test
npm run e2e:install  # once — downloads the Chromium browser
npm run e2e          # runs the 4 flows
```

The 4 E2E flows (`tests/e2e/*.spec.ts`) are deliberately not exhaustive —
they cover the demo lifecycle, a real CSV import, forecast staleness after
a data correction, and the portfolio refresh boundary. Each spins up its
own isolated backend against a temp SQLite database and never touches your
real `plaincents_v2.db`. The Portfolio flow uses a deterministic offline
stand-in for `yfinance` (`tests/e2e/fixtures/fake_yfinance/`, wired in only
via `PYTHONPATH` for that test process) so it never depends on live market
data being reachable — see that package's docstring for exactly how and why.

---

## Repository layout

```
PlainCents/
├── backend/            # V2 FastAPI app: api/, services/, repositories/, db/, scripts/
├── frontend/            # V2 React app
├── pipeline/            # ingest, features, cluster (V1 K-Means, retired), forecast (train_and_predict = selected Naive; V1 RF retained), portfolio
├── ml/                   # ML-B/C evaluation: candidates, splitting, metrics, bake-offs, FINAL runners
├── db/                  # V1 schema/seed (untouched) + db/migrations/ (V2 SQLite migrations)
├── data/evaluation/      # Frozen Tier B categorization benchmark + split (committed evidence)
├── reports/ml/           # ML-B/C/E reports, selection record, claim matrix, results (committed evidence)
├── docs/                 # Frozen V2 PRD / TRD / Build Plan / ML Spec
├── tests/
│   ├── backend/          # V2 backend unit + API tests
│   ├── ml/               # ML evaluation-infrastructure tests
│   ├── e2e/              # Playwright E2E (4 flows)
│   └── fixtures/         # td_csv/ fixtures, deterministic test ML artifacts
├── models/               # tfidf_logreg_v1.pkl (selected, production), kmeans_model.pkl, rf_model.pkl (retired) — all gitignored
├── scripts/              # build_production_logreg_model.py + V1 synthetic-data generators
├── main.py, viz/, data/  # V1 — see below
└── playwright.config.ts, package.json   # E2E tooling only (no app code here)
```

---

## V1 — historical predecessor

PlainCents V1 was a Python batch pipeline: run a script, it ingests a CSV,
categorizes and forecasts, writes to SQLite, and produces a PDF report /
PowerBI export. It has no web UI. V2 superseded it with an interactive
app, but V1's files remain in the repo, untouched, for historical reference
and regression checking — its ML implementations (`pipeline/cluster.py`,
`pipeline/forecast.py`, `pipeline/portfolio.py`) are also what V2 reuses.

V1 entrypoints (all still runnable independently of V2, against V1's own
`plaincents.db`):

```bash
python main.py                      # full pipeline orchestration
python -m pipeline.cluster          # train/evaluate the K-Means categorizer
python -m pipeline.forecast         # train/evaluate the Random Forest forecaster
python db/seed_synthetic_data.py    # seed V1's SQLite tables with synthetic data
python viz/report.py                # generate a Matplotlib PDF report
python viz/powerbi_export.py        # export CSVs for the PowerBI dashboard
```

`python -m pipeline.cluster` and `python -m pipeline.forecast` require
`data/raw/synthetic_24mo.csv` (or regenerate it via `scripts/generate_synthetic_24mo.py`).
`viz/report.py` and `viz/powerbi_export.py` require an existing V1
`plaincents.db` (produced by `python main.py`).

### V1 model performance (synthetic data only — not real-world evidence)

- K-Means categorization: 90% accuracy on a 40-transaction held-out set
  (ARI = 0.81 on 779 transactions).
- Forecast model MAPE: 15.7% against ground-truth labels; 29.4% end-to-end
  pipeline MAPE (the gap is quantified clustering-label noise, not model
  error).
- All figures above come from a synthetic 24-month dataset generated for
  development purposes — they characterize this codebase's behavior on
  that synthetic data, not real-world forecasting accuracy, and should
  never be read as either.

---

## Author

**Kapil Iyer**
Bachelor of Honours Mathematics, University of Waterloo
Applied Mathematics (Scientific ML) and Statistics, Computing Minor

[GitHub](https://github.com/Kapil-Iyer) · [LinkedIn](https://github.com/Kapil-Iyer) · [Portfolio](https://kapil-iyer-portfolio.vercel.app/)
