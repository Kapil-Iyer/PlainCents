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
                              ML/analytics: K-Means categorizer, Random
                              Forest forecaster (reused from V1's pipeline/,
                              wrapped behind V2's service boundaries)
```

- **Frontend:** `frontend/` — React 19 + TypeScript + Vite, Tailwind, a
  small shadcn/ui-style component set, Recharts, TanStack Query, React
  Router.
- **Backend:** `backend/` — FastAPI, with `api/routes` → `services` →
  `repositories` → SQLite (`plaincents_v2.db`, separate from V1's
  `plaincents.db`).
- **ML:** categorization (K-Means) and forecasting (Random Forest) reuse
  V1's `pipeline/` implementations, called from `backend/services/` — no ML
  logic is duplicated between V1 and V2.

Product surfaces: **Dashboard**, **Transactions**, **Import**, **Forecast**,
**Portfolio** — plus a demo/onboarding flow for a brand-new, empty install.

---

## MVP / product status

- **M5 — App-Demonstrable MVP: complete.** All five product surfaces work
  end-to-end, packaged reviewer mode works, and the acceptance criteria in
  the frozen V2 PRD (§19) pass under manual verification.
- **ML scientific evaluation is a separate, ongoing track.** The
  categorizer and forecaster currently running in the app are V1's
  original K-Means/Random Forest implementations, used as-is behind V2's
  service boundaries so app development wasn't blocked on model research.
  A rigorous scientific evaluation/model-selection pass (baselines, error
  analysis, candidate models, acceptance gates) is planned as a separate
  milestone and is **not** claimed complete by M5/Phase 10 being done.
- **TD CSV import is fixture-tested, not field-verified.** Import has been
  tested against synthetic fixtures shaped like TD's CSV export format
  (`tests/fixtures/td_csv/`), including a plausible headerless/positional
  variant. It has **not** been verified against an actual TD account
  export — do not read any claim in this repo as confirming that.

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

If you already have a trained categorizer at `models/kmeans_model.pkl`,
nothing else is needed. If not, run `python -m pipeline.cluster` once (see
[Model artifact](#model-artifact) below) — or just use **Explore demo**,
which doesn't need it at all.

## Development workflow

Two servers, with hot reload:

```bash
# terminal 1 — backend (http://localhost:8000)
uvicorn backend.main:app --reload

# terminal 2 — frontend (http://localhost:5173), proxies /api to :8000
cd frontend && npm run dev
```

Open `http://localhost:5173`. Frontend tests: `cd frontend && npm test`.
Backend tests: `pytest` from the repo root.

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

The categorizer needs a trained artifact at `models/kmeans_model.pkl`
(gitignored — never committed). To produce one from the included synthetic
training data:

```bash
python -m pipeline.cluster
```

If this file is missing, the app still runs: `/api/health` reports the
categorizer as unavailable, real CSV import is blocked with a clear message
until a model is present, and **Explore demo** is unaffected (its data is
pre-labeled, not run through the categorizer).

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
| Backend (pytest) | `pytest` | 210 tests |
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
├── pipeline/            # ML: ingest, features, cluster (K-Means), forecast (RF), portfolio
├── db/                  # V1 schema/seed (untouched) + db/migrations/ (V2 SQLite migrations)
├── docs/                 # Frozen V2 PRD / TRD / Build Plan / ML Spec
├── tests/
│   ├── backend/          # V2 backend unit + API tests
│   ├── e2e/              # Playwright E2E (4 flows)
│   └── fixtures/         # td_csv/ fixtures, deterministic test ML artifact
├── models/               # kmeans_model.pkl, rf_model.pkl (gitignored)
├── main.py, viz/, scripts/, data/   # V1 — see below
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
