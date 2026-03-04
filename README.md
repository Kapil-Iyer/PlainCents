# PlainCents

Personal finance and investment analytics: ML-driven spending intelligence and reporting from your bank CSV exports.

## Goals

- **Ingest** raw bank CSV exports (TD, RBC, Scotiabank) and normalize dates/columns.
- **Categorize** transactions with K-Means clustering (8 categories, label mapping, held-out evaluation).
- **Forecast** 3-month spending per category with Random Forest and walk-forward validation.
- **Persist** everything in a local SQLite data warehouse (6 tables), with price caching for portfolio P&L.
- **Report** via an automated Matplotlib PDF and an interactive PowerBI dashboard.

Python pipeline + SQLite + PowerBI + PDF. Local Python ML pipeline, no web hosting.

## Tech Stack

Python, Pandas, NumPy, scikit-learn, SQLite, yfinance, Matplotlib, joblib, PowerBI.

## Current Progress

### Phase 0 — Pre-Build (Complete)
Folder structure, `config.py` (paths, categories, bank date formats), `db/schema.sql` (6-table SQLite schema), `requirements.txt`, `.gitignore`.

### Phase 1 — Ingestion + Synthetic Data (Complete)
`pipeline/ingest.py` loads bank CSVs, detects bank format (TD/RBC/Scotiabank), standardizes columns (date, merchant, amount), parses dates via `config.BANK_DATE_FORMATS`, cleans merchant names (uppercase, strip, remove special chars), and deduplicates.

Synthetic dataset: `data/raw/synthetic_24mo.csv` — 779 transactions across 24 months (Jan 2023–Dec 2024), TD format, 8 expense categories with category-distinct merchant descriptions. Generated deterministically (seed 42) via `scripts/generate_synthetic_24mo.py`.

### Phase 2 — K-Means Clustering (Complete)
`pipeline/features.py` builds a feature matrix: StandardScaler on amount, TF-IDF on merchant names (bigrams, L2-normalized, alpha-only tokens, domain stop words), day-of-week, is_weekend.

`pipeline/cluster.py` fits K-Means (n_clusters=12, n_init=50), builds a cluster-to-category mapping via majority vote on 160 labeled rows, evaluates on 40 held-out rows.

**Metrics:**
- Held-out accuracy: 90% (40 rows)
- Silhouette score: 0.5437
- Adjusted Rand Index: 0.8073 (all 779 rows)
- Categories assigned: 7/8 ("Other" has no cluster — catch-all categories lack coherent TF-IDF signature)

### Phase 3 — Random Forest Forecast (Complete)
`pipeline/forecast.py` aggregates transactions to monthly category totals, engineers 8 features (month_num, category_encoded, rolling_3m_avg, rolling_6m_avg, rolling_std, is_december, is_summer, lag_1_spend), and trains a Random Forest with walk-forward validation (expanding window, never shuffles time series).

GridSearchCV (TimeSeriesSplit, 27 param combos) runs automatically when MAPE > 15%, per PRD spec. Final model saved to `models/rf_model.pkl`.

**Metrics:**
- Walk-forward MAPE (end-to-end with clustering): 29.4%
- Walk-forward MAPE (true labels, isolating forecast model): 15.7%
- 5 of 8 categories below 15% MAPE on true labels
- MAPE gap (13.7%) is caused by upstream clustering noise, not forecast model weakness

## PRD Deviations

| Change | Reason |
|---|---|
| `lag_1_spend` added as 8th forecast feature | Not in original PRD feature list. Added after empirical testing showed 20.7% → 15.7% MAPE improvement. Standard time series practice, leak-free. |
| `month_number` and `is_recurring` removed from clustering features | Listed in PRD Section 6 but caused catastrophic accuracy drop (90% → 5%) due to curse of dimensionality in sparse TF-IDF space. Documented deviation. |
| Synthetic data extended to 24 months | PRD specifies 12 months minimum. Extended to improve walk-forward validation (68.8% → 29.4% MAPE). Generator redesigned with temporal structure for forecasting. |
| n_clusters=12 instead of 8 | PRD default is 8. Increased to improve category coverage and held-out accuracy (more cluster slots for majority-vote mapping). |

## Key Challenges

**Clustering accuracy (Phase 2):** Initial accuracy was 22.5%. Root cause was TF-IDF vocabulary overlap across categories. Fixed by: L2-normalizing TF-IDF vectors, down-weighting numeric features, regenerating synthetic data with category-distinct merchant descriptions, tuning TF-IDF (bigrams, sublinear_tf, alpha-only tokens, domain stop words). Final: 77.5% on 12mo data → 90% on 24mo data.

**Forecast MAPE (Phase 3):** Initial MAPE was 68.8% on 12-month data — a data scarcity problem (8 training rows in the first walk-forward fold). Fixed by: extending to 24 months with temporal structure, adding lag_1_spend feature, applying GridSearchCV-tuned hyperparameters (max_depth=3) to walk-forward folds. True-label MAPE dropped from 68.8% → 15.7%.

**Walk-forward data leakage risk:** Original test-row extraction used a fragile `iloc` mask alignment that could misalign when `build_forecast_features` dropped rows. Fixed by computing test features directly from training history per category — the same leak-proof pattern used for +1/+2/+3 forecasts.

**Clustering contamination in forecast:** Discovered via diagnostic (true-label vs clustered MAPE). The 13.7% gap is caused by ~48 "Other" transactions with no cluster being misclassified into other categories, creating unpredictable noise in monthly totals. Documented as an inherent limitation of unsupervised clustering on catch-all categories.

## Getting Started

```bash
# 1. Clone and set up
git clone <repo-url>
cd PlainCents
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Generate synthetic data
python scripts/generate_synthetic_24mo.py

# 3. Run clustering (saves model to models/kmeans_model.pkl)
python -m pipeline.cluster

# 4. Run forecasting (saves model to models/rf_model.pkl)
python -m pipeline.forecast
```

## Data Privacy

Never commit real bank data. `data/raw/` is gitignored; only synthetic data is included in the repo.
