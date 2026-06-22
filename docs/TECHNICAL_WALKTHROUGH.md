# PlainCents — Technical Walkthrough (Interview Prep)

A code-anchored walkthrough of the ML pipeline: what each part does, how it connects, and how to explain it in an interview.

---

## 1. Project Overview

**What PlainCents does**  
PlainCents is a personal finance analytics pipeline. It takes raw bank CSV exports (no category column), assigns each transaction to one of eight expense categories using unsupervised clustering, then forecasts spending for the next three months per category. Results and portfolio snapshots are stored in SQLite for reporting (e.g. PowerBI).

**Problem it solves**  
Bank exports usually have date, description, and amount only. The pipeline (1) **categorizes** transactions without hand-labeled training data (K-Means on features from text + amount + date), and (2) **forecasts** monthly spend per category so you can plan and compare predicted vs actual later.

**Full ML pipeline (data flow)**  
```
Bank CSV (date, description, amount)
  → ingest.py          (load, standardize columns, parse dates, clean merchant, dedupe)
  → features.py        (feature matrix: amount_scaled, TF-IDF on merchant, day_of_week, is_weekend)
  → cluster.py         (K-Means fit/predict, cluster→category mapping, add category column)
  → forecast.py        (aggregate to monthly category totals → RF features → walk-forward validation → 3‑month forecasts)
  → database.py        (persist transactions, predictions, monthly_summary, portfolio, etc.)
  → main.py            (orchestrates all steps; optional: report/PowerBI later)
```

So: **transactions CSV → preprocessing → feature engineering → K-Means clustering → category labels → Random Forest forecasting → results stored in SQLite.**

---

## 2. Folder and File Structure

| Path | Role |
|------|------|
| **main.py** | Entry point. Opens DB, runs ingest → cluster → forecast → portfolio, then writes to all relevant tables. Uses a single `SESSION_ID` per run. |
| **config.py** | Paths (`DATA_RAW`, `DB_PATH`, `KMEANS_MODEL_PATH`, `RF_MODEL_PATH`), `CATEGORIES` (8 names), `BANK_DATE_FORMATS`, chart color placeholders. No logic. |
| **pipeline/ingest.py** | Loads CSV from `data/raw/`, maps bank-specific column names to `date`/`merchant`/`amount`, parses dates via config, cleans merchant (uppercase, strip, normalize chars), dedupes. Returns a DataFrame; no DB or file writes. |
| **pipeline/features.py** | Builds the **clustering** feature matrix only: scaled amount, L2-normalized TF-IDF on merchant (max 50 terms, bigrams), day_of_week, is_weekend. Used by cluster.py. |
| **pipeline/cluster.py** | Trains K-Means (or loads saved model), maps cluster IDs to category names via majority vote on 160 labeled rows, evaluates on 40 held-out. **Inference:** `predict_categories(df)` loads the saved pkl, transforms with same scaler/vectorizer, predicts cluster, maps to category. |
| **pipeline/forecast.py** | Aggregates transactions to monthly category totals, builds RF features (rolling, lag, month flags), runs **walk-forward validation**, fits final RF, produces +1/+2/+3 month forecasts. **Diagnostic:** `__main__` runs the full forecast again with ground-truth labels and prints both MAPEs. |
| **pipeline/portfolio.py** | Fetches prices (cache-first, 1h TTL via `db`), computes P&L, inserts portfolio rows. Used by main after forecast. |
| **db/database.py** | SQLite connection (and schema execution), insert/upsert/query helpers for all 6 tables. No ML or pipeline logic. |
| **db/schema.sql** | Defines the 6 tables. Used by `get_connection()` on every run. |
| **db/seed_synthetic_data.py** | Standalone script: clears tables and inserts synthetic data for all 6 tables (demo only). Not called by main.py. |

**Connections:**  
- **main.py** calls ingest → cluster → forecast → portfolio in order, then uses `db.database` for all writes.  
- **cluster.py** uses **features.py** to get `X`; **forecast.py** only needs the DataFrame with a `category` column (from cluster or from the diagnostic’s true labels).  
- **forecast.py** does not call cluster; it only aggregates by `category` and runs RF. So the “pipeline” and “diagnostic” runs are just two different ways of setting `df["category"]` before calling `fit_and_forecast`.

---

## 3. K-Means Clustering Pipeline

**How transactions are processed**  
- **Input:** DataFrame from `load_and_clean()` with columns `date`, `merchant`, `amount` (strings/datetime and numeric).  
- **Feature build:** `build_feature_matrix()` in `features.py` is called with `fit=True` (training) or `fit=False` (inference with saved scaler/vectorizer).

**Features created**  
- **Numeric:**  
  - `amount`: standardized with `StandardScaler`, then multiplied by `0.2`.  
  - `day_of_week` (0–6), `is_weekend` (0/1), each multiplied by `0.1`.  
- **Text:**  
  - `merchant` → TF-IDF with `TfidfVectorizer(max_features=50, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b", ngram_range=(1,2), sublinear_tf=True)`.  
  - The TF-IDF matrix is converted to dense and **L2-normalized** (`normalize(..., norm='l2')`).  
- **Final X:** `np.hstack([amount_weighted, merchant_dense, dow_weighted, weekend_weighted])` — one row per transaction.

**How TF-IDF is used**  
TF-IDF turns each merchant string into a 50-dimensional vector of term weights (unigrams + bigrams, alpha tokens only). L2 normalization makes K-Means behave more like cosine similarity (direction matters more than length). The pipeline uses this so that similar merchant names end up in the same cluster.

**How K-Means is trained**  
- In `cluster.py`, `fit_and_evaluate()`:  
  - Builds `X` via `build_feature_matrix(df, fit=True)`.  
  - Fits `KMeans(n_clusters=12, random_state=42, n_init=50)` and gets `cluster_ids`.  
  - True labels come from `_get_true_labels(df)` (keyword match on merchant using `MERCHANT_KEYWORDS`).  
  - 200 rows with labels are chosen; 160 used for mapping, 40 for held-out accuracy.  
  - For each cluster ID, majority vote over those 160 rows gives one category → `cluster_to_category`.  
  - Held-out accuracy = fraction of the 40 rows where `cluster_to_category[cluster_id]` equals the true label.  
  - Saves `kmeans`, `scaler`, `vectorizer`, `cluster_to_category` to `models/kmeans_model.pkl`.

**How clusters are mapped to category labels**  
- After fitting, each cluster ID is assigned the **most frequent true label** among the 160 mapping rows that fell in that cluster.  
- At inference, `predict_categories()` loads the pkl, runs `build_feature_matrix(..., fit=False)`, `kmeans.predict(X)`, then maps each cluster ID to a category name via `cluster_to_category`.

**How clustering accuracy was evaluated**  
- **Held-out accuracy:** 40 rows not used for mapping; compare predicted category (from cluster + mapping) to true label (from `MERCHANT_KEYWORDS`). Reported as a single percentage (e.g. 90%).  
- **Silhouette score:** `silhouette_score(X, cluster_ids)` on the full training X (diagnostic only).  
- **Adjusted Rand Index (ARI):** In `cluster.py` `__main__`, `adjusted_rand_score(true_labels, df_labeled["category"])` over **all** rows (e.g. 779), so clustering agreement with ground truth is measured on the full set, not just 40 rows.

---

## 4. Random Forest Forecasting

**What the forecasting model predicts**  
For each of the 8 categories, it predicts **total spend (dollars)** for each of the next three calendar months (month +1, +2, +3). So 24 predictions per run (8 categories × 3 months).

**Features used**  
All are derived from **monthly** series (one row per category per month) in `build_forecast_features()`:  
- `month_num` (1–12)  
- `category_encoded` (LabelEncoder on `CATEGORIES`)  
- `rolling_3m_avg`, `rolling_6m_avg` (mean of prior months’ spend)  
- `rolling_std` (std of last 3 months’ spend, ddof=1)  
- `is_december`, `is_summer` (month in {6,7,8})  
- `lag_1_spend` (previous month’s spend for that category)  

Rolling and lag use **only past months**; no future data. Rows with insufficient history (e.g. missing `lag_1_spend` or rolling) are dropped before training.

**How training data is constructed**  
- `aggregate_monthly(df)` groups by `(month, category)` and sums `amount` → one row per (month, category) with `total_spend`.  
- `build_forecast_features(monthly_df)` adds the rolling/lag/calendar features per category; target is `total_spend`.  
- So the model is trained on (X = features for that month/category, y = that month’s total spend).

**Validation method: walk-forward, expanding window**  
- In `walk_forward_validate()`:  
  - Months are sorted. For each candidate test month `M`, training set = all months **strictly before** `M`.  
  - Train: `train_df` = monthly rows with month < M; features/target from `build_forecast_features(train_df)`.  
  - Test: for month M, for each category present in that month, build **one test row** using only **training history** (last 3/6 months and lag from train), then predict.  
  - So test features are computed from past data only — no lookahead.  
  - APE per (month, category) = |actual − predicted| / max(|actual|, 1e-9) × 100.  
  - MAPE = mean of those APEs.  
- So it’s **expanding window**: train on 1..M−1, test on M; then train on 1..M, test on M+1; etc. No shuffling; no future data in features.

**Training loop in code terms**  
- `fit_and_forecast(df)` (1) aggregates to monthly, (2) calls `walk_forward_validate(monthly_df)` to get MAPE and validation predictions, (3) builds features on **all** monthly data with `build_forecast_features(monthly_df)`, (4) optionally runs GridSearchCV if MAPE > 15%, (5) fits final `RandomForestRegressor` on full X, y, (6) generates +1/+2/+3 month forecasts by building one row per (future_month, category) using last-known rolling/lag from the series, (7) saves model + encoder to `rf_model.pkl` and returns forecast DataFrame plus MAPE(s).

---

## 5. Diagnostic Experiment (Error Decomposition)

**Purpose**  
To separate “forecast model error” from “error due to wrong category labels.” If we run the **same** forecast pipeline with (a) K-Means labels and (b) ground-truth labels, the difference in MAPE tells us how much is clustering noise.

**Implementation: the forecast is run twice with two different label sets**

There is **no** single function that takes two label sets and returns both MAPEs. The diagnostic is implemented in the **`__main__`** block of `pipeline/forecast.py` (lines 364–386).

**Run 1 — Predicted (K-Means) labels**  
- `df = load_and_clean("synthetic_24mo.csv", bank="TD")`  
- `df = predict_categories(df)` → `df["category"]` is set by the saved K-Means model + cluster→category mapping.  
- `forecast_df, mape, cat_mape = fit_and_forecast(df)`  
- This uses **K-Means-assigned categories** for aggregation and walk-forward. The printed “Overall MAPE” is the **pipeline MAPE** (e.g. 29.4%).

**Run 2 — Ground-truth labels**  
- `df2 = load_and_clean("synthetic_24mo.csv", bank="TD")` — same CSV, new DataFrame.  
- `df2["category"] = _get_true_labels(df2)` — categories from `MERCHANT_KEYWORDS` (substring match on merchant), **no** K-Means.  
- `forecast_df2, mape2, cat_mape2 = fit_and_forecast(df2)`  
- This uses **ground-truth categories** for aggregation and walk-forward. The printed “True-label MAPE” is the **forecast-model-only MAPE** (e.g. 15.7%).

**Where the comparison happens**  
- The comparison is **not** inside `fit_and_forecast`. It’s in the `__main__` block: two separate calls to `fit_and_forecast` with two different DataFrames that differ only in how `category` was set (K-Means vs keyword).  
- MAPE is computed **inside** `fit_and_forecast` → `walk_forward_validate()`: for each (month, category) in the validation loop, APE = |actual − predicted| / max(|actual|, 1e-9) × 100; overall MAPE = mean(APE).  
- So we get two numbers: `mape` (pipeline) and `mape2` (true-label). The **decomposition** is by **interpretation**: the gap (e.g. 29.4% − 15.7% = 13.7 percentage points) is attributed to clustering contamination (wrong categories → wrong monthly totals → harder to predict).  
- The code does **not** compute “clustering error” as a separate number; it only runs the same pipeline twice and you compare the two MAPEs.

**What to say in an interview**  
- “We run the full forecast pipeline twice on the same transactions: once with K-Means-assigned categories and once with ground-truth categories from keyword rules. The same walk-forward validation and MAPE are used in both runs. The difference between the two MAPEs (e.g. 29% vs 16%) isolates the effect of clustering noise: the 16% is the ceiling for the forecast model when labels are correct; the extra ~13 points come from misclassified transactions affecting monthly totals.”

---

## 6. Database Layer

**How SQLite is used**  
- One file: `plaincents.db` (path from `config.DB_PATH`).  
- `get_connection()` in `db/database.py` opens it, sets `PRAGMA journal_mode=WAL`, and runs `schema.sql` so all 6 tables exist.  
- Every insert/upsert does `conn.commit()`. No connection pooling; main opens one connection and passes it through the pipeline.

**Tables**  
| Table | Purpose |
|-------|--------|
| **transactions** | One row per transaction: session_id, date, merchant, amount, category, cluster_id. Append-only per run. |
| **predictions** | One row per forecast: session_id, category, month_offset (1/2/3), forecast_month, predicted_amount. Append-only per run. |
| **portfolio** | Per-run snapshot: session_id, ticker, shares, avg_cost, current_price, pnl. |
| **price_cache** | One row per ticker: ticker (UNIQUE), current_price, fetched_at. UPSERT by ticker; 1-hour TTL used in portfolio logic. |
| **monthly_summary** | One row per month: month (UNIQUE), total_spend, category_spend_json, forecast_next_month, portfolio_value. UPSERT on month so re-runs overwrite. |
| **forecast_vs_actual** | Monitoring: category, forecast_month, predicted_value, actual_value, absolute_error, pct_error. Filled when actuals exist for a month that was previously forecast (e.g. in main.py Step 5d). |

**Why persistence is useful**  
- Enables re-running the pipeline without losing history (sessions keyed by `session_id`).  
- PowerBI (or any client) can read from one DB.  
- forecast_vs_actual supports “predicted vs actual” reporting.  
- price_cache avoids hitting yfinance on every run (TTL in application logic).

---

## 7. Key Engineering Decisions

| Decision | Reasoning |
|---------|-----------|
| **K-Means for categorization** | No labeled data; need unsupervised grouping. Merchant text + amount + date give a feature space where similar behavior clusters together; majority-vote mapping converts clusters to human category names. |
| **Random Forest for forecasting** | Tabular, small-to-medium data; RF handles mixed features and non-linearity without heavy tuning. Interpretable (feature importance) and robust. |
| **TF-IDF + L2 for merchant** | Merchant is the main signal for category. TF-IDF captures which terms appear; L2 normalization makes Euclidean distance in K-Means more like cosine similarity, which suits text. |
| **Walk-forward validation** | Time series; we must not use future data. Expanding-window walk-forward mimics real deployment (train on past, predict next month) and gives an honest MAPE. |
| **SQLite instead of Postgres** | Single-user, local, no network; SQLite is enough. Simpler setup and no server; good for MVP and portfolio projects. |
| **Separate diagnostic run for true-label MAPE** | Lets us quantify how much of pipeline MAPE is due to clustering vs the forecast model, without changing the production code path (production always uses K-Means labels). |

---

## 8. System Limitations

- **Clustering:** “Other” and similar catch-all categories have no clear TF-IDF signature; one or more clusters may not map to them, so some transactions are forced into other categories. Held-out accuracy is on 40 rows (high variance). ARI on full data is more stable.  
- **Features:** No external signals (e.g. merchant IDs, geography). Only one lag and fixed rolling windows; no automatic lag selection.  
- **Forecasting:** Same RF for all categories; no per-category tuning. +2 and +3 month forecasts use same last-known rolling/lag (no chaining of predictions), which can understate uncertainty.  
- **Scale:** In-memory pandas and single SQLite file; not built for millions of transactions or distributed compute.  
- **Improvements:** Add more lags or simple AR terms; try per-category models or at least per-category hyperparameters; add confidence intervals; consider supervised categorization if some labels become available; move to a proper DB and batch processing if scale grows.

---

## 9. Simple Interview Explanation

You can say something like:

“PlainCents is an end-to-end ML pipeline for personal finance. It takes raw bank CSV exports and, without any hand-labeled categories, assigns each transaction to one of eight expense categories using K-Means on features from merchant text (TF-IDF), amount, and date. Then a Random Forest model forecasts total spending per category for the next three months using walk-forward validation. We store everything in SQLite and use the same data for a dashboard. To separate clustering error from forecast error, we run the same forecast pipeline twice—once with K-Means labels and once with ground-truth labels from keyword rules—and compare MAPEs so we can explain the gap as clustering contamination.”

---

## 10. Hard Technical Questions and Answers

**1. How do you know your clustering is good enough for the downstream forecast?**  
We don’t only look at held-out accuracy (40 rows). We run a diagnostic: same forecast pipeline with K-Means labels vs ground-truth labels. The MAPE gap (e.g. 29% vs 16%) quantifies how much error is due to wrong categories. We also report ARI on the full dataset so clustering quality is measured on all rows, not just a small hold-out.

**2. Why is walk-forward validation used instead of a single train/test split?**  
Spending is a time series; we must not use future data in training or in features. A single split would allow leakage from future months. Walk-forward with an expanding window trains on 1..M−1 and tests on M, then expands, which matches how we’d use the model in production and gives a realistic MAPE.

**3. Exactly how did you decompose pipeline error into clustering vs forecast error?**  
We run the **same** `fit_and_forecast()` twice: first on data with `category` from K-Means, second on data with `category` from keyword-based ground truth. Same aggregation, same features, same walk-forward, same MAPE formula. The first MAPE is the full pipeline; the second is the forecast model when labels are correct. The difference is interpreted as the part due to clustering. We do not run a single model that outputs two MAPEs; we run the full pipeline twice with two label sources.

**4. How do you avoid data leakage in the forecast, especially with lag_1_spend?**  
In walk-forward, for a test month M, training uses only months &lt; M. When we build the test row for month M, `lag_1_spend` is set to the **last month’s spend from the training history** for that category (e.g. M−1), not from the test month. So lag is always “last known” from the past. In `fit_and_forecast`, the +1/+2/+3 month forecast rows use the last available actual spend from the series as lag, again no future data.

**5. Why 12 clusters when you have 8 categories?**  
We have 8 target categories but more than 8 behavioral clusters in the feature space (e.g. “recurring utilities” vs “one-off bills”). Using 12 clusters gives the majority-vote mapping more flexibility to assign clusters to categories and improves held-out accuracy. Some clusters may map to the same category; that’s acceptable.

---

## Diagnostic: Exact Implementation Summary

- **Does the code run the forecast twice with two different label sets?**  
  **Yes.** In `pipeline/forecast.py` `__main__`:  
  - First: `df = predict_categories(df)` then `fit_and_forecast(df)`.  
  - Second: `df2["category"] = _get_true_labels(df2)` then `fit_and_forecast(df2)`.  
- **Where does the comparison happen?**  
  In the same `__main__` block, by printing both MAPEs (`mape` and `mape2`). There is no function that returns “clustering MAPE” and “forecast MAPE”; the decomposition is the **difference** between the two runs.  
- **So you can say:** “We run the full forecast pipeline twice—once with K-Means labels, once with ground-truth labels—and compare the two MAPEs to isolate clustering contamination,” and that matches the code.
