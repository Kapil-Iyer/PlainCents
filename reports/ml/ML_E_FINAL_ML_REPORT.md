# ML-E: Final ML Report

Status: ML track scientifically and operationally closed for V2 (pending Cursor audit + human authorization to commit). Written at HEAD `9e1c8877ab59ccf7d27b60e7130d4adbfe65ecfa`, ML-C freeze at `2c06181a12fb270e6e534564c98ccebd2998088c`.

This report is the single narrative account of PlainCents's ML work: what V1 did, why it wasn't good enough evidence, how V2's evaluation was redesigned, what was found, what shipped, and what is and is not proven. Negative results are not buried — K-Means's near-chance VALIDATION score and Naive beating every fitted forecaster are as load-bearing to this report as the numbers that made it into production.

## 1. V1 ML architecture

V1 (`pipeline/cluster.py`, `pipeline/forecast.py`) is a batch CLI pipeline: `python main.py` ingests a CSV, categorizes with K-Means, forecasts with Random Forest, writes SQLite, and renders a PDF/PowerBI export. Categorization: `KMeans(n_clusters=12)` fit on TF-IDF + amount + day-of-week features, cluster→category assigned by majority vote against keyword-derived heuristic labels. Forecasting: `RandomForestRegressor` on 8 engineered features (rolling means/std, lag-1, calendar flags) via expanding-window walk-forward validation with conditional `GridSearchCV`.

## 2. Why V1 evaluation was insufficient

Three compounding problems, confirmed by direct code reading (`reports/ml/PRE_EXPERIMENT_REPORT.md` §§9-16):
- **Ground truth was circular.** V1's "true" categorization labels came from `pipeline/cluster.py::_get_true_labels`, a keyword-substring dictionary — the same kind of signal K-Means's TF-IDF features could trivially exploit. A 90% "accuracy" measured against labels derived from the same surface text as the features is not independent evidence of real category understanding.
- **No held-out generalization test.** V1's 160/40 split was an internal split of the *same* synthetic, same-distribution dataset used to author the keyword labels — no merchant-level separation, so a merchant seen in "train" could recur verbatim in "test."
- **Forecasting validation, while directionally sound (expanding-window, no shuffling), was never compared against a naive baseline** — so "15.7% MAPE" had no reference point establishing the model was earning its complexity.

## 3. V2 evaluation redesign

`docs/V2_ML_SPEC.md` (frozen before any ML-B code ran) fixed, in advance: the exact candidate sets for both problems (§5, §11), the split protocol (§6: merchant-grouped, category-stratified for categorization; §12: chronological expanding-window for forecasting), the metrics (§7, §14), and a strict FINAL-evaluation discipline (§6/§20: VALIDATION drives every decision, FINAL is consulted exactly once, on the already-selected candidate only, after selection is frozen). This session's own addendum additionally permits a bounded reproducibility re-run of FINAL for the selected candidate only — used in ML-E, see §24.

## 4. Evidence tiers

- **Categorization: Tier B** — an independently curated benchmark (`data/evaluation/tier_b_benchmark.csv`, 228 rows / 81 merchant groups), hand-authored with a merchant vocabulary verified non-overlapping with V1's `MERCHANT_KEYWORDS`, specifically to break the circularity problem in §2. Still not Tier A (real bank data) — no real bank export has ever been used anywhere in this repo's ML evaluation.
- **Forecasting: Synthetic** — `data/raw/synthetic_24mo.csv` (779 rows, 24 months), categorized read-only through the production K-Means artifact. Legitimate for testing forecaster *mechanism and behavior* (does expanding-window validation work, does a simple baseline beat a complex one), never a real-world spending-accuracy claim.

## 5. Categorization problem definition

8-way multiclass classification over `config.CATEGORIES` (fixed taxonomy, PRD §9.4, unchanged throughout ML-B/C/D). Input: free-text `merchant` description. Output: one of 8 categories.

## 6. Why merchant-group leakage matters

If the same merchant string ("STARBUCKS #4521") appears in both TRAIN and VALIDATION/TEST, a model can succeed by memorizing that specific string rather than learning transferable signal — inflating the measured score relative to how the model will perform on a genuinely new merchant. `ml/common/splitting.py::merchant_grouped_stratified_split` assigns every group to exactly one partition, verified structurally (`verify_split_isolation`) with zero tolerance for intersection.

## 7. Tier B benchmark construction/provenance

`data/evaluation/tier_b_benchmark.csv`: 228 rows, 81 merchant groups, hand-authored by the project's single author, category distribution imbalanced but every category has ≥8 merchant groups (Food & Dining 46, Transport 35, Entertainment 29, Rent & Utilities 29, Shopping 27, Subscriptions 21, Other 21, Healthcare 20). Vocabulary independently verified non-overlapping with V1's keyword dictionary. `data/evaluation/tier_b_split_v1.json` freezes the merchant→partition assignment (seed 42): TRAIN 133 rows/47 groups, VALIDATION 50 rows/17 groups, FINAL_TEST 45 rows/17 groups.

## 8. Candidate categorization models

Exactly three, per the frozen spec (`ml/categorization/candidates.py`):
1. **K-Means** — re-evaluated under corrected isolation (TRAIN-only fit of scaler/vectorizer/KMeans/cluster-to-category mapping).
2. **TF-IDF + Logistic Regression** — `TfidfVectorizer(max_features=50, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b", ngram_range=(1,2), sublinear_tf=True)` → `LogisticRegression(C=1.0, max_iter=1000, random_state=42)`.
3. **TF-IDF + Linear SVM** — same TF-IDF config, `LinearSVC`.

## 9. Validation metrics

Macro F1 (primary, per §7's imbalanced-class rationale) and accuracy (secondary), computed on VALIDATION for model comparison.

## 10. Error analysis

`reports/ml/results/categorization_error_analysis.json` — structured per-category confusion analysis on VALIDATION, computed only for candidates already fit on TRAIN; FINAL_TEST was never touched during this analysis (§6/§20 discipline).

## 11. ML-C selection

VALIDATION macro F1: K-Means 0.0566, TF-IDF+LogReg 0.2552, TF-IDF+LinearSVM 0.2405. K-Means's 12.0% accuracy is statistically indistinguishable from the ~12.5% random-chance floor for 8 categories. LogReg beat LinearSVM by a modest but consistent margin. **Selected: TF-IDF + Logistic Regression** (`reports/ml/ML_C_SELECTION_RECORD.json`).

## 12. Selected LogReg recipe

Exactly the `TfidfLogRegCandidate` configuration in §8 above, fit on TRAIN only (`fit_isolation` field in the selection record explicitly records this). No amount/day-of-week/is-weekend features — merchant text only.

## 13. Held-out categorization result

FINAL_TEST (45 rows / 17 merchant groups, sealed until this single pass): **macro F1 = 0.4405421207145345, accuracy = 0.4222222222222222** (`reports/ml/results/final_categorization.json`). Per-category F1 ranges 0.22 (Rent & Utilities, Entertainment) to 1.0 (Other), each on only 4-9 support rows.

## 14. Categorization limitations

Tier B, not Tier A — never claim real-world bank-transaction accuracy from this number (see `ML_E_CLAIM_MATRIX.json` for exact wording rules). n=45 is a small single measurement, not a distribution; FINAL_TEST scoring higher than VALIDATION (0.4405 vs 0.2552) is disclosed in `ML_C_EXPERIMENT_REPORT.md` §25/§29 as small-sample volatility, not a resolved phenomenon, and was not used to reopen selection.

## 15. Forecasting problem definition

Per-category monthly spend forecasting, 3-month horizon (+1/+2/+3), 8 categories, on top of `pipeline.forecast.aggregate_monthly`'s (month, category) totals.

## 16. Chronological evaluation requirement

Time series must never be shuffled or split randomly — a model must only ever see months strictly before the month it's predicting, mirroring how the deployed system will actually be used (no future data leakage).

## 17. Expanding-window validation

For each of 14 origins (`reports/ml/ML_C_FOLD_STABILITY.json`), train on all months before the origin, predict the next 1-3 months, score, then advance the origin — the window only grows, never resets or slides backward.

## 18. Candidate forecasting models

Four, per the frozen spec (`ml/forecasting/baselines.py`, `pipeline/forecast.py`): Naive (lag-1), Seasonal Naive (same month, prior year), Random Forest, Ridge — RF and Ridge each evaluated under two multi-step strategies (last-known-history, recursive).

## 19. WAPE / MAE / RMSE / MAPE roles

WAPE (weighted absolute percentage error, sum of absolute errors over sum of actuals) is primary — robust to near-zero-spend months where MAPE explodes (observed directly: `final_forecasting.json`'s `mape_all` is a nonsensical 4.86e11 due to a near-zero actual, versus a sane `mape_nonzero_only` of 15.8). MAE/RMSE are supporting diagnostics.

## 20. Fold-stability findings

Across 14 origins: Naive mean WAPE 0.191 (std 0.041), beats Seasonal Naive in 11/11 comparable origins, by construction never "beats itself." RF (last-known-history) beats Naive in only 6/14 origins (43%); RF (recursive) in 4/14 (29%); Ridge (last-known-history) in 3/14 (21%); Ridge (recursive) in 4/14 (29%). No fitted candidate reliably clears Naive.

## 21. §14 eligibility rule

The frozen spec's explicit rule (§14, quoted in `docs/V2_ML_SPEC.md:381`): if RF/Ridge do not reliably beat naive/seasonal-naive with reasonable stability, the scientifically correct decision is to ship the simpler model. §20's findings triggered exactly this outcome.

## 22. Selected Naive model

`ml/forecasting/baselines.py::naive_predict(spend_history) -> spend_history[-1]` — the most recently observed month's actual total, unchanged.

## 23. Strategy N/A

Naive has no meaningful recursive/last-known-history distinction: repeating the last observation recursively produces the identical value a non-recursive repeat would. Selected strategy recorded as `"N/A"` (`ML_C_SELECTION_RECORD.json`).

## 24. Final temporal result

Reserved period 2024-10/11/12, trained on 2023-01 through 2024-09 (21 months), 24 predictions (8 categories × 3 horizons). **Combined WAPE = 0.18865752437529387** (`reports/ml/results/final_forecasting.json`). Per-horizon: +1 = 0.1938, +2 = 0.1469, +3 = 0.2217. Re-verified byte-identical (modulo `git_commit`/timestamp) by re-running `ml/forecasting/run_final.py` during ML-E, per this session's addendum to the FINAL-test discipline (Section 5) — see `ML_E_REPRODUCIBILITY.md` §Verification.

## 25. Forecasting limitations

Synthetic evidence tier throughout — never a real-world spending-accuracy claim. `mape_all` is not usable (near-zero-actual blowup); `mape_nonzero_only` (15.8%) and WAPE (18.87%) are the only meaningful percentage-error summaries. Rent & Utilities has the largest per-category WAPE (0.765) — a single 3-prediction category, not a stable estimate.

## 26. ML-D production integration

`backend/services/categorization_service.py` now loads `models/tfidf_logreg_v1.pkl` (built by `scripts/build_production_logreg_model.py`, fit on the frozen Tier B TRAIN partition only — 133 rows / 47 merchant groups, never VALIDATION or FINAL_TEST). `pipeline/forecast.py::train_and_predict` (the only function `ForecastService.run_forecast()` calls) now implements Naive directly instead of fitting a Random Forest. No API/schema/frontend contract changes (`docs/V2_ML_SPEC.md:499`).

## 27. Categorization artifact lifecycle

Built offline (`python -m scripts.build_production_logreg_model`), loaded once at FastAPI startup (`backend/main.py`'s lifespan hook), held in memory for the process lifetime. Missing/corrupt artifact reports `status="missing"/"error"` via `/api/health` rather than crashing; prediction-dependent writes get a 503 until a valid artifact is present.

## 28. Forecasting runtime lifecycle

No persisted artifact — Naive is parameterless code, recomputed fresh from the caller's current `monthly_df` on every `POST /api/forecasts/run` call. `GET` endpoints never touch it (verified by `test_check_status_never_fits`/`test_get_latest_never_fits`).

## 29. Correction workflow

A user correction writes `confirmed_category` only; `predicted_category` (written once, at import/creation time) is never overwritten. `effective_category = COALESCE(confirmed_category, predicted_category)`, computed by a SQL view, used by both dashboard and forecast aggregation.

## 30. Forecast staleness workflow

`date`/`amount`/`confirmed_category` changes, transaction creation, transaction deletion, and import confirmation all call `ForecastService.mark_stale(reason)`, which flips `is_stale` on the latest non-stale run for the current `data_mode` (idempotent — an already-stale run's original reason is preserved). Forecast generation itself remains manual (`POST /api/forecasts/run`); nothing auto-regenerates.

## 31. Selected-vs-rejected model preservation

K-Means (`pipeline/cluster.py`, `models/kmeans_model.pkl`), Random Forest/Ridge/Seasonal Naive (`pipeline/forecast.py::fit_and_forecast`/`walk_forward_validate`, `ml/forecasting/baselines.py::seasonal_naive_predict`), and all ML-B/ML-C evidence (`reports/ml/`) are untouched — ML-D only changed which implementation the production *service* calls, never deleted or overwrote the rejected candidates' code or evidence.

## 32. Reproducibility workflow

See `ML_E_REPRODUCIBILITY.md` for exact commands. Summary: seeds fixed at 42 throughout; `data/evaluation/tier_b_split_v1.json` freezes the categorization split; `ml/categorization/run_final.py`/`ml/forecasting/run_final.py` are deterministic and were re-run during ML-E, reproducing byte-identical metrics (only `git_commit`/timestamp differ).

## 33. Claim-certification summary

See `ML_E_CLAIM_MATRIX.json` for the full 23-claim table. Headline: real numbers (42.2%/0.4405 macro F1; 18.87% WAPE) are SUPPORTED_WITH_QUALIFICATION — true, but tied to Tier B/synthetic evidence tiers, never to be described as real-world performance.

## 34. Known limitations

Small Tier B sample (228 rows total); no real bank data anywhere in ML evaluation; forecasting evidence entirely synthetic; FINAL_TEST's higher-than-VALIDATION categorization score is unexplained small-sample volatility, disclosed not resolved; TD CSV support is fixture-tested, not field-verified against a real export.

## 35. Future ML directions (explicitly NOT required for V2 completion)

A Tier A (real, consented bank-transaction) categorization dataset; a larger Tier B sample; real transaction history for forecasting evaluation (once available in production, real user data could eventually support a genuine forecasting re-evaluation — not synthetic); investigating why FINAL_TEST scored higher than VALIDATION for categorization; a confidence-score UI surface (LogReg's `predict_proba` exists but is currently unused).
