# PlainCents V2 — ML-B: Scientific Evaluation Report

**Status:** ML-B complete. This report documents TRAIN/VALIDATION evidence only.
**Commit evaluated:** `bc2735369a2ebe420d2c7d02395edca1ca96e427` (plus this ML-B work, uncommitted at time of writing)
**Authority:** `docs/V2_ML_SPEC.md` (frozen). Where this report and the ML Spec appear to disagree, the ML Spec wins — flag it, don't silently resolve it in the report's favor.
**Reproduce:** `python -m ml.run_all` (resets `reports/ml/results/experiment_log.jsonl`, regenerates every file in `reports/ml/results/`). Requires the pinned environment in `requirements.txt`. Seed: 42 throughout.

> **ML-B DOES NOT SELECT A PRODUCTION WINNER.** Every number below is a VALIDATION or TRAIN diagnostic result. The FINAL UNTOUCHED TEST partition (categorization) and the reserved final temporal period (forecasting) were **not evaluated, not inspected, and remain sealed** — see §7 and §19. Model/strategy selection is ML-C's job, done later, from this evidence.

---

## 1. What question ML-B was trying to answer

Not *"which model should PlainCents ship?"* — that is ML-C. ML-B answers a narrower, prior question: **given the frozen candidate sets and the frozen evaluation methodology (ML Spec §5–§17), what does TRAIN+VALIDATION evidence actually show?** That includes negative and inconvenient results. A candidate performing badly, or a baseline beating a more sophisticated model, is not a failure of ML-B — reporting that honestly *is* ML-B succeeding.

## 2. Why V1's evidence was insufficient

V1's headline numbers (90.0% categorization accuracy, 29.4%/15.7% forecast MAPE) were computed under conditions that don't generalize:

- The 90.0% figure came from 40 rows drawn from the **same 200-row pool** used to build the K-Means cluster→category mapping, and that pool's labels were assigned by keyword rules written by the same person who wrote the synthetic merchant vocabulary (`MERCHANT_KEYWORDS` in `pipeline/cluster.py` vs. `scripts/generate_synthetic_24mo.py`) — the two share substrings like `"NETFLIX"` deliberately. Evaluating on data this coupled to its own labeling rule cannot show how the model handles a genuinely unfamiliar transaction description.
- V1's optional `GridSearchCV` step used `TimeSeriesSplit` on rows sorted `(category, month)` — row-count-based folds do not respect calendar-month boundaries when 8 categories are interleaved, so a fold boundary can land mid-month.
- Neither evaluation reserved a temporal or merchant-based holdout that was genuinely never consulted during development.

ML-B's job was to remove exactly these three problems: decouple the categorization evaluation data from the labeling rule, split by calendar month boundary (not row count) for forecasting, and hold a slice of each back until a later, single, untouched pass.

## 3. Datasets and evidence tiers actually available

| Dataset | Tier | Used for | Provenance |
|---|---|---|---|
| `data/raw/synthetic_24mo.csv` (779 rows, 2023-01–2024-12) | Synthetic (ML Spec §3.1) | Forecasting bake-off (Part B) | `scripts/generate_synthetic_24mo.py`, hand-authored generator |
| `data/evaluation/tier_b_benchmark.csv` (228 rows, 81 merchant groups) | **Tier B** — independently curated/constructed benchmark (ML Spec §3.2) | Categorization bake-off (Part A) | Hand-authored for ML-B (`ml/data/build_tier_b_benchmark.py`), this session |

**No Tier A (real, naturally-occurring transaction data) and no vetted Tier C (genuinely-natural public dataset) were available.** This was surfaced as a blocking decision before any code was written (ML Spec §24 item 1) and the user explicitly chose the Tier B path over supplying Tier A data, searching for Tier C, or halting ML-B. That choice is recorded here so the tier ceiling on every categorization claim below is traceable to a decision, not an oversight.

**No claim in this report, and no claim that may ever be built on it, may describe Tier B categorization numbers as real-world TD accuracy.** They describe performance on a hand-constructed benchmark only (ML Spec §21).

## 4. What the labels mean and how they were produced

- **Categorization (Tier B):** each of the 228 rows' `true_category` was assigned by hand, once, at authoring time, in `ml/data/build_tier_b_benchmark.py` — a static per-row value, not a runtime keyword-matching function. This is closer to a human-annotation process than V1's `_get_true_labels()` (which computes a label from merchant text on every call). The **honest limitation**: the same person who wrote the descriptions also assigned the labels — a true independent-annotator process would be stronger evidence. What the benchmark *does* buy: vocabulary and formatting genuinely independent of `MERCHANT_KEYWORDS`/the synthetic generator (verified programmatically — zero substring overlaps after two rounds of conflict detection and fixing; see §28).
- **Forecasting (synthetic):** monthly category totals are built by running the production K-Means artifact (`models/kmeans_model.pkl`, read-only, never retrained here) over `synthetic_24mo.csv`, matching the ML Spec §2 interpretation of "the relevant current end-to-end synthetic result" (K-Means-derived categories feeding the forecaster) rather than the heuristic-label bypass V1's diagnostic used.

## 5. Why merchant-grouped splitting matters

A classifier can score well by memorizing a specific merchant string rather than learning what makes a transaction belong to a category. If "TIM HORTONS #1234" is in TRAIN and "TIM HORTONS #5678" is in TEST, a model doing pure string memorization still scores correctly — but that's not evidence it would generalize to a merchant it has never seen. Real bank exports repeat merchants with varying suffixes (store numbers, cities, reference codes), so this isn't a hypothetical: it's the default failure mode of any naive random split. The Tier B benchmark was authored specifically to have this structure — `merchant_group` identities (e.g. `"STARBUCKS COFFEE"`) with 2–6 description variants each — so the split code has something real to protect against.

## 6. How TRAIN / VALIDATION / reserved FINAL were constructed

`ml/common/splitting.py`'s `merchant_grouped_stratified_split`: within each of the 8 categories independently, merchant groups are shuffled (seed 42) and divided (~60/20/20 by group count, with a rounding rule guaranteeing every category with ≥3 groups contributes at least one group to VALIDATION and one to FINAL_TEST). All transaction rows belonging to a merchant group inherit that group's single partition assignment — no row-level splitting.

**Actual frozen split** (`data/evaluation/tier_b_split_v1.json`, seed=42):

| Partition | Rows | Merchant groups |
|---|---|---|
| TRAIN | 133 | 47 |
| VALIDATION | 50 | 17 |
| FINAL_TEST | 45 | 17 |

Every one of the 8 categories has ≥1 merchant group in TRAIN and ≥1 in VALIDATION (verified programmatically, not just by construction — see `tests/ml/test_splitting.py::test_tier_b_benchmark_actually_splits_cleanly`).

## 7. How FINAL remained sealed during ML-B

Three independent layers, not just a naming convention:

1. `ml/categorization/run_bakeoff.py` loads all three partitions into memory (it has to, to build the split), but `final_df` is passed to exactly one function all run: `assert_final_test_sealed()`, which checks merchant-group set disjointness and raises `AssertionError` otherwise. It is never passed to any candidate's `.fit()` or `.predict()`, and never scored.
2. `verify_split_isolation()` is called before any candidate is fit; if any of the three pairwise merchant-group intersections is non-empty, `run()` raises `RuntimeError` before training starts — a leak would halt the whole run, not just get logged.
3. `tests/ml/test_splitting.py` and `tests/ml/test_kmeans_isolation.py` assert this structurally and are part of the repo's permanent regression suite, not one-off checks.

For forecasting, sealing is chronological rather than merchant-based: `ml/forecasting/temporal_eval.build_folds()` restricts every VALIDATION fold to the "development region" (`all_months[: len(all_months) - n_final_reserved_months]`) at the data-structure level — a reserved month literally cannot appear in a fold's `train_months` or `target_months` list, and `assert_no_reserved_month_used()` is called immediately after generating folds, before any model is fit.

**What was actually reserved and never touched:** categorization — 45 rows across 17 merchant groups (all 8 categories represented). Forecasting — the 3 most recent calendar months, **2024-10, 2024-11, 2024-12**. No metric, error example, or prediction from either was computed, printed, or written to any file in this ML-B pass.

## 8. How leakage was prevented (summary — mechanism detail in §9–§11)

Three distinct leakage channels were addressed, each with its own dedicated code path and test:
- **Merchant leakage** (categorization) — §6/§7 above.
- **Fitting leakage** (K-Means's two-stage fit+mapping structure) — §9 below.
- **Temporal leakage** (forecasting) — §10/§18 below, verified by `tests/ml/test_temporal_eval.py` and `tests/ml/test_forecast_leakage.py`.

## 9. How K-Means evaluation differs from V1

V1's `fit_and_evaluate()` draws its 160-row "mapping" set and 40-row "held-out" set from the *same* synthetic/heuristic-labeled pool — both slices see the same generator vocabulary. `ml/categorization/candidates.py::KMeansCandidate` instead:

1. Fits `StandardScaler` + `TfidfVectorizer` (reusing `pipeline.features.build_feature_matrix`, unmodified) on **TRAIN only** (133 rows).
2. Fits `KMeans(n_clusters=12, random_state=42, n_init=50)` — same hyperparameters as V1, not re-tuned — on the TRAIN feature matrix only.
3. Builds the cluster→category mapping by majority vote over **TRAIN labels only**, using the *entire* TRAIN set (no internal 160/40 re-split within TRAIN — VALIDATION already plays that held-out role at the outer level, per ML Spec §6.1's explicit instruction not to build an ad hoc internal split).
4. VALIDATION rows are transformed with the already-fitted scaler/vectorizer and predicted with the already-fitted KMeans model — `predict()` never calls `.fit()` on anything (verified by `tests/ml/test_kmeans_isolation.py::test_predict_on_final_test_like_rows_never_calls_fit`, which monkeypatches `KMeans.fit_predict` to count calls and asserts exactly one across a `.fit()` + two `.predict()` calls).

This module never imports `pipeline.cluster.fit_and_evaluate`/`predict_categories` and never writes to `models/kmeans_model.pkl` — it is a fully separate, evaluation-only K-Means instance.

## 10. How cluster→category mapping works

For each of the 12 cluster IDs: among TRAIN rows assigned to that cluster, take the majority-vote `true_category`. An empty cluster (zero TRAIN rows assigned) falls back to the alphabetically-first category present in TRAIN labels — a documented, deterministic, arbitrary tie-break, same spirit as V1's own fallback to `CATEGORIES[0]`, not a modeling decision.

## 11. How TF-IDF represents transaction text

Each cleaned merchant string becomes a sparse vector over up to 50 vocabulary terms (unigrams and bigrams of ≥2-letter words), weighted by term frequency (log-scaled, `sublinear_tf=True`) times inverse document frequency, then L2-normalized. Intuition: a word that appears in *most* merchant strings (like "INC" or "STORE") gets down-weighted; a word that appears in *few* but appears *strongly* in those few gets up-weighted. This configuration is reused verbatim from `pipeline/features.py` (not re-tuned) for both K-Means and the two supervised candidates, per ML Spec §A4's instruction to avoid an unmotivated hyperparameter search.

## 12. Why Logistic Regression is a reasonable candidate

It's the natural supervised counterpart to K-Means's already-adopted TF-IDF representation: same features, but the category→feature mapping is learned directly from labels instead of discovered unsupervised and labeled after the fact. It's cheap, produces genuine probabilities (unused by the product today, but available if ever needed), and its coefficients are directly inspectable per category/term — useful for the error analysis in §8/§15 below.

## 13. Why Linear SVM is a reasonable candidate

A standard strong baseline specifically for sparse, high-dimensional TF-IDF features — margin-maximization tends to generalize better than distance-based methods (like KNN, deliberately excluded, ML Spec §5) in this kind of feature space, without KNN's linear-in-training-set-size inference cost.

## 14. Categorization VALIDATION results — all three candidates

50 VALIDATION rows, 8 categories, all candidates fit on the same 133 TRAIN rows.

| Candidate | VALIDATION macro F1 | VALIDATION accuracy | TRAIN accuracy (diagnostic) |
|---|---|---|---|
| K-Means (TRAIN-only isolated) | **0.0566** | 12.0% | 42.1% |
| TF-IDF + Logistic Regression | **0.2552** | 32.0% | — |
| TF-IDF + Linear SVM | **0.2405** | 26.0% | — |

(Full per-category precision/recall/F1 and confusion matrices: `reports/ml/results/categorization_results.json`.)

**These numbers are far below V1's reported 90.0%, and that gap is the single most important finding of Part A** — see §16.

Random-chance accuracy with 8 roughly-balanced categories is ~12.5%. K-Means's 12.0% VALIDATION accuracy is statistically indistinguishable from random guessing; both supervised candidates land meaningfully, but modestly, above chance.

## 15. Structured VALIDATION categorization error analysis (ML Spec §8)

All 50 VALIDATION rows were reviewed per candidate (small-dataset case, §8's "review all misclassified rows if the total is small"). Full per-row detail: `reports/ml/results/categorization_error_analysis.json`.

**Dominant, structural failure mode — unseen merchants (§8's first required category):** by construction of a correct merchant-grouped split, *every* VALIDATION merchant group is one K-Means/TF-IDF has never seen. Checking VALIDATION merchant text against TRAIN's fitted 50-term TF-IDF vocabulary directly: **0 of 50 VALIDATION rows share even one vocabulary token with TRAIN's top-50 terms.** That vocabulary turned out to be dominated by merchant-specific proper nouns (`"farm boy"`, `"pizza pizza"`, `"sobeys"`, `"hertz"`, `"kfc"`, ...) rather than generic descriptive words — an artifact of a small (133-row, 47-group) TRAIN set where proper nouns dominate term frequency more than they would at real production scale. This is why K-Means collapses toward a single majority cluster prediction (mostly "Transport") for almost everything, and why both supervised candidates fall back heavily toward the TRAIN-majority class ("Food & Dining").

**Concrete, representative errors, present across all three candidates:**
- `DAIRY QUEEN #217 BRAMPTON` / `DQ GRILL AND CHILL 217` (true: Food & Dining) → predicted Transport (K-Means), Shopping (SVM) — a genuinely unseen fast-food merchant with no lexical overlap to any TRAIN Food & Dining example.
- `ESCAPE ROOM EXPERIENCE` / `PUZZLE ESCAPE GAMES TORONTO` (true: Entertainment) → predicted Transport, Shopping, or Food & Dining depending on candidate — no TRAIN Entertainment example shares vocabulary with "escape"/"puzzle"/"room".
- `E TRFR SENT` / `SENT E-TRANSFER` (true: Other, tagged `generic_transfer_description`) → predicted Food & Dining by Logistic Regression — a maximally uninformative description defaulting to the majority class.

**Where supervised models *did* generalize — a category with recurring generic vocabulary:** Logistic Regression's Rent & Utilities predictions were 100% precise (4/4 correct when predicted, though recall was only 57%) and Subscriptions likewise 100% precise. TRAIN's fitted vocabulary happens to include generic, category-associated words — `"bill"`, `"billing"`, `"monthly"`, `"rent"`, `"rental"`, `"membership"`, `"subscription"`, `"corp monthly"` — that recur across *multiple different* Rent & Utilities / Subscriptions merchants (utility bills and software subscriptions are described similarly regardless of which company issues them), unlike Food & Dining/Shopping/Transport merchants, whose category signal in this benchmark is almost entirely merchant-proper-noun-driven. **This is the clearest positive finding in Part A**: TF-IDF-based models generalize to unseen merchants specifically when the category has recurring *generic* descriptive vocabulary, and fail when it doesn't.

**Other required §8 categories, reviewed and largely absent or inconclusive at this dataset scale:** multi-purpose merchants (Walmart/Costco, tagged in the benchmark) fell in TRAIN for this split and were not exercised in VALIDATION errors this run; refund/credit rows (negative amounts) and malformed/low-information descriptions were present in the benchmark but, at n=50 VALIDATION rows, none happened to land as a *misclassification* this run — their presence in the dataset is confirmed (`error_analysis_tag` column), but 50 VALIDATION rows is too small to guarantee every tagged phenomenon surfaces as an actual error every run. This is recorded as a dataset-scale limitation (§31), not swept under a claim that these failure modes don't exist.

## 16. What the categorization results do and do not prove

**Do not prove:** that PlainCents' categorizer achieves ~90% accuracy on new transactions, that K-Means is a poor model in general, or that Logistic Regression/SVM are production-ready. 228 rows across 81 merchant groups, one author, is a small and imperfect benchmark.

**Do prove:** that V1's 90.0% figure was measuring something much closer to "can this model recognize merchants it was shown during development" than "can this model categorize an unfamiliar transaction" — because correcting exactly one methodological flaw (merchant leakage) while holding everything else constant in spirit collapses K-Means's accuracy to chance level. This is scientifically the most important thing ML-B needed to establish, and it required getting the low numbers, not avoiding them.

## 17. How monthly forecasting examples are constructed

Raw transactions → K-Means-predicted category (production artifact, read-only) → grouped by `(month, category)` → summed. Unlike `pipeline.forecast.aggregate_monthly` (which silently omits a `(month, category)` combination with zero transactions), `ml/forecasting/data_prep.build_monthly_grid()` builds the full 24-month × 8-category grid explicitly and zero-fills missing combinations, per ML Spec §10's explicit requirement that a zero-spend month is a valid data point, not a missing one. This surfaced a real, previously-invisible fact: the production K-Means model assigns **zero transactions to "Other" across all 24 months** of this synthetic dataset (see §26).

## 18. Why expanding-window evaluation is required

A model must only ever be trained on data from before the month it's predicting. A random or shuffled split would let August's actual spend inform a January prediction — meaningless for a genuinely forward-looking forecaster. V1's `walk_forward_validate` already does this correctly for its main loop; ML-B's `ml/forecasting/temporal_eval.py` reimplements the same expanding-window discipline as a general-purpose fold generator so every candidate/strategy shares one leakage-safe harness (verified: `tests/ml/test_temporal_eval.py`, `tests/ml/test_forecast_leakage.py`).

## 19. Exact definitions used for each forecasting baseline/model

- **Naive:** predicted spend for every horizon = the single most recent actual observed month (`lag_1_spend`). Same value for +1, +2, and +3 — there's no "recursive" variant since repeating a fixed value recursively still produces that fixed value.
- **Seasonal Naive:** predicted spend for target month = the actual value from the same calendar month one year earlier. Eligibility requires ≥13 months of TRAIN history relative to that specific horizon (`ml/forecasting/baselines.py::seasonal_naive_predict`); when ineligible, the function returns `(None, False)` — never a fabricated 0 or an extrapolated guess.
- **Random Forest:** `n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42` — the **exact** hyperparameters `pipeline.forecast.train_and_predict()` ships in production, not `walk_forward_validate`'s diagnostic `max_depth=3` and not a GridSearchCV-tuned configuration (ML Spec §11's explicit requirement; a regression test, `tests/ml/test_forecast_leakage.py::test_rf_hyperparameters_match_trd_shipped_defaults_not_diagnostic_defaults`, pins this).
- **Ridge:** `alpha=1.0` (sklearn default, undocumented/untuned by design — ML Spec §11 permits "the smallest defensible TRAIN/VALIDATION-safe approach" when Ridge's hyperparameters aren't frozen), features standardized with a `StandardScaler` fit on that fold's TRAIN rows only.

Feature set for RF/Ridge (identical to V1, reused via `pipeline.forecast.build_forecast_features` for the TRAIN fitting matrix): `month_num, category_encoded, rolling_3m_avg, rolling_6m_avg, rolling_std (3mo, ddof=1), is_december, is_summer, lag_1_spend`.

## 20. Exact definition of each evaluated multi-step strategy

- **Strategy A — last-known-history** (V1's current shipped approach): the rolling/lag features are computed **once**, from real TRAIN history, and reused unchanged for the +1, +2, and +3 feature rows — only the calendar-derived fields (`month_num`, `is_december`, `is_summer`) differ by horizon.
- **Strategy B — recursive**: +1's prediction is appended to the working spend-history array as if it were a real observation before the +2 feature row is built (so `rolling_3m_avg`/`lag_1_spend` at +2 reflect the model's own +1 guess); +2's prediction is likewise appended before +3.
- **Strategy C — direct** (separate per-horizon models): **not implemented in this ML-B pass.** ML Spec §11.1 states it is "only pursued if evidence from A/B shows a genuine horizon-specific pattern... not adopted by default." §22's A-vs-B comparison shows a *consistent, monotonic* degradation pattern for B relative to A (not a pattern suggesting a fundamentally different per-horizon relationship a direct model would specifically fix) — so C was not built. This is a decision made *after* seeing A/B evidence, not a shortcut taken in advance, and is explicitly available to revisit in ML-C if a case emerges.

Strategy A/B distinction was **only evaluated for RF and Ridge** — Naive and Seasonal Naive have no meaningful strategy variant (§19); applying "strategy" terminology to them would imply an experiment that doesn't actually exist.

Correctness of both strategies is verified by construction, not just by inspection: `tests/ml/test_strategies.py` uses a synthetic "echo" model whose prediction *is* one of its input features, proving Strategy A feeds identical history-derived values to all three horizons while Strategy B's +2/+3 features are computed from a history array that has actually grown by one element.

## 21. Exact VALIDATION forecasting results — all required candidates/strategies

14 expanding-window folds (origins with ≥7 TRAIN months, up to 3 horizons each, within the 21-month development region — the 3 most recent months, 2024-10/11/12, reserved and never touched). Full detail including per-category breakdowns: `reports/ml/results/forecasting_metrics.json`; raw long-format predictions: `reports/ml/results/forecasting_predictions_long.csv`.

**Combined (all horizons pooled), primary metric WAPE:**

| Candidate / Strategy | WAPE | MAE | n |
|---|---|---|---|
| **Naive** | **0.1903** | 34.82 | 312 |
| Ridge — last-known-history | 0.2237 | 40.93 | 312 |
| Ridge — recursive | 0.2395 | 43.83 | 312 |
| Random Forest — last-known-history | 0.2423 | 44.34 | 312 |
| Random Forest — recursive | 0.2565 | 46.93 | 312 |
| Seasonal Naive | 0.2631 | 47.14 | 216 (eligibility-limited, §24) |

**Naive has the lowest (best) combined WAPE of every candidate, including both ML/tree-based candidates.** This is a legitimate, spec-anticipated possible outcome (ML Spec §14: "if RF or Ridge does not reliably clear naive... the scientifically correct decision is to ship the simpler model") — not a bug, and not something ML-B is permitted to act on by itself (that's ML-C's job).

## 22. +1 / +2 / +3 horizon behavior

| Candidate / Strategy | +1 WAPE | +2 WAPE | +3 WAPE |
|---|---|---|---|
| Naive | 0.2020 | 0.1904 | 0.1765 |
| Seasonal Naive | 0.2631 | 0.2631 | 0.2631 |
| RF — last-known-history | 0.2426 | 0.2354 | 0.2495 |
| RF — recursive | 0.2426 | 0.2524 | 0.2771 |
| Ridge — last-known-history | 0.2020 | 0.2235 | 0.2493 |
| Ridge — recursive | 0.2020 | 0.2347 | 0.2887 |

Per ML Spec §13.1/§14's explicit requirement to check for "a good +1 result masking a poor +2/+3 result": **Ridge's last-known-history +1 WAPE (0.2020) is identical to Naive's**, but Ridge visibly degrades at +2/+3 (0.2235, 0.2493) while Naive holds steady or improves — a +1-only report would have hidden that Ridge is not actually competitive at the horizons that matter most for the product's 3-month forecast view. **The recursive strategy's cost compounds specifically with horizon**: RF-recursive and Ridge-recursive both start identical to their last-known-history counterparts at +1 (expected — no feedback has occurred yet) and then diverge increasingly at +2 and sharply at +3 (Ridge-recursive: 0.2493→0.2887, RF-recursive: 0.2495→0.2771 vs. last-known-history's 0.2493/0.2495) — direct, quantified evidence that feeding predictions back into features accumulates error rather than correcting it, at this data scale.

## 23. Why WAPE is primary

`WAPE = Σ|actual − predicted| / Σ|actual|`, aggregated across the whole evaluated set rather than averaged per-row. A single near-zero actual contributes proportionally to both numerator and denominator, so it can't dominate the aggregate the way it dominates a per-row percentage. Verified directly in `tests/ml/test_metrics.py::test_wape_small_actual_does_not_dominate_like_mape_would`: a $0.01 actual missed by $4.99 barely moves WAPE, but would produce a ~50,000% single-row MAPE.

## 24. Why MAPE can be problematic (and how it showed up in these results)

`mape_safe()` reports MAPE two ways: `mape_all` (V1's exact `max(|actual|, 1e-9)` guard, comparable to V1's own numbers) and `mape_nonzero_only` (excludes rows with |actual| ≤ $1). The gap between the two is large whenever near-zero actuals are present — visible concretely in the "Other" category (§26): every one of its 39 evaluated (month, horizon) points has actual = $0 (K-Means never predicts "Other" for this dataset), so `mape_all` explodes toward the `1e-9`-guard ceiling while `mape_nonzero_only` is undefined (no eligible rows) and **WAPE itself is undefined (NaN, 0/0)** — the correct behavior per its own definition, not a bug (`ml/common/metrics.py::wape` returns `NaN` explicitly rather than a silently wrong 0). Seasonal Naive's eligibility also directly reflects §11's own definition: only origins with ≥12–13 months of TRAIN history (depending on horizon) qualify, which is why it has 216 evaluated rows against every other candidate's 312 — **not** an inconsistency, a documented, expected consequence of a data-dependent eligibility rule (verified: `tests/ml/test_forecast_leakage.py::test_seasonal_naive_ineligible_returns_none_not_zero`).

## 25. History-length sensitivity findings (ML Spec §15)

**Scope limitation, disclosed up front:** only 24 months of synthetic history exist, 3 reserved for the sealed final period. Testing all four required truncation lengths (6/9/12/18 months) on a like-for-like origin set requires ≥18 months of available prior history at that origin — only **3 origins** in the whole dataset satisfy this. The experiment ran on those 3 origins; this is a small, explicitly-flagged sample, not a statistically robust basis for changing the frozen 12-month product rule (`reports/ml/results/history_length_sensitivity.json` records this limitation verbatim).

| Truncated TRAIN history | RF WAPE | RF MAE | Naive WAPE (reference) |
|---|---|---|---|
| 6 months | **n/a** — see below | — | 0.2054 |
| 9 months | 0.3111 | 54.99 | 0.2054 |
| 12 months | 0.1846 | 32.63 | 0.2054 |
| 18 months | **0.1459** | 25.80 | 0.2054 |

**At exactly 6 months of TRAIN history, RF cannot produce a single valid prediction at all** — not a poor score, a structural inability: `pipeline.forecast.build_forecast_features`'s own 6-prior-month rolling-window requirement means a 6-month window yields zero rows surviving `dropna()` for any category. This is itself a finding: **RF's own feature engineering imposes a de facto ≥7-month floor, independent of and tighter than intuition about the product's 12-month eligibility rule** at the very shortest histories. From 9 to 18 months, RF improves monotonically (with a dip at 9 months worse than even Naive) and only overtakes Naive's WAPE at 18 months — RF does **not** show it reliably beats Naive below 18 months of history in this small sample, which, if anything, argues the current 12-month threshold is not obviously "unnecessarily conservative" for RF specifically (though 3 origins is too small a sample to treat as confirmatory either way — ML-C should not over-read this).

## 26. Sparsity / zero-spend findings (ML Spec §16)

Per-category sparsity over the 21-month development region (`reports/ml/results/sparsity_analysis.json`):

| Category | Bucket | Nonzero months |
|---|---|---|
| Food & Dining, Healthcare, Rent & Utilities*, Shopping, Subscriptions, Transport | dense | 20–21/21 |
| Entertainment | intermittent | 20/21 |
| **Other** | **always_zero** | **0/21** |

*Rent & Utilities has exactly 1 zero month within the dev region.

**"Other" is entirely zero-spend across the whole evaluated window** — a consequence of the production K-Means model (as currently trained on this synthetic data) never mapping any cluster to "Other" for these transactions, not a data-generation artifact (the raw synthetic generator does produce "Other"-labeled-by-heuristic transactions; K-Means's cluster→category mapping simply never routes any cluster there). Metric consequence, concretely observed: WAPE is undefined (NaN) for "Other" under every candidate (0/0 by definition), while MAE is trivially near-zero for any candidate (since predicting ≈$0 against an always-$0 actual is nearly free) — **a near-perfect MAE for "Other" reflects the degenerate input, not genuine forecasting skill**, exactly the distortion §16 asks to be surfaced rather than presented as a candidate's strength.

**None of the four candidate eligibility rules named in ML Spec §16 is adopted here** — `sparsity_analysis.json` records how each of the four rules (min total months, min non-zero months, min recent non-zero, survives-dropna floor) would classify every category, for ML-C's future reference, without selecting one.

## 27. Cases where a simpler model matches or beats a more complex one

- **Naive beats both RF and Ridge on combined WAPE, at every individual horizon except Ridge tying Naive exactly at +1** (§21/§22).
- **Ridge (a single linear model) beats Random Forest** on every combined and per-horizon metric reported (e.g., combined WAPE 0.2237 vs. 0.2423) — notable given RF is the current production forecaster and Ridge is the "in-between" candidate the ML Spec added specifically to test whether RF's complexity earns its keep (§11: "exactly the 'is the extra complexity earning its keep' comparison point this bake-off needs"). At this data scale, it apparently does not.
- **Ridge's +1 prediction is numerically indistinguishable from Naive's** (0.2020 WAPE, matching MAE to two decimals) — plausible evidence that, with only a few dozen TRAIN rows per fold, a linear model's fitted coefficients end up dominated by `lag_1_spend` because there isn't enough data to meaningfully weight the other 7 features differently.

## 28. Important failed/invalid experiments or weak results

No experiment is marked INVALID (no leakage bug or implementation error was found in the final harness — see the experiment log, all 9 entries `SUCCESS`). Two rounds of construction did surface and get corrected before the real run, preserved here for transparency rather than silently fixed:

- The first draft of the Tier B benchmark accidentally reused two `MERCHANT_KEYWORDS` substrings (`"PRESTO"` in an OC Transpo description; `"MICROSOFT"` in an Xbox Game Pass description) — caught by a dedicated substring-conflict check (`ml/data/build_tier_b_benchmark.py`'s design doc references this) before the benchmark was used for any evaluation. Not an ML-B experiment result, but exactly the kind of near-miss the whole Tier B exercise exists to prevent, and it is disclosed rather than quietly fixed and forgotten.
- History-length sensitivity's 6-month truncation (§25) produced zero usable RF predictions — reported as a genuine structural finding (RF's feature floor), not hidden or treated as an error.

**Weak results, preserved rather than omitted:** K-Means's near-chance VALIDATION accuracy (§14–§16); RF and Ridge both losing to Naive combined (§21); the recursive strategy underperforming last-known-history at every horizon it was tested at (§22).

## 29. Unexpected findings

- **The magnitude of K-Means's/TF-IDF's VALIDATION collapse.** A drop from V1's reported 90.0% to near-chance (12–32%) after removing exactly one confound (merchant leakage) was larger than a qualitative reading of the ML Spec's warnings alone would predict — it took actually measuring the TRAIN/VALIDATION TF-IDF vocabulary overlap (0 of 50 rows share a single top-50 term) to understand *why* it was this severe: a 133-row TRAIN set's "top 50" TF-IDF terms are mostly merchant proper nouns, not generalizable descriptive words.
- **Recursive multi-step forecasting did not help even at +3**, where one might expect propagated-prediction feedback to occasionally correct course. It monotonically hurt every horizon it touched, for both RF and Ridge, at this data scale — evidence against assuming V1's known "last-known-history" limitation (ML Spec §1.2/§11.1) is actually costing accuracy in practice.
- **Ridge essentially reproducing Naive's +1 behavior** exactly, rather than under- or over-performing it, was a specific, quantifiable coincidence worth flagging for ML-C rather than glossing as "roughly similar."

## 30. Runtime/complexity/maintainability observations

- K-Means and the two supervised text candidates all fit in well under a second on this data scale; no runtime concern at any evaluated size.
- RF/Ridge per-fold fitting (14 folds × 2 models) completed in a few seconds total — the interactive `ForecastService.run_forecast()` contract (a single fresh fit per user action, TRD §12.3) is not remotely stressed by either candidate at this data volume.
- The recursive strategy adds real implementation complexity (sequential, horizon-ordered prediction with state threaded between steps) for a result that was *worse*, not better, than the simpler last-known-history strategy already shipped — at this data scale, that complexity is not currently earning its keep either.

## 31. Limitations of this evaluation

- **Tier B, single-author benchmark, 228 rows.** Every categorization number is bounded by this. A larger, multi-annotator, more heavily-trafficked-merchant-realistic benchmark would materially change confidence in these results (though probably not the qualitative "leakage matters enormously" finding).
- **Small VALIDATION size (50 rows) for structured error analysis** — several §8-required phenomena (multi-purpose merchant, refund/credit, malformed description) are present in the benchmark by design but did not happen to surface as *misclassifications* in this specific 50-row VALIDATION draw; their presence is confirmed in the data, their error-rate effect is not measured this run.
- **24 months of synthetic-only forecasting history** limits the history-length experiment to 3 usable origins (§25) and the calendar-boundary VALIDATION loop to 14 folds — enough for a first read, not enough for tight confidence intervals on any WAPE difference.
- **"Other" is a degenerate all-zero series** in this specific run because of how the production K-Means model happens to cluster this particular synthetic dataset — this is a fact about this evaluation's specific inputs, not a general property of the "Other" category.
- **Forecasting evidence tier is synthetic throughout** (§3) — every forecasting number in this report describes pipeline/mechanism behavior, never real-user spending forecast accuracy.

## 32. What additional evidence would strengthen the conclusions

- A larger Tier B benchmark (or, ideally, Tier A real de-identified data) with more merchant groups per category, to test whether the "generic vocabulary generalizes, proper-noun vocabulary doesn't" pattern (§15) holds at scale or is an artifact of this benchmark's small size.
- A second, independent annotator for the Tier B labels, to remove the single-author limitation (§31).
- More than 24 months of forecasting history (real or a longer synthetic generation run) to widen the history-length experiment beyond 3 origins and produce tighter VALIDATION fold counts.
- A version of the Tier B benchmark or a real dataset large enough that "Other"/low-frequency categories aren't at risk of ending up entirely zero-spend or single-digit-support by chance.

## 33. Questions ML-C must decide

- Given K-Means's near-chance VALIDATION performance under a leakage-safe evaluation, does *any* candidate in the frozen set clear a bar worth shipping, or does this argue for revisiting the categorization approach entirely (a question ML-C, not ML-B, is scoped to raise, per ML Spec §0/§14's "the current baseline is not a predetermined winner")?
- Given Naive beats both RF and Ridge combined, and Ridge beats RF specifically, does the forecaster warrant staying with Random Forest, moving to Ridge, or moving to Naive — including the explicit possibility ML Spec §14 names outright: shipping Naive itself?
- Does the evidence in §22 (Ridge's +1-only strength masking +2/+3 weakness relative to Naive) change how "beats naive" should be interpreted for any candidate?
- Is the recursive strategy worth retaining as a documented-but-unshipped alternative, or is last-known-history's continued use now *evidence-backed* rather than merely inherited from V1?
- Does the Tier B benchmark's scale/single-author limitation (§31) block a categorization decision until better data exists, or is TRAIN+VALIDATION evidence at this scale sufficient for ML-C's purposes?

## 34. ML-B does not select the production winner

Every comparison above is stated as an observation ("Candidate X achieved lower VALIDATION WAPE than Candidate Y"), never a directive ("therefore ship X"). No production code, model artifact, or configuration was changed as a result of any number in this report. `models/kmeans_model.pkl` and `models/rf_model.pkl` are untouched; `CategorizationService`/`ForecastService` behavior is identical to `bc27353`.

## 35. The FINAL UNTOUCHED TEST remains unevaluated

No categorization metric was computed against the 45 FINAL_TEST rows / 17 merchant groups. No forecasting metric was computed against 2024-10/11/12. Both remain sealed, per §7. ML-C's eventual candidate selection, followed by ML Spec §20's single final pass, are the only steps in the frozen process permitted to open either.

---

# Interview & Deep-Dive Notes

This section is a working record for later study and interview preparation — not a certified claim document (that's ML-E's job) and not a scripted Q&A. It preserves *why* each choice was made, what the math is doing, what actually happened on VALIDATION, and what an interviewer could reasonably push back on.

## Categorization

### K-Means — objective/intuitive math
K-Means minimizes within-cluster sum-of-squared distances to a centroid: for `k` clusters, find centroids `μ₁...μₖ` minimizing `Σᵢ ‖xᵢ − μ_{c(i)}‖²` where `c(i)` is point `i`'s assigned cluster. It has **no concept of "Shopping" or "Healthcare"** — it only groups points that are geometrically close in the 53-dimensional feature space (amount, TF-IDF merchant terms, day-of-week, weekend flag). The semantic label is bolted on afterward by majority vote. **Why we tested it:** it's the current production baseline (ML Spec §0) — must be beaten or matched with justification, not assumed inferior. **What leakage would have looked like:** fitting the scaler/vectorizer/KMeans model, or building the cluster→category mapping, using any VALIDATION or FINAL_TEST row — this would let the model "see" the exact distribution it's later scored on. **What we found:** near-chance VALIDATION accuracy (12.0%) once leakage was removed, vs. 42.1% on its own TRAIN diagnostic — a ~30-point train/validation gap is a textbook generalization-failure signature. **What an interviewer could challenge:** "Is 12 clusters for 8 categories the right choice?" — Answer: it's V1's original choice, reused deliberately (not re-tuned) so the comparison isolates the *evaluation methodology* fix, not a simultaneous hyperparameter change; a follow-up experiment varying `n_clusters` would need its own VALIDATION-only tuning pass, not done here. **What we still cannot claim:** that K-Means is inherently worse than the supervised candidates in general — only that, on this specific 228-row Tier B benchmark, under a correct merchant-grouped split, it did not clear chance level.

### TF-IDF — intuition
Term Frequency × Inverse Document Frequency: a word's weight in a document is high when it appears often *in that document* but rarely *across all documents*. This is why "the"/"inc"/"store" get down-weighted and a distinctive merchant word gets up-weighted — but critically, **IDF is a property of the fitted TRAIN corpus**, so a word absent from every TRAIN document (any brand-new merchant name) contributes nothing at inference time. That's exactly the mechanism behind §15's core finding.

### Logistic Regression — classification intuition
Learns one linear decision boundary per category (one-vs-rest under the hood for multiclass) over the TF-IDF feature space, squashed through a softmax/sigmoid to produce a probability per category; predicts the highest-probability category. **What it did well on VALIDATION:** Rent & Utilities and Subscriptions — both are categories where TRAIN examples share *generic* recurring words ("bill," "monthly," "membership") that also appear on genuinely new merchants of the same type. **What it did poorly:** Healthcare, Shopping — 0% recall on both; the model essentially never had a discriminating generic word to hang a Healthcare/Shopping prediction on, and defaulted to the TRAIN-majority class (Food & Dining) instead. **Representative failure:** `ESCAPE ROOM EXPERIENCE` → predicted Food & Dining, purely because "escape"/"room"/"experience" are absent from TRAIN's vocabulary and the model's intercept/prior favors the majority class.

### Linear SVM / margin intuition
Finds the hyperplane per class boundary that maximizes the margin (distance) to the nearest training points of each class, rather than modeling a probability directly. In sparse, high-dimensional TF-IDF space this tends to be robust to the "curse of dimensionality" problems that hurt distance-based methods like KNN. **Surprising result:** SVM (26.0% VALIDATION accuracy) landed *between* K-Means and Logistic Regression, not clearly ahead of LogReg as textbook framing might suggest for sparse text — plausible explanation: at this data scale (133 TRAIN rows), the theoretical advantage of margin maximization over probabilistic fitting is swamped by simply not having enough data for either to learn robust decision boundaries for the categories with proper-noun-dependent vocabulary.

### Macro F1 vs. accuracy
Accuracy = fraction correct, weighted implicitly by how common each category is. Macro F1 = the *unweighted* average of F1 across all 8 categories, so a model that nails the frequent categories and completely misses "Healthcare"/"Other" gets penalized in macro F1 even if its accuracy looks fine. Both metrics were reported here (K-Means 12.0% accuracy / 0.0566 macro F1 — F1 is proportionally *lower* than accuracy, correctly reflecting that its correct predictions are concentrated in a couple of categories rather than spread across all 8).

### Merchant leakage — the core methodological lesson
If "STARBUCKS COFFEE #0442" and "STARBUCKS COFFEE #1187" can land in different partitions, a model can score well by recognizing the string "STARBUCKS" specifically, without learning anything that transfers to a brand-new coffee shop. The fix (merchant-grouped splitting) doesn't make the *model* worse — it makes the *measurement* honest. The ~50-80-point accuracy drop we measured is direct, quantified evidence of how large that honesty gap can be at small data scale.

### Cluster→semantic-category mapping — why it's a separate leakage surface
K-Means's "fit" step never sees a label at all — that's what makes it unsupervised. But converting cluster IDs into "Shopping"/"Healthcare" *does* use labels (majority vote), and that step must obey the same TRAIN-only rule as any other label-consuming step, or the two-stage structure (unsupervised fit, then supervised-style mapping) becomes a hidden way to leak VALIDATION/TEST information back into the "trained" object.

## Forecasting

### Naive forecast
Predicts next month = last observed month. It's not "no model" — it's the sharpest possible null hypothesis: if a fancier model can't beat "assume nothing changes," the fancier model isn't adding value. **What we found:** it wasn't beaten, by either RF or Ridge, at this data scale.

### Seasonal naive
Predicts next month = same calendar month, one year ago. Tests whether a model needs to learn seasonality at all or whether "just look up last December" already captures most of it. Its 12–13-month eligibility floor is a real constraint, not an implementation inconvenience — with only 24 months of history, plenty of early folds simply can't produce a seasonal-naive prediction, which is why its evaluated sample (216 rows) is smaller than every other candidate's (312).

### Random Forest regression
An ensemble of decision trees, each trained on a bootstrap resample with random feature subsets at each split, predictions averaged. Captures non-linear interactions (e.g., "December AND Entertainment" behaving differently from either alone) that a linear model can't. **What it did well:** improved steadily with more history (§25, up to 18 months). **What it did poorly:** lost to Naive combined and at every horizon — with only a few dozen TRAIN rows per fold, RF's capacity to model non-linear interactions likely isn't earning its keep against simpler alternatives yet. **Complexity vs. performance tradeoff:** RF is the most implementation-complex forecasting candidate evaluated (ensemble, more hyperparameters, longer fit time) and it did not win on the primary metric at this data scale — a textbook example of ML Spec §0's principle that complexity isn't a success criterion by itself.

### Ridge regression
Linear regression with an L2 penalty (`minimize Σ(y − Xβ)² + α‖β‖²`) shrinking coefficients toward zero, trading a little bias for less variance — helpful with few training examples relative to feature count, which is exactly this project's regime (single-digit-to-low-dozens of TRAIN rows per fold). **Surprising result:** its +1 predictions were numerically identical to Naive's, suggesting the fitted model effectively collapsed onto "coefficient on `lag_1_spend` ≈ 1, everything else ≈ 0" — plausible with this little data, and worth a follow-up look at Ridge's actual fitted coefficients in ML-C if Ridge is a serious contender.

### Last-known-history multi-step strategy
Same real-history-derived features reused for +1/+2/+3; only calendar fields vary. Simple, and — per this data — not obviously worse than the alternative that's supposed to fix its "known limitation."

### Recursive multi-step strategy
Feed each prediction back in as if it were an observation for the next horizon. Intuitively appealing (uses the best information available at each step) but mechanically compounds any single-step error forward — exactly what we measured (§22: monotonically worse than last-known-history at +2 and +3, for both RF and Ridge).

### Direct multi-step strategy
Not evaluated this pass (§20) — reserved for future work if A/B evidence ever shows a genuine horizon-specific pattern that a shared model/strategy can't capture. Current A/B evidence shows a *smooth, monotonic* degradation with horizon for the recursive strategy specifically, not a sharp regime change that would motivate per-horizon specialized models.

### Expanding-window validation
Train only on the past, test on the immediate future, then fold that future month into TRAIN and move the window forward one calendar month. Never a fixed-size sliding window (which would arbitrarily forget older history) and never row-count-based splitting (which can misalign with calendar boundaries when multiple categories share a month, V1's exact GridSearchCV flaw, ML Spec §12).

### WAPE vs. MAE vs. RMSE vs. MAPE
- **WAPE** — aggregate dollar error over aggregate dollar actual; primary, because it isn't distorted by any single near-zero actual (§23).
- **MAE** — average absolute dollar error; same units a person intuitively understands, secondary.
- **RMSE** — like MAE but squares errors first, so it penalizes large misses disproportionately; useful for "does this candidate occasionally blow up badly," diagnostic/sensitivity only here.
- **MAPE** — average of *per-row* percentage errors; primary failure mode is a near-zero actual producing an enormous, meaningless percentage that then gets averaged in unweighted (§24) — retained only for compatibility with V1's reporting style, always paired with the near-zero-actual count.

### +1/+2/+3 horizons
The product displays all three from day one (PRD §11.8), so a candidate that looks good only at +1 is shipping two-thirds of an unvalidated feature. §22's finding (Ridge tying Naive at +1 but losing at +2/+3) is the concrete case this requirement exists to catch.

### History-length sensitivity
More history should help a model that's actually learning temporal structure. It did, for RF, monotonically from 9→18 months (with an anomalous dip at 9 months) — but the *sample* behind that finding is only 3 origins, disclosed explicitly rather than presented with more confidence than 3 data points support.

### Sparse/zero-spend series behavior
A category that is genuinely always zero in the evaluated window (here, "Other," a fact about this specific K-Means-labeled synthetic run, not a property of the category name in general) makes WAPE mathematically undefined and MAE trivially small — a naive read of "MAE=0.00, this candidate is great at Other!" would be exactly backward. This is the concrete reason ML Spec §16 exists.

---

*Reproduce every number in this report with `python -m ml.run_all` from a clean checkout at the commit named at the top of this file, using `requirements.txt`'s pinned environment. `reports/ml/results/experiment_log.jsonl` records every experiment run, including the two near-miss vocabulary conflicts caught and fixed before any evaluation numbers were produced (§28).*
