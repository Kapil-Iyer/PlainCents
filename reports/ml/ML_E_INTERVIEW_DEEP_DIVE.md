# ML-E: Interview / Deep-Dive Record

Factual, implementation-grounded explanations for technical discussion of this project. No fabricated answers — every claim here traces to a specific file/report cited inline. Cross-reference `ML_E_CLAIM_MATRIX.json` before repeating any number aloud.

## Categorization

**What V1 K-Means actually did.** `pipeline/cluster.py::fit_and_evaluate` builds a feature matrix (TF-IDF over merchant text + scaled amount + day-of-week, via `pipeline/features.py::build_feature_matrix`), fits `KMeans(n_clusters=12)`, then assigns each of the 12 clusters to one of 8 semantic categories by majority vote against keyword-derived labels.

**How unlabeled clusters became semantic categories.** K-Means itself has no notion of "Food & Dining" — it only groups similar feature vectors. The cluster→category mapping is a post-hoc majority vote: for each cluster, count which heuristic label appears most often among its members, assign that label to the whole cluster. Any transaction landing in that cluster inherits the cluster's assigned label, right or wrong.

**Why V1's synthetic diagnostic looked strong.** The keyword-derived "true" labels used to both build synthetic merchant vocabulary and score the model were produced by matching the same surface merchant text the TF-IDF features already see (`pipeline/cluster.py::_get_true_labels`). A model can score well on this kind of circular ground truth without having learned anything a human would recognize as "categorization judgment" — it can win by re-detecting the very keywords used to author the labels.

**Why that did not establish generalization.** No merchant-level held-out test existed — V1's own 160/40 split was drawn from the same distribution, potentially the same or highly similar merchant strings in both halves. A model can look excellent on data drawn from its own training distribution and still fail on any genuinely new input.

**What merchant leakage is.** If "STARBUCKS #4521" appears in both the training set and the test set, a model can memorize that specific string→category mapping rather than learning transferable text patterns — inflating the test score in a way that doesn't predict real-world performance on merchants never seen before.

**How merchant-group splitting helps.** `ml/common/splitting.py::merchant_grouped_stratified_split` assigns every row sharing a `merchant_group` identifier to exactly one of TRAIN/VALIDATION/FINAL_TEST — never split across partitions. A model can only succeed on VALIDATION/FINAL_TEST by generalizing to merchants it has never seen fit, which is exactly the property a shipped categorizer needs.

**What TF-IDF does.** Term Frequency-Inverse Document Frequency converts free text into a numeric vector: each dimension is a token (or token pair, here `ngram_range=(1,2)`), weighted by how often it appears in this document (term frequency) discounted by how common it is across all documents (inverse document frequency) — so a distinctive word like "NETFLIX" gets more weight than a common one. `sublinear_tf=True` applies a log dampening to raw counts so a word appearing 10 times doesn't get 10x the weight of appearing once.

**What a TF-IDF vector looks like conceptually.** For "TIM HORTONS COFFEE" with a 50-dimension vocabulary (`max_features=50`), the vector is mostly zeros, with small positive weights at the dimensions corresponding to tokens like "tim," "hortons," "coffee," and any 2-grams like "tim hortons" that made the vocabulary cut.

**How Logistic Regression works for multiclass text classification.** For each category, LogReg learns a weight vector over the TF-IDF dimensions; the weighted sum (a linear combination) passed through a softmax-like decision produces a probability per category, and the highest-probability category is predicted. It is a linear model — no cluster geometry, no majority-vote indirection; it directly learns which token weights predict which category from labeled examples.

**Why Logistic Regression won this evaluation.** On VALIDATION (`reports/ml/ML_B_EXPERIMENT_REPORT.md` §14): LogReg 0.2552 macro F1 / 32.0% accuracy vs. K-Means's 0.0566 / 12.0% (statistically indistinguishable from ~12.5% random chance for 8 categories) and Linear SVM's 0.2405 / 26.0%. LogReg directly optimizes for the classification objective on the actual labeled task, rather than relying on unsupervised clustering plus a majority-vote post-hoc mapping (K-Means) that has no mechanism to correct a cluster containing a mix of categories.

**Why Linear SVM was close but rejected.** Both are linear models over the same TF-IDF features and scored similarly (0.2405 vs 0.2552 macro F1) — LogReg's margin was modest but consistent, and the frozen spec's rule is to select the higher-scoring candidate on VALIDATION, not to re-litigate a close call.

**Why K-Means collapsed under stricter evaluation.** Once evaluated in isolation (scaler/vectorizer/KMeans/mapping all fit on TRAIN only, no leakage from the mapping step) and merchant-group-separated, K-Means's 12.0% VALIDATION accuracy is essentially random guessing for 8 categories — the unsupervised clustering geometry does not align with the true category boundaries once the model can't lean on same-distribution leakage.

**What accuracy means.** Fraction of predictions exactly matching the true label. Simple, but misleading under class imbalance — a model that always predicts the majority class can still score high accuracy while being useless for minority categories.

**What macro F1 means.** F1 (harmonic mean of precision and recall) computed separately per category, then averaged unweighted across categories — so a category with only 4 support rows counts equally to one with 46. This is why it was chosen as primary (`docs/V2_ML_SPEC.md` §7): it doesn't let strong performance on frequent categories (like Food & Dining) mask poor performance on rare ones (like Healthcare).

**Why macro F1 was primary.** The Tier B benchmark's category distribution is imbalanced (20-46 rows per category) — accuracy alone could look good while silently failing several categories.

**What the final 42.2% accuracy does and does not mean.** It does mean: on a 45-row held-out slice of an independently curated benchmark, with merchant groups never seen during fitting, this specific model correctly predicted the category 42.2% of the time — well above the ~12.5% random-chance floor for 8 categories. It does not mean: this is the accuracy to expect on real bank transactions, nor a stable population estimate (n=45 is small), nor an improvement over V1's 90% figure (different, non-comparable evidence tiers — see the claim matrix).

**Why FINAL > VALIDATION does not imply the final number is "truer."** FINAL_TEST scored higher (0.4405 macro F1) than VALIDATION (0.2552) for the same model — this is disclosed in `ML_C_EXPERIMENT_REPORT.md` as small-sample volatility (45 and 50 rows respectively, easily swung by a few examples), not evidence that FINAL_TEST is an inherently more representative sample. Neither number was used to revise the other; the selection was frozen using VALIDATION alone, before FINAL was ever consulted.

**Why more real labeled data is the next scientific need.** Both the small-sample volatility above and the persistent gap between Tier B and real-world data point the same direction: a larger, and eventually real (Tier A), labeled dataset is the highest-value next investment, not further tuning of the current linear model on the same 228-row benchmark.

## Forecasting

**What V1 Random Forest did.** `pipeline/forecast.py::fit_and_forecast` builds 8 engineered features per (month, category) — rolling 3/6-month averages, rolling std, lag-1, calendar flags — then fits a single `RandomForestRegressor` via expanding-window walk-forward validation, with conditional `GridSearchCV` if MAPE exceeded 15%.

**Why time series cannot use ordinary random splits.** A random train/test split lets the model see future months while "training" to predict earlier ones — the model would be allowed information (later economic conditions, seasonal patterns from months chronologically after the test point) that would never be available at real prediction time. This silently inflates the measured score.

**Expanding-window validation.** Start with a minimum history window; at each step, train on everything up to the current origin month, predict the next 1-3 months, score, then advance the origin forward by one month and repeat — the training window only ever grows, mirroring how a real deployed forecaster accumulates history over time.

**+1/+2/+3 horizons.** Three separate forecasts per category per run: next month, the month after, and the month after that — giving the user a short 3-month outlook rather than a single point estimate.

**Naive forecast definition.** `naive_predict(spend_history) -> spend_history[-1]` — predict that next month's spend equals the most recently observed month's actual spend. No fitting, no parameters.

**Seasonal Naive definition.** Predict that the target month's spend equals the actual spend from the same calendar month one year prior (`seasonal_naive_predict`, requires ≥13 months of history to be eligible; returns `None` rather than a fabricated guess if not).

**Ridge.** Linear regression with L2 regularization (a penalty on large coefficients) over the same 8 engineered features RF uses — simpler and less prone to overfitting than RF, but still a fitted, parametric model unlike Naive.

**RF (in the ML-C bake-off).** Same `RandomForestRegressor` architecture as V1, refit at each expanding-window origin, evaluated under both multi-step strategies below.

**Last-known-history strategy.** For all three horizons, the rolling/lag features are computed once from the most recently observed actual data — only the calendar-derived features (month number, is-December, is-summer) vary by horizon. The model's numeric inputs don't compound error across horizons.

**Recursive strategy.** The model's own prediction for +1 becomes part of the input history used to predict +2, and so on — errors from earlier horizons can compound into later ones.

**Recursive error compounding.** `reports/ml/ML_C_FOLD_STABILITY.json` shows both RF and Ridge's recursive strategy underperforming last-known-history at +2/+3 — consistent with the mechanism above: a prediction error at +1 feeds forward and amplifies at later horizons.

**WAPE.** Weighted Absolute Percentage Error = sum(|actual − predicted|) / sum(|actual|), aggregated across all predictions rather than averaged per-row — robust to near-zero actuals that would otherwise blow up a per-row percentage metric.

**MAE.** Mean Absolute Error — average absolute dollar error, in the same units as spend, easy to communicate but not scale-normalized across categories with very different typical spend levels.

**RMSE.** Root Mean Squared Error — like MAE but squares errors before averaging, penalizing large individual misses more heavily.

**MAPE zero-spend problems.** Per-row percentage error divides by the actual value — when a category has near-zero actual spend in a given month, even a small absolute error produces an enormous percentage error. `final_forecasting.json`'s `mape_all` (485,833,333,346%) versus `mape_nonzero_only` (15.8%) is a direct, observed illustration of this — the near-zero-actual "Other" category month distorted the naive aggregate MAPE into a nonsensical number, which is exactly why WAPE was chosen as the primary metric instead.

**Why Naive beat RF/Ridge here.** Across 14 expanding-window origins, Naive's mean WAPE (0.191) beat every RF/Ridge variant (0.235-0.257), and RF/Ridge's win rate against Naive never exceeded 43% of origins in either strategy (`ML_C_FOLD_STABILITY.json`). With this synthetic dataset's structure, the added model complexity of RF/Ridge's engineered features didn't translate into a reliable accuracy edge over simply repeating the last observed value.

**Why selecting a simple baseline is good engineering.** The frozen spec's §14 rule makes this explicit: ship the simplest model that the evidence actually supports. Shipping RF here would mean deploying more complexity, more moving parts, and a persisted-artifact concept, for no measured accuracy benefit and worse interpretability — that tradeoff has no justification once the evidence says the simple baseline wins.

**Why strategy is N/A for the selected Naive model.** Naive's prediction never depends on its own prior output (there's no recursion to speak of) and never recomputes anything per horizon — "last-known-history" and "recursive" are not meaningfully different concepts for a model this simple, so the strategy field is not a fitted choice, it's a structural non-applicability.

**Why forecasting evidence remains synthetic.** No real user transaction history has accumulated in this product yet (it is a local-first, individually-run app) — `data/raw/synthetic_24mo.csv` is the only dataset with enough months and category coverage to exercise the forecasting mechanism end-to-end. This is disclosed, not hidden, in every forecasting result artifact's `evidence_tier` field.

## Systems

**Where model loading happens.** `backend/main.py`'s `lifespan` hook, once at FastAPI startup: `CategorizationService(LOGREG_MODEL_PATH)`, stored on `app.state.categorization_service`.

**Where inference happens.** `CategorizationService.predict()`/`predict_batch()` — `self._vectorizer.transform(...)` then `self._model.predict(...)`, called from `TransactionService.create_manual()` and `IngestionService.parse_and_stage()`/`commit_import()`.

**How FastAPI reaches the service.** Dependency injection via `backend/api/deps.py::get_categorization_service`, which reads `request.app.state.categorization_service` — the same singleton instance loaded at startup, never reconstructed per-request.

**Why model is not trained per request.** Fitting is expensive and, more importantly, would make every user's inference non-deterministic and dependent on whatever data happened to be in the DB at request time — the selected model is a fixed, evaluated artifact; training it on the fly would silently create a different, unevaluated model on every request.

**Predicted vs confirmed vs effective category.** `predicted_category` (immutable, written once at insert time) records what the model said; `confirmed_category` (nullable) records a user's explicit correction; `effective_category = COALESCE(confirmed_category, predicted_category)` (a SQL view column) is what the rest of the app (dashboard, forecasting) actually reads.

**How corrections work.** `TransactionService.update()` allows `confirmed_category` as an updatable field via `TransactionRepository.update()`'s `allowed` set — `predicted_category` is not in that set, so it can never be overwritten by a correction.

**Why corrections do not retrain.** There is no code path anywhere that feeds a correction back into `CategorizationService`'s model or artifact — a correction is a pure data write. `tests/backend/services/test_categorization_service.py::test_predict_never_fits_the_underlying_model`/`test_predict_batch_never_fits_the_underlying_model` assert this directly by mocking `.fit()`/`.fit_transform()` and checking they're never called during inference.

**How forecasts are persisted.** `ForecastService.run_forecast()` writes one `forecast_runs` row plus N `forecast_predictions` rows inside a single unit-of-work transaction (`backend/db/unit_of_work.py`) — both succeed or neither does.

**Why GET never trains.** `ForecastService.check_status()`/`get_latest()` only ever read via `TransactionRepository`/`ForecastRepository` — neither imports nor calls `pipeline.forecast.train_and_predict`, verified directly by `test_check_status_never_fits`/`test_get_latest_never_fits`.

**What remains evaluation-only code.** Everything under `ml/` (candidate definitions, bake-off runners, metrics, splitting), plus V1's own `pipeline/forecast.py::fit_and_forecast`/`walk_forward_validate` and `pipeline/cluster.py::fit_and_evaluate` — none of these are called from `backend/`.

**What Naive/RF-swap actually changed.** `pipeline/forecast.py::train_and_predict`'s function body — same signature, same DataFrame output shape (`category, month_offset, forecast_month, predicted_amount, is_available, unavailable_reason`), so `ForecastService` needed zero changes beyond the `model_impl_version` string.

## Claims

**What can be said on a resume.** "Built an evidence-based ML evaluation pipeline for transaction categorization and spending forecasting: merchant-grouped/temporal leakage-safe splits, benchmarked multiple candidates per problem, selected models via held-out validation, and integrated the selected implementations into a production FastAPI service with artifact versioning and a documented correction workflow." True, verifiable, and doesn't lean on any single accuracy number.

**What requires qualification.** Any specific number (42.2% accuracy, 0.4405 macro F1, 18.9% WAPE) — always paired with its evidence tier (Tier B curated benchmark; synthetic temporal test) in the same sentence, per `ML_E_CLAIM_MATRIX.json`.

**What must never be claimed.** Real-world bank-transaction accuracy at these numbers; real-world spending-forecast accuracy at 18.9% WAPE; a V1-to-V2 accuracy "improvement" (non-comparable evidence tiers); real TD-export verification; automatic retraining/online learning (there is none — verified by direct test).
