# PlainCents V2 — ML Specification

**Status:** FROZEN — V2 ML Specification
**Traces to:** `docs/V2_PRD.md` (FROZEN), `docs/V2_TRD.md` (FROZEN), `docs/V2_PREBUILD_REALITY_CHECK.md`
**Scope:** How PlainCents V2 scientifically evaluates, compares, selects, versions, and integrates its ML components. This document does not implement, train, or modify any model. It does not redesign the TRD's service boundaries (`CategorizationService`, `ForecastService`) — it defines how the implementations *behind* those boundaries earn the right to be called final.

---

## 0. Non-Negotiable ML Principle

PlainCents V1's **K-Means categorizer** and **Random Forest forecaster** are **initial reusable implementations and baselines**, not predetermined V2 winners. The frozen TRD deliberately made both replaceable behind stable contracts (`predict(transaction) -> {predicted_category}`; `check_status()/get_latest()/run_forecast()`) for exactly this reason. This document defines the evidence a replacement — or a decision to keep the current implementation — must produce.

### Dual-track development

- **App Track:** V2 integrates the current K-Means + RF implementations behind the TRD's service boundaries immediately, so FastAPI/React development is not blocked (TRD §7.3/§7.4, §12.1–§12.3).
- **ML Track:** In parallel — data preparation → corrected evaluation → baseline/candidate experiments → error analysis → model selection → artifact versioning → final integration.

**PlainCents V2 is not scientifically final until the acceptance gates in §20 pass**, even if the application is fully functional and demonstrable on the initial implementations.

---

## 1. V1 ML Baseline Audit

### 1.1 Categorization (inspected: `pipeline/features.py`, `pipeline/cluster.py`)

- **Features** (`features.py:11-68`): `amount` (StandardScaler, weighted ×0.2), merchant text (TF-IDF, `max_features=50`, `token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"`, `ngram_range=(1,2)`, `sublinear_tf=True`, then L2-normalized, unweighted), `day_of_week` (weighted ×0.1), `is_weekend` (weighted ×0.1). Final feature vector width: `1 (amount) + 50 (TF-IDF) + 1 (dow) + 1 (weekend) = 53` dimensions.
- **Model** (`cluster.py:112-113`): `KMeans(n_clusters=12, random_state=42, n_init=50)` — 12 clusters for 8 semantic categories (more clusters than categories, intentionally, so majority-vote mapping has flexibility; PRD/TRD do not require this specific choice to persist).
- **Cluster → category mapping** (`cluster.py:127-135`): after fitting, 200 rows carrying a heuristic label (see below) are shuffled once (`random.seed(random_state)`) and split into 160 "mapping" rows and 40 "held-out" rows. For each of the 12 cluster IDs, the majority heuristic label among that cluster's mapping-set members becomes that cluster's assigned category. A cluster with zero mapping-set members defaults to `CATEGORIES[0]` ("Food & Dining") — an arbitrary fallback, not a modeled decision.
- **Heuristic labels** (`cluster.py:23-84`, `_get_true_labels`): a fixed dictionary of merchant-substring keywords (e.g., `"NETFLIX" → "Entertainment"`) is matched against each transaction's uppercased merchant string; the first matching keyword wins; unmatched merchants default to `"Other"`. **Terminology note, honored per this document's instruction:** these are **keyword-derived (heuristic) labels**, not "pseudo-labels" — they are not model output relabeled as ground truth; they are hand-authored substring rules written by the same person who also wrote the synthetic merchant vocabulary that the rules match against (`scripts/generate_synthetic_24mo.py:18-71` uses near-identical keyword substrings, e.g. `"NETFLIX STREAMING CINEMA"`). This coupling is material to interpreting the resulting accuracy figure (§2).
- **Training path**: `fit_and_evaluate(df)` (`cluster.py:87-152`) requires ≥200 rows, fits K-Means, builds the mapping, evaluates on the 40 held-out rows, computes silhouette on the full `X`, and saves `{kmeans, scaler, vectorizer, cluster_to_category}` via `joblib.dump` to `models/kmeans_model.pkl`.
- **Inference path**: `predict_categories(df)` (`cluster.py:155-173`) reloads the artifact from disk on every call (flagged as a performance concern in TRD §7.3, not an ML-correctness concern), transforms with the saved scaler/vectorizer, predicts cluster IDs, and maps to category via the saved `cluster_to_category` dict.
- **Diagnostic-only metrics**: silhouette score (`cluster.py:140`) and, in the `__main__` block, Adjusted Rand Index against heuristic labels over the full dataset (`cluster.py:189-192`). Both are computed against the *feature space K-Means was fit on* and the *same heuristic labeling rules*, not an independent ground truth.
- **Known limitations** (verified, not assumed): (a) heuristic labels and synthetic merchant vocabulary are authored by the same process — see §2; (b) the arbitrary `CATEGORIES[0]` fallback for empty clusters is a modeling artifact, not a principled default; (c) `tests/test_pipeline.py` exists but is currently **empty** — there is no automated regression protection on categorization behavior today (verified by reading the file).

### 1.2 Forecasting (inspected: `pipeline/forecast.py`)

- **Aggregation** (`forecast.py:22-50`): transactions are summed to `(month, category, total_spend)`; raises `ValueError` if fewer than 12 unique months exist (`forecast.py:46-48`) — this is the source of the TRD/PRD's frozen 12-month MVP rule (TRD §12.5, PRD §21).
- **Features** (`forecast.py:53-138`, `build_forecast_features`): `month_num` (1–12), `category_encoded` (fixed `LabelEncoder` fit on `config.CATEGORIES`), `rolling_3m_avg`, `rolling_6m_avg`, `rolling_std` (3-month, `ddof=1`), `lag_1_spend`, `is_december`, `is_summer` (June/July/August). Rolling/lag values are computed strictly from **prior** months within each category's own time series (`forecast.py:98-114`); rows lacking 6 prior months are dropped (`forecast.py:128`). This construction is **leakage-safe for a single evaluation point** — verified by inspection, not assumed.
- **Walk-forward validation** (`forecast.py:141-249`, `walk_forward_validate`): for each candidate test month (skipping until ≥7 training months exist), a **fresh `RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=42)`** is fit on all data strictly before that month, and a hand-built one-row feature vector (using only training-history rolling/lag values, `forecast.py:198-219`) predicts each category's spend for the test month. APE is computed per `(month, category)`; MAPE is the mean.
- **GridSearchCV** (`forecast.py:281-299`): triggered only if walk-forward `overall_mape > 15`. Grid: `n_estimators ∈ {50,100,200}`, `max_depth ∈ {3,5,10}`, `min_samples_leaf ∈ {3,5,10}` (27 combinations) × `TimeSeriesSplit(n_splits=3)` on `X_all, y_all` — the **full aggregated `(month, category)` row set**, ordered by `(category, month)` after the `sort_values(["category","month_dt"])` in `build_forecast_features` (`forecast.py:86`), not by month alone. **This is the exact V1 weakness this document is instructed to correct (§12):** `TimeSeriesSplit` assumes row order reflects chronological order; here, rows for 8 categories are interleaved/grouped by category first, so a given fold boundary can place, e.g., `(Food & Dining, 2024-06)` in a later fold than `(Transport, 2024-05)` despite April preceding June — later-period information for one category can sit in an earlier fold's "future" relative to another category's real chronology. This does not corrupt `walk_forward_validate` (which is correctly expanding-window by calendar month), only the optional GridSearchCV step.
- **Final fit and forecast generation** (`forecast.py:301-364`): one more RF fit on all aggregated data (post-GridSearchCV params if triggered), then for each of the next 3 months × 8 categories, a hand-built feature row (using the *last available* rolling/lag values from the full series, not test-time-only history) produces a prediction. Model + label encoder saved to `models/rf_model.pkl`.
- **Known limitations** (verified): (a) GridSearchCV fold assignment is not calendar-safe (above); (b) `+2`/`+3` month forecasts reuse the same last-known rolling/lag features as `+1` rather than chaining predictions forward, likely understating multi-month uncertainty (also noted in `docs/TECHNICAL_WALKTHROUGH.md` §8); (c) walk-forward validation and the diagnostic `__main__` comparison (K-Means labels vs. heuristic labels) both run the full pipeline twice with no persisted metrics file — nothing is currently written to disk for reproducibility (§19).

---

## 2. Verified V1 Metrics and Interpretation

Verified against `docs/V2_PREBUILD_REALITY_CHECK.md` and the code inspected in §1. The reality check's numbers are consistent with the code and are recorded as-is; no repository evidence contradicts them.

| Metric | Value | What it actually measures |
|---|---|---|
| Categorization held-out accuracy | 90.0% (36/40) | Fraction of 40 synthetic-data rows (drawn from the same 200-row heuristically-labeled pool used to build the cluster→category mapping) where the majority-vote mapping's category matches the **heuristic keyword label** for that row. |
| Wilson 95% CI on the above | ≈[76.9%, 96.0%] | The statistical uncertainty inherent in evaluating on only 40 rows — a wide interval reflecting small-sample fragility, not a tight real-world bound. |
| Silhouette score | ≈0.5437 | How well-separated the 12 K-Means clusters are in the 53-dimensional feature space. Diagnostic only — says nothing about whether cluster identity aligns with human-meaningful spending categories. |
| ARI (all 779 rows) | ≈0.8073 | Agreement between K-Means cluster assignment (mapped to category) and heuristic labels, over the full dataset rather than just the 40-row eval slice. Same caveat as accuracy: measured against heuristic labels, not independent ground truth. |
| End-to-end forecast MAPE (K-Means-derived categories) | ≈29.4% | Walk-forward MAPE when the forecast pipeline runs on categories assigned by the trained K-Means model — this is V1's actual production-shaped result. |
| Diagnostic MAPE (heuristic-derived categories) | ≈15.7% | Walk-forward MAPE when the same forecast pipeline instead uses the heuristic keyword labels directly as "category," bypassing K-Means entirely. |

**Required interpretive statements** (verified against the coupling described in §1.1/§1.2, stated here exactly as instructed):

- 90.0% is a **controlled synthetic-data diagnostic result**, not real-world TD categorization accuracy.
- The 40 evaluation rows come from the **same synthetic distribution** (same generator, same merchant vocabulary) used throughout development — there is no independent real-world holdout.
- The synthetic merchant vocabulary (`scripts/generate_synthetic_24mo.py`) and the heuristic labeling rules (`cluster.py`'s `MERCHANT_KEYWORDS`) are **authored together** and share substrings deliberately (e.g., both use `"NETFLIX"`), so the categorization task as evaluated is easier than categorizing real, unstructured bank descriptions would be.
- Silhouette measures **feature-space cluster separation**, not semantic category correctness — a well-separated cluster can still map to the wrong human category.
- ARI is a diagnostic against heuristic labels, not proof of real-world generalization.
- 15.7% is **not** the current end-to-end production result — it is a diagnostic that bypasses K-Means entirely.
- 29.4% is the relevant **current end-to-end synthetic result** for the actual production-shaped pipeline (K-Means categories feeding the forecaster).
- The 29.4% vs. 15.7% gap is an **observed difference**, not a validated causal decomposition — it is consistent with the interpretation "categorization noise contributes to forecast error," but no controlled experiment isolates *only* the categorization variable while holding everything else (including which specific rows are misclassified vs. correctly classified in ways that could interact with forecast features) constant. This document does not treat the gap as a proven causal quantity, consistent with `docs/TECHNICAL_WALKTHROUGH.md`'s own framing ("by interpretation," not by a computed decomposition function).
- V1's rolling/lag feature construction (§1.2) is, on inspection, **leakage-safe** for a single walk-forward evaluation point.
- V1's GridSearchCV fold arrangement is **not fully time-series-safe**, for the specific reason given in §1.2 (category-grouped row order defeats `TimeSeriesSplit`'s chronological assumption).

**No resume claim may treat any of the above synthetic diagnostics as independent, real-world evidence** (§21).

---

## 3. Data Strategy

### 3.1 Synthetic data

**Use for:** development, unit/integration tests, demo data (PRD §10a), regression tests, controlled diagnostics, leakage tests, known-seasonality experiments (§17).

**Never use as sole evidence for a final real-world performance claim.**

### 3.2 Independent labeled / realistic evaluation data (categorization)

**Required properties:**
- Transaction descriptions resembling real bank export formatting (abbreviations, inconsistent casing, extraneous reference numbers) — not the clean, keyword-rich synthetic strings in `scripts/generate_synthetic_24mo.py`.
- Human-confirmed category labels against the fixed 8-category MVP taxonomy (PRD §9.4) — not derived from the same keyword rules used anywhere else in this project.
- Independent of the synthetic generator's vocabulary — sharing merchant name substrings with `MERCHANT_KEYWORDS`/the synthetic generator would reintroduce the exact coupling flagged in §2.
- Known provenance and usage rights.
- No accidental contamination between train/validation/test (§6).
- Reasonable representation across all 8 categories, acknowledging "Other" and low-frequency categories (e.g., Healthcare) may be inherently sparse.

**No suitable dataset of this kind currently exists in the repository** — verified: `data/raw/`, `data/processed/`, and `data/exports/` contain only synthetic/generated content per the repository structure inspected for this document, and no independent-evaluation file was found. This document does not fabricate access to one.

**Acquisition protocol (to be executed in ML-B, not now)** — three tiers, distinguished explicitly because they support **different strength claims** (§21):

**Tier A — naturally occurring / real transaction data.** Examples: the product builder's own de-identified real TD transaction history (used internally for evaluation, never committed to the repository or published in raw form — this is the builder's own data used for their own model's evaluation, not third-party data, so PRD §15's privacy principle is not violated by this internal use); or a genuinely real public transaction dataset with suitable provenance and licensing. **If labeled and evaluated correctly under this document's protocols, Tier A data may support an appropriately scoped real-data performance claim** (§21) — this is the only tier that can.

**Tier B — independently curated / constructed benchmark data.** Examples: manually hand-written realistic transaction descriptions with deliberately varied vocabulary, formatting quirks, and ambiguous cases, distinct from `scripts/generate_synthetic_24mo.py`'s vocabulary; or any other deliberately constructed benchmark built independently of the V1 generator and its heuristic labeling rules. This tier is useful for reducing the exact generator/heuristic circularity described in §2, for candidate comparison, and for robustness testing — but **it remains synthetic/constructed data.** It **must not** be described as real-world TD accuracy, real-bank performance, or naturally-occurring-transaction performance under any framing. Any resume/interview claim resting on Tier B data must explicitly describe it as a curated/constructed benchmark, not as real-world evidence.

**Tier C — public datasets, evaluated on their own merits, never assumed real by default.** A public dataset supports a Tier-A-style real-data claim **only if** it is genuinely naturally-occurring data (not itself synthetically generated) and its documented provenance/licensing actually support that characterization — "publicly available" is not itself evidence of "naturally occurring." No such dataset is assumed to exist or to qualify by this document; identifying and verifying one is future ML-B work, not a completed step here, and this document does not fabricate access to any dataset.

Whichever tier ultimately supplies the evaluation data, the resulting dataset is versioned (§18), its tier (A/B/C) is recorded alongside the version identifier so every later report/claim can trace back to which tier of evidence it rests on, and its final test partition is never re-consulted during iterative model development (§6).

### 3.3 User application data

Real TD transactions imported into the running V2 application are **inference data**, not automatically training data. `confirmed_category` corrections a user makes are potential **candidate labeled examples** for a future **offline** retraining dataset only through the controlled process in §9 — never consumed automatically, never causing online/per-correction retraining (PRD §9.3, TRD constraint #6 context).

---

## 4. Categorization Task Definition

**Task:** given one transaction, assign exactly one category from the fixed 8-category MVP taxonomy (PRD §9.4): Food & Dining, Transport, Rent & Utilities, Entertainment, Healthcare, Shopping, Subscriptions, Other. This taxonomy is **not altered** by this document.

**Input** (where justified by the candidate, §5): merchant/normalized description, amount, date-derived features (day-of-week, weekend flag) — matching what V1's feature pipeline already extracts (§1.1). Raw (pre-clean) description may be evaluated as an additional signal for a text-based candidate if it preserves information the normalized `merchant` field discards (V1's `ingest.py:133-136` normalization strips punctuation and casing, which could remove weak signal — an empirical question for ML-B, not decided here).

**Example training record (independent evaluation data, hypothetical):** `{merchant: "TIM HORTONS #4521 TORONTO ON", amount: 6.75, date: "2026-03-04"} → "Food & Dining"` (human-confirmed).

**Example inference record (production):** `{merchant: "AMZN MKTP CA*AB1CD2EF3", amount: 42.10, date: "2026-09-01"} → predicted_category: "Shopping"` (model output, no human confirmation yet).

**Unseen/ambiguous merchants:** the current K-Means approach handles an unseen merchant gracefully (it still lands in *some* cluster and gets that cluster's mapped category), but with no notion of "I don't know" — every transaction receives a definite category, even a low-confidence one. A future supervised candidate faces the same requirement (the product has no "unclassified" state, PRD §9.3/§11.5) but may additionally expose a confidence score if the candidate produces one (§7) — the product does not currently depend on this (TRD §11.2), so it is optional, not required.

**"Other" treatment:** "Other" is both a legitimate category (ATM withdrawals, bank fees, genuinely miscellaneous spend) and a common fallback for anything unmatched. This creates a systematic asymmetry: a classifier can achieve deceptively good aggregate accuracy while performing poorly specifically on "Other" if it over- or under-predicts this catch-all class. Per-category metrics (§7) exist specifically to catch this.

**Unsupervised vs. supervised, explicitly distinguished:** K-Means (the current baseline) **discovers** unlabeled groupings in feature space; the semantic category names are assigned **afterward** by majority vote against heuristic labels — the clustering itself has no concept of "Shopping" or "Healthcare." A supervised candidate (§5) instead learns a direct mapping from features to the named category during training. This is a meaningful methodological difference this document keeps visible rather than blurring: K-Means's "prediction" is really "look up which category this cluster was assigned," while a supervised model's prediction is a direct function of the input.

---

## 5. Categorization Candidate Set

**Frozen bake-off — exactly three candidates, no more without a future ML-Spec amendment:**

| Candidate | Include? | Reasoning |
|---|---|---|
| **1. Existing K-Means baseline** | **Yes** — always included, it's the current production baseline (§0) | Must be beaten or matched with justification, not merely assumed inferior. Evaluated under the strict TRAIN-only isolation in §6.1. |
| **2. TF-IDF + Logistic Regression** | **Yes** | Supervised, interpretable (coefficients per term/category), natively probabilistic (usable confidence if ever needed), cheap to train/infer, and a natural "next step" from V1's existing TF-IDF feature work — reuses `features.py`'s TF-IDF vectorization conceptually rather than discarding it. |
| **3. TF-IDF + Linear SVM** | **Yes** | A standard strong baseline for sparse TF-IDF text classification, typically outperforms KNN on this kind of feature space (KNN degrades in high-dimensional sparse spaces and has higher inference cost — a concern for the interactive `CategorizationService`, TRD §7.3). |

**Explicitly excluded from the initial bake-off, not from future consideration:**
- **`SGDClassifier`** — may be mentioned as a possible future fallback (e.g., if Linear SVM's training cost becomes a concern at a much larger data scale than this project currently has), but is **not** part of the frozen three-candidate set.
- **KNN** — excluded outright from the initial bake-off for the reasons given for candidate 3 above.
- **Neural networks / Transformers / embeddings / LLM classification** — not justified by dataset size (a personal-finance app's transaction volume, even generously estimated, is small for these approaches), adds dependency/inference-cost weight disproportionate to the gain, and is explicitly excluded by the non-negotiable principle (§0).

**No additional model family is added to this bake-off unless future evidence motivates a new, explicit ML-Spec amendment** — this frozen set is deliberately small and is not expanded informally during ML-B execution.

**Per-candidate evaluation against `CategorizationService`'s contract** (`predict(transaction) -> {predicted_category}`, TRD §7.3/§11.2):

- **K-Means:** fits the contract as-is (already wrapped). No probability/confidence produced (V1 has none, and none is fabricated, TRD §11.2). Training cost: cheap (single `fit_predict` on ≤1000s of rows). Inference cost: cheap once artifact is loaded once (TRD §7.3's fix for the per-call reload).
- **TF-IDF + Logistic Regression:** fits the contract identically — `predict()` returns the class label. Produces genuine class probabilities (`predict_proba`) if a confidence field is ever added to the product (not required now, TRD §11.2). Training cost: cheap. Inference cost: cheap (single sparse matrix-vector product per prediction).
- **Linear SVM / SGDClassifier:** fits the contract identically. `SGDClassifier(loss='hinge')` does not produce calibrated probabilities without extra calibration (`CalibratedClassifierCV`), which is not required (§7). Training/inference cost: cheap, comparable to Logistic Regression.

All three candidates are drop-in compatible with the TRD's model-agnostic boundary — none requires a schema, API, or contract change (TRD §11.4, verified against §5's contract definitions above).

---

## 6. Categorization Data Splitting

**Contamination risks evaluated:**
- **Repeated/near-identical merchant strings:** if the same logical merchant (e.g., "TIM HORTONS #1234" and "TIM HORTONS #5678") appears in both train and test, a model can memorize the merchant rather than generalize — inflating measured accuracy relative to how it would perform on a genuinely new merchant.
- **Synthetic-generator vocabulary:** any evaluation using the existing synthetic dataset inherits the coupling in §2 — this is why §3.2 requires *independent* evaluation data for the metrics that matter for final claims.
- **Heuristic labeling-rule leakage:** if the evaluation labels are themselves derived from the same keyword rules used anywhere in candidate feature engineering, the evaluation is circular. Independent evaluation data's labels must be human-confirmed, not keyword-derived (§3.2).
- **Iterative test-set consultation:** repeatedly checking performance against the same "final" test set while tuning invalidates its role as an unbiased estimate.

**Options considered:**

| Strategy | Verdict |
|---|---|
| Random stratified split | Rejected as the *primary* protocol — does not address merchant-repetition contamination; a near-identical merchant variant could land in both train and test purely by chance. |
| Merchant-grouped split | **Chosen as primary** — see below. |
| Temporal split | Not primary for categorization (categorization, unlike forecasting, is not inherently time-ordered — a merchant seen in January is not meaningfully "the past" relative to one seen in June for the purpose of classifying it). Considered secondary/optional if the independent dataset has a natural time axis worth checking for drift, but not required. |
| Hybrid (grouped + stratified) | **Chosen** — grouped by merchant identity, then stratified by category within the resulting groups where feasible, to keep category representation reasonably balanced across splits despite the grouping constraint. |

**Chosen protocol:** merchant-grouped, category-stratified split. All transactions sharing the same normalized merchant identity are assigned to exactly one of TRAIN / VALIDATION / FINAL TEST — never split across them — so a model cannot succeed merely by recognizing a merchant string it already saw during training. Within that grouping constraint, category balance is preserved as closely as feasible across the three partitions.

- **TRAIN:** used for fitting candidate models.
- **VALIDATION:** used for iterative comparison between candidates (§7) and any hyperparameter choices.
- **FINAL UNTOUCHED TEST:** consulted exactly once, after a candidate is selected using VALIDATION performance, to produce the number that may appear in §21's resume claims. Not used for any decision that could feed back into model or hyperparameter choice.

This mirrors V1's own 160/40 mapping/held-out split *in spirit* (a dedicated untouched evaluation slice) but corrects its main weakness: V1's 40-row held-out set is drawn from the same coupled synthetic/heuristic pool (§2), whereas this protocol requires the untouched test to come from the independent dataset (§3.2) and to be merchant-disjoint from training.

### 6.1 K-Means TRAIN/VALIDATION/TEST isolation (required for a fair bake-off)

V1's own held-out evaluation (§1.1/§2) builds the cluster→category mapping from 160 of the 200 labeled rows and evaluates on the other 40 — but under the *independent* evaluation protocol in this document, that same discipline must be applied with the TRAIN/VALIDATION/TEST partitions defined above, not an ad hoc internal 160/40 split of whatever labeled pool exists. Without this, K-Means would receive a systematically different (and easier) evaluation regime than the supervised candidates, invalidating the comparison in §5/§20.

**TRAIN only:**
- Fit the `StandardScaler` and `TfidfVectorizer` (or their equivalents) used to build K-Means's feature matrix.
- Fit `KMeans` itself.
- Construct the cluster → semantic-category mapping (majority vote) using **TRAIN labels only**.

**VALIDATION:**
- Transform VALIDATION rows using the **already-fitted TRAIN** scaler/vectorizer (no refitting).
- Predict cluster assignments using the **already-fitted TRAIN** K-Means model (no refitting).
- Map clusters to categories using the **already-built TRAIN** cluster→category mapping (no re-derivation from VALIDATION labels).
- VALIDATION labels are used **only** to score the resulting predictions (§7's metrics) and to compare K-Means against Logistic Regression/Linear SVM (§5) — never to adjust the scaler, vectorizer, K-Means fit, or the mapping itself.

**FINAL UNTOUCHED TEST:**
- Transform/predict/map using the exact frozen pipeline selected after VALIDATION comparison (no refitting, no re-mapping).
- FINAL TEST labels are used only for the single final evaluation pass (§20).

**Explicitly prohibited at every stage after TRAIN:** VALIDATION or FINAL TEST labels influencing preprocessing fitting, K-Means fitting, cluster→category mapping construction, feature-selection decisions, or any hyperparameter choice (e.g., `n_clusters`) evaluated against the final test set. This is the same no-leakage discipline §6's split already applies to the supervised candidates, made explicit for K-Means's two-stage (unsupervised fit + supervised-style mapping) structure specifically, since it is the one candidate where "training" secretly has two label-consuming steps (fitting and mapping) rather than one.

---

## 7. Categorization Metrics

**Required:** accuracy, macro F1, per-category precision/recall/F1, confusion matrix.

**Why accuracy alone is insufficient:** with 8 categories of uneven real-world frequency (a personal budget likely has far more Food & Dining transactions than Healthcare ones), a model could achieve high aggregate accuracy while performing poorly on low-frequency categories — exactly the "Other" asymmetry noted in §4. Macro F1 (unweighted average across categories) surfaces this in a way overall accuracy cannot.

**Class imbalance:** expected and not treated as a defect to eliminate at all costs — real spending data *is* imbalanced (PRD §9.4 doesn't require balance) — but it must be visible in reporting (per-category metrics, confusion matrix) rather than hidden behind one aggregate number.

**Confidence/calibration:** evaluated only for candidates that naturally produce probabilities (Logistic Regression; SVM only if calibrated). **Not required** — the current frozen product/API has no confidence field (TRD §6/§11.2) and none is added merely because a candidate happens to support it.

**Primary model-selection metric:** **macro F1 on VALIDATION**, because it weights all 8 categories equally regardless of their real-world frequency, directly addressing the imbalance/`Other` concern above and better reflecting "does this model work across the whole taxonomy" than accuracy.
**Secondary metrics:** overall accuracy (for continuity with V1's reporting style and intuitive communication), per-category F1 (for error analysis, §8), confusion matrix (for error analysis, §8).
**Unacceptable failure condition:** any category with F1 substantially below the others (a specific numeric threshold is not fixed here without evidence — determined during ML-B by inspecting the actual confusion matrix and error analysis, not set arbitrarily in advance) triggers mandatory error analysis (§8) before that candidate can be considered for selection, regardless of its macro F1.

---

## 8. Categorization Error Analysis

**Required manual review categories** (all must be examined, not merely mentioned): unseen merchants, vague/truncated descriptions, a single merchant serving multiple spending purposes (e.g., a big-box store selling both groceries and electronics), Subscriptions vs. Shopping ambiguity, Dining vs. grocery-adjacent merchants, refunds/credits (negative amounts — V1's `ingest.py` does not appear to net debits/credits per the reality check's finding, so this is worth checking against whatever the independent evaluation data actually contains), transfers, category imbalance effects, "Other" specifically, and malformed/low-information descriptions.

**Recording:** each reviewed error is logged with `{merchant/description, amount, true_category, predicted_category, candidate_model, notes}` in a plain evaluation report (§18/§23) — no dedicated tooling is introduced for this.

**Sample size:** all misclassified VALIDATION rows are reviewed if the total is small (consistent with a personal-project data scale); if the independent dataset grows large enough that this becomes impractical, a representative stratified sample (proportional across categories, weighted toward categories with the most errors) of at least 30–50 misclassified rows is reviewed instead — an exact universal number is not fixed here, since it depends on how much independent data ends up being collected (§3.2, an open question until ML-B executes).

**Contamination guard:** error analysis and any resulting feature/model adjustments use only TRAIN/VALIDATION rows. The FINAL UNTOUCHED TEST set is never inspected for error patterns before the single final evaluation pass (§6) — inspecting it earlier to motivate a fix would defeat its purpose as an unbiased estimate.

---

## 9. Human Corrections / Future Learning Loop

The product stores `predicted_category`, `confirmed_category`, `effective_category` (TRD §4.1). The future **offline** loop:

```
user corrections (confirmed_category != predicted_category)
    → candidate labeled examples
    → quality/provenance review (was the correction itself plausible? is the transaction real, not demo data — TRD §4.5 data_mode)
    → versioned addition to the training dataset (§18)
    → offline model (re)training
    → validation on VALIDATION split
    → single evaluation on FINAL UNTOUCHED TEST
    → deploy only if §20's acceptance gates pass again
```

**Explicitly prohibited:**
- Automatic retraining triggered by any single correction (matches TRD §9.3/constraint context — corrections never trigger retraining).
- Online/incremental learning on arbitrary user edits without the controlled process above.
- Silently changing a transaction's stored `predicted_category` after the fact — `predicted_category` remains what the model said *at the time*, permanently (TRD §4.1); a later-retrained model's opinion is a new inference on a new artifact version, not a retroactive edit.
- Evaluating a retrained model on the very corrections it was trained on (this is the same TRAIN/TEST discipline as §6, applied to the correction-derived data specifically).

---

## 10. Forecasting Task Definition

**Prediction unit:** total spend (CAD) for one category, for one calendar month.
**Horizon:** 3 months ahead (`month_offset ∈ {1,2,3}`), matching V1 exactly (`forecast.py:314`) and the frozen PRD/TRD (PRD §9.6, TRD §4.6).
**Aggregation:** sum of transaction amounts grouped by `(month, category)` — verified unchanged from V1's `aggregate_monthly` (§1.2).
**Category input for V2:** forecasting operates on **effective category** (`COALESCE(confirmed_category, predicted_category)`, TRD §4.1/§6), not raw `predicted_category` — this is a V2 product requirement (a user's correction should count toward their real spending history) that V1 has no equivalent of, since V1 has no correction concept at all. This is a genuine, intentional difference from V1's aggregation input, not an oversight.
**Information available at forecast time:** all transaction history strictly before the month being predicted, and — for a walk-forward validation point specifically — nothing from the test month itself or later (§12).
**Prohibited information:** any transaction dated in or after the month being forecast; any rolling/lag feature computed using data from the target month or later; any hyperparameter tuned by observing performance on the target month before predicting it.
**Zero-spend / sparse months:** a category with zero spend in a given month is a valid data point (total_spend = 0), not a missing one, and must not be silently dropped from aggregation — only rows failing the *rolling-window history* requirement are dropped (§1.2's existing `dropna` behavior), which is a data-sufficiency exclusion, not a "no spending happened" exclusion. This distinction matters for §16.

---

## 11. Forecast Baselines and Candidates

**Frozen bake-off — exactly four candidates, no more without a future ML-Spec amendment:**

| Candidate | Include? | Reasoning |
|---|---|---|
| **1. Naive** (next month ≈ latest observed month, i.e., `lag_1_spend`) | **Yes** | The minimum bar any ML model must clear — if RF cannot beat "just use last month's number," RF adds no value (§14). |
| **2. Seasonal naive** (next month ≈ same month, prior year, where ≥13 months of history exist) | **Yes, where data permits** | Personal spending has real seasonality (V1's own `is_december` feature exists because of this); a seasonal-naive baseline tests whether RF's added complexity captures more than "December is expensive" alone. |
| **3. Existing Random Forest** | **Yes** — current baseline (§0), must be beaten or matched with justification | Re-evaluated under the corrected calendar-month-boundary temporal protocol (§12), not simply re-reported from V1's existing (partially flawed) evaluation — and evaluated using the **same hyperparameter configuration the TRD's interactive `train_and_predict()` path actually ships** (fixed defaults `n_estimators=100, max_depth=10, min_samples_leaf=5`, TRD §12.3), not V1 walk-forward's internal diagnostic default (`max_depth=3`, `forecast.py:182-184`) or a GridSearchCV-tuned configuration the product will never run interactively. Evaluating a configuration the product doesn't actually use would make the acceptance gate (§20) meaningless. |
| **4. Ridge Regression** | **Yes** | Cheap, highly interpretable (coefficients per feature), and a meaningful complexity step *between* the naive baselines and RF — exactly the "is the extra complexity earning its keep" comparison point this bake-off needs. |

**Explicitly excluded from the initial bake-off, not from future consideration:**
- **Lasso** — may be mentioned as a future fallback (e.g., if Ridge's dense coefficients turn out to obscure which features matter and sparse feature selection becomes useful), but is **not** part of the frozen four-candidate set.
- **Moving-average baseline** — a simple moving average is effectively what `rolling_3m_avg`/`rolling_6m_avg` already represent as *features*; a standalone moving-average-as-forecast baseline would be highly correlated with the naive/seasonal-naive baselines already included and adds limited additional comparison value for the cost of another full evaluation pass.
- **XGBoost / gradient boosting** — adds a new dependency and materially more hyperparameter surface for a dataset this small (a handful of categories × a few dozen months at most), and RF already represents the "tree ensemble" family — a second tree ensemble candidate would test tuning more than it would test a genuinely different modeling approach.
- **LSTM / Transformer / deep time-series models** — not justified by data volume (a single-user app's monthly-aggregated history is, at most, dozens of rows per category) — explicitly excluded per §0.

**No additional model family is added to this bake-off unless future evidence motivates a new, explicit ML-Spec amendment.** If RF and Ridge both underperform the naive baselines, that is a stronger signal to ship the simpler baseline (§14) than to reach for a heavier model — **RF is not selected merely because V1 already uses it.**

### 11.1 Multi-step (+1/+2/+3) forecasting strategy

V1's current approach for +2/+3 has a known limitation (§1.2, `docs/TECHNICAL_WALKTHROUGH.md` §8): it reuses the **same last-known-history** rolling/lag feature values for all three horizons rather than propagating information forward — the +2 and +3 feature rows are built from identical historical inputs to the +1 row, just with different `month_num`/`is_december`/`is_summer` flags. **This document does not assume that behavior should remain final** merely because it is what V1 does today.

**Candidate multi-step strategies to evaluate in ML-B** (not decided here):

| Strategy | Description |
|---|---|
| **A. Current V1 approach (last-known-history)** | All three horizons' feature rows use the same rolling/lag values computed from the actual observed history available at forecast time; only calendar-derived features (`month_num`, `is_december`, `is_summer`) vary by horizon. |
| **B. Recursive strategy** | The +1 prediction is fed back into the feature construction for +2 (as if it were the new "last known" observation), and the +2 prediction similarly feeds +3 — propagating the model's own predictions forward rather than reusing only real historical values. |
| **C. Direct horizon strategy** | Separate horizon-specific models or feature configurations, each trained specifically to predict its own horizon directly from history, rather than one shared model applied three times. Only pursued if evidence from A/B shows a genuine horizon-specific pattern (e.g., systematically different error structure at +2/+3) that a single shared approach cannot capture — not adopted by default, since it triples the artifact/evaluation surface for a personal-finance dataset's modest scale. |

**Selection rule:** the goal is **not** to adopt the most sophisticated strategy — it is to choose the **simplest strategy that produces defensible +1/+2/+3 performance** under the calendar-month-boundary temporal protocol (§12) and the per-horizon metrics required by §13.1. Strategy A remains a legitimate outcome if it performs adequately; recursive (B) is evaluated specifically to test whether V1's known limitation actually costs accuracy in practice, not adopted on the assumption that it must.

**Binding rule — no post-MVP deferral of the strategy actually shipped:** because the MVP product displays all three horizons from day one (PRD §11.8, TRD §4.6), the multi-step strategy backing the categorizer/forecaster the product actually ships **must** be the same strategy evaluated under §12/§13.1's protocol — it is not acceptable to ship one strategy (e.g., A, because it's already implemented) while treating full +2/+3 scientific evaluation as post-MVP ML work. Phase ML-A (§22) may bootstrap the *application* on V1's current approach so FastAPI/React work is not blocked, but the forecaster **acceptance gate (§20)** is not satisfied until the shipped strategy specifically has been evaluated at all three horizons — bootstrapping on an unevaluated strategy is permitted for app development; calling that strategy scientifically final is not.

---

## 12. Temporal Validation

**Required principles** (all satisfied by the design below): chronological ordering; no future month informs an earlier prediction; expanding-window (not rolling/sliding) evaluation, matching V1's existing walk-forward loop's actual behavior (`forecast.py:166-176`, verified correct); any feature preprocessing (e.g., a scaler, if a candidate needed one — Ridge/Lasso may benefit from feature scaling) fit only on the training portion available at each step; time-aware hyperparameter tuning; a final evaluation period held separate where feasible.

**V1's walk-forward loop itself (`walk_forward_validate`) is correctly expanding-window and calendar-safe** — verified by inspection (§1.2): it iterates `all_months` in sorted chronological order and trains only on months strictly before the current test month, refitting fresh each time. This part of V1 is **not** the flawed piece.

**The flaw is isolated to the optional GridSearchCV path** (§1.2, §2): `TimeSeriesSplit` is applied to `X_all, y_all`, whose row order (from `build_forecast_features`'s `sort_values(["category", "month_dt"])`, `forecast.py:86`) is grouped by category first and month second — so `TimeSeriesSplit`'s assumption "row N is chronologically before row N+1" does not hold across category boundaries.

**Correction, required for ML-B implementation (not implemented here) — explicit calendar-month boundaries only:** merely re-sorting the aggregated `(month, category)` rows by month-first, category-second and continuing to use scikit-learn's row-count-based `TimeSeriesSplit` is **not sufficient** and is **not** an acceptable fix on its own. `TimeSeriesSplit` divides by row *count*, not by calendar boundary — with 8 categories sharing each month, a fold boundary can still land in the middle of a month's 8 rows (e.g., after 5 of that month's 8 category-rows), which means some of that month's information (via any within-month feature dependency, or simply via the fold's training set already containing part of "the future" relative to the excluded categories) sits on both sides of the boundary. Sorting alone cannot guarantee a fold boundary coincides with a month boundary when the split point is chosen by row index.

**The only acceptable mechanism is an explicit expanding-window loop keyed on the calendar month itself**, not row position — the same style already used correctly in `walk_forward_validate` (§1.2), extended to sweep a hyperparameter grid instead of one fixed configuration:

```
TRAIN: all categories, all months through 2025-01
VALIDATE: all category rows for 2025-02

TRAIN: all categories, all months through 2025-02
VALIDATE: all category rows for 2025-03

... continuing forward one calendar month at a time.
```

**No calendar month may appear partly in TRAIN and partly in VALIDATION at any step** — a month's rows (across all 8 categories) move from VALIDATE to TRAIN as a whole unit when the window expands, never split. For each candidate hyperparameter configuration, this loop is run in full and its per-fold metrics are averaged for comparison (§13); this reuses proven-correct logic from the existing `walk_forward_validate` rather than trusting a generic scikit-learn utility to handle a row shape it was not designed for.

**Practical scope for a small-data regime:** given a personal-finance dataset's realistic size (single-digit years × 8 categories = well under a thousand aggregated rows even at 24+ months of history), a full nested cross-validation (outer temporal loop wrapping an inner temporal hyperparameter search) is evaluated but **not required by default** — the corrected single-level expanding-window loop above, run once per hyperparameter configuration, is considered sufficient rigor for this data scale; nested validation is reserved as a fallback only if the simpler protocol's results are ambiguous (e.g., no clear winner and high variance across the walk-forward folds).

### 12.1 Separating candidate/hyperparameter selection from final temporal evaluation

Mirroring the categorization TRAIN/VALIDATION/FINAL-TEST discipline (§6), forecast evaluation should also reserve a later, untouched temporal period for final evaluation where the available history makes this defensible:

- **Earlier history** (e.g., the first N−3 months of available history) → used for the expanding-window candidate comparison and hyperparameter selection described above (§12).
- **Later, untouched temporal period** (e.g., the most recent 2–3 months) → reserved for a **final rolling-origin evaluation of the already-selected configuration only** — never consulted while comparing candidates or tuning hyperparameters, exactly as the FINAL UNTOUCHED TEST set is never consulted during categorization model selection (§6).

**If the available naturally-occurring history is too short to reserve a defensible untouched final temporal period** (a realistic possibility given the frozen 12-month MVP threshold, §15) — the strongest feasible calendar-boundary expanding-window validation from §12 is used instead, but the resulting number is explicitly labeled **"temporal validation performance,"** never **"untouched temporal-test performance."** This distinction must be disclosed in any evaluation report (§19) and any resume/interview claim (§21) — a validation-only result described as if it were a held-out test result would overstate the evidence's strength. **No final holdout is ever fabricated merely to make the methodology appear more rigorous than the available data supports.**

---

## 13. Forecast Metrics

**Evaluated:** MAE, RMSE, MAPE (matching V1's existing reporting, extended per below).

**MAPE's known problems, explicitly discussed:**
- **Zero actual spend:** MAPE is undefined (division by zero) — V1's own `walk_forward_validate` already guards this with `max(abs(actual), 1e-9)` (`forecast.py:221`), which avoids a crash but produces an enormous, meaningless percentage for a near-zero actual (e.g., predicting $5 against an actual of $0.01 yields an ~50,000% "error" that is not informative).
- **Near-zero actual spend:** even without hitting exactly zero, a category with a genuinely small true value (e.g., a $12 Healthcare month) can produce a huge percentage error from a small absolute miss, disproportionately dominating an aggregate MAPE relative to its real-dollar impact.
- **Sparse categories:** categories with few historical months (Healthcare, per the synthetic data's own amount ranges being among the smallest, `seed_synthetic_data.py:44-51`) are exactly where both of the above problems concentrate.

**Recommendation — WAPE (Weighted Absolute Percentage Error) as primary, not raw MAPE:** WAPE (`sum(|actual - predicted|) / sum(|actual|)`, aggregated across the evaluation set rather than averaged per-row) is far less distorted by individual near-zero actuals than MAPE, because a small actual's contribution to the denominator is naturally weighted by its own small magnitude rather than inflating a per-row percentage that then gets equally averaged with every other row. **MAE is the recommended secondary metric** (same absolute-dollar units a user/reviewer intuitively understands, no percentage-of-small-number distortion at all). **MAPE is retained only as a tertiary, clearly-labeled compatibility metric** for continuity with V1's existing reporting style and this project's prior documentation (README, TECHNICAL_WALKTHROUGH) — not as the number driving model selection.

**Primary model-selection metric: WAPE**, computed both in aggregate (all categories/months pooled) and per-category (to catch the sparse-category distortion WAPE reduces but does not eliminate — a category can still have a poor WAPE in isolation even if it's a rounding error in the aggregate).
**Secondary metrics:** MAE (aggregate + per-category), RMSE (sensitivity to large misses), MAPE (compatibility/communication only, always reported alongside a caveat).

These metrics are **not** chosen merely because V1 already prints MAPE — WAPE/MAE are added specifically to correct MAPE's documented failure modes for this project's actual data shape (small dollar amounts, sparse categories).

### 13.1 All three horizons must be evaluated — not just +1

PlainCents V2's product surface displays a **3-month forecast** (`month_offset ∈ {1,2,3}`, TRD §4.6/§6) — a candidate that looks strong at +1 but degrades badly at +2/+3 would be shipping a product feature that was never actually validated for two-thirds of what it displays. **Reporting is therefore required at every horizon separately, not only in combination:**

| Horizon | Required metrics |
|---|---|
| **+1** | WAPE, MAE |
| **+2** | WAPE, MAE |
| **+3** | WAPE, MAE |
| **Combined** (all three horizons pooled) | Aggregate WAPE, aggregate MAE |

RMSE and MAPE remain secondary/diagnostic and may be reported combined only, not mandatorily per-horizon, without weakening the acceptance gate (§20) — the primary WAPE/MAE numbers are what must be broken out by horizon.

**The model-selection review (§14) must explicitly check for a good +1 result masking a poor +2/+3 result** — a candidate is not accepted on the strength of its +1 performance alone; §20's forecaster gate requires all three horizons' numbers to be recorded and reviewed together, per-category, not just in aggregate.

---

## 14. Forecast Model-Selection Rule

**Rule:** a candidate becomes eligible to be the final forecaster only if it (a) outperforms the naive baseline (and seasonal-naive, where evaluated) on the primary metric (WAPE), **evaluated separately at +1, +2, and +3 (§13.1) — not only in combination** — under the corrected calendar-month-boundary temporal validation protocol (§12), by a margin large enough to not be plausibly explained by fold-to-fold noise (assessed qualitatively from the walk-forward fold-level results — an arbitrary fixed percentage improvement threshold, e.g. "must beat naive by 10%," is **not** invented here without evidence, per the explicit instruction), (b) shows reasonably stable per-category WAPE across folds/time windows **and across all three horizons** (no category or horizon with wildly inconsistent performance from one evaluation window to the next — a candidate that wins convincingly at +1 but loses to naive at +2/+3 does not pass this rule at the aggregate/combined level alone), and (c) remains computationally appropriate for the interactive, synchronous forecast-generation path the TRD already scoped down to a single fit (TRD §12.3) — i.e., candidate training time must not reintroduce the performance problem TRD §12.1–§12.3 solved by removing walk-forward/GridSearchCV from the user path.

**A candidate may pass at some horizons and not others.** If this occurs, the scientifically correct outcome may be a **per-horizon decision** (e.g., use the winning candidate's multi-step strategy for +1 but fall back to naive/seasonal-naive for +2/+3 if nothing beats it there) rather than forcing one model across all three horizons for consistency's sake alone — though a single strategy that passes at all three horizons is preferred for implementation simplicity (TRD §12.2's single `run_forecast()` call) if the evidence supports it.

**Explicit possible outcome — the naive/simple baseline wins:** if Random Forest or Ridge does not reliably clear naive/seasonal-naive on WAPE with reasonable per-category, per-horizon stability, **the scientifically correct decision is to ship the simpler model** (or even the naive baseline itself, if nothing beats it, at some or all horizons). Model complexity is not itself a success criterion, and choosing RF anyway "because it's more sophisticated" would directly violate §0's non-negotiable principle.

---

## 15. History Requirement / 12-Month Cold Start

**ML-F AMENDMENT (reports/ml/ML_F_SELECTION_RECORD.json's forecasting_selection):** the experiment this section specifies was executed in ML-F, extended to also cover the rolling-mean/EWMA candidates ML-F added. Its finding — the selected recipe's pooled WAPE is stable across 6/9/12/18-month truncated history, exactly as it was for ML-C's Naive — was carried back through the PRD amendment process this section itself requires (PRD §21 now reads 6 months, TRD §12.5 updated to match). The frozen-at-authoring-time text below is preserved as the historical record of what was asked for and why; it is no longer the current product rule.

The frozen product rule — **12 unique months** overall eligibility (PRD §21, TRD §12.5) — is **not changed by this document**. This section defines the experiment that would inform a *future* PRD amendment, not a change made here.

**Experiment design (ML-B, not executed in this document):** using the independent evaluation data if it has sufficient time span, or synthetic data explicitly labeled as such for this specific experiment (§16 — synthetic history-length experiments are legitimate for testing model *behavior*, just not for claiming real-world accuracy), evaluate the chosen forecaster's WAPE/MAE under the corrected walk-forward protocol (§12) using training histories artificially truncated to 6, 9, 12, and 18+ months, holding the same set of test months constant across all four runs where possible. The specific questions this answers: does forecast quality meaningfully degrade below 12 months, and does the model beat naive/seasonal-naive at each truncation length?

**If evidence from this experiment suggests 12 months is unnecessarily conservative (e.g., 9 months performs comparably) or insufficiently conservative (e.g., even 12 months is unstable), that finding is recorded as a candidate future PRD amendment** — the frozen product rule is not silently changed by an ML experiment; any change goes back through the PRD amendment process the PRD itself describes (PRD header: "changes to frozen product scope require an explicit PRD amendment").

---

## 16. Per-Category Sparsity

V1's current per-category eligibility is **incidental**: a category simply doesn't appear in the final feature set if its `dropna` on rolling/lag columns removes all its rows (§1.2), which is a side effect of the feature-construction code, not a deliberately chosen statistical threshold.

**Candidate rules evaluated** (none finalized without evidence — this is explicitly ML-B work):
- **Minimum total historical months for that category** (e.g., the category must have appeared in at least N of the available months) — simple, but doesn't distinguish "sparse but consistent" from "one huge outlier month."
- **Minimum non-zero-spend months** — addresses a category that technically has "history" but almost all zeros (e.g., Healthcare in a given user's data), which would produce a degenerate/near-constant-zero forecast that is technically "available" but not useful.
- **Minimum recent observations** (e.g., at least one non-zero month within the last 6) — guards against a category that had activity long ago but has since gone dormant, where forecasting future spend from stale history is misleading.
- **Sufficient rows survive `build_forecast_features`'s existing rolling/lag `dropna`** — this is V1's current de facto rule; retained as a floor (a category can never be "available" if it doesn't even survive feature construction) but not treated as sufficient on its own, since it doesn't capture the non-zero/recency distinctions above.

**Until ML-B produces evidence favoring one specific rule, the TRD contract is preserved exactly as specified:** a category that cannot be defensibly forecast returns `is_available: false, unavailable_reason: "insufficient_history"` (TRD §6/§12.5) — this document does not silently substitute a new, unevaluated rule into that contract; it only names what evidence would be needed to refine the rule later.

---

## 17. Synthetic Forecast Tests

Synthetic data remains legitimate and useful for testing **pipeline behavior and known statistical properties**, never for claiming real-user accuracy:

- **Known constant-spend process** — a category with a fixed, unchanging monthly amount; the forecaster should predict close to that constant (sanity check).
- **Known trend** — a linearly increasing/decreasing series; verifies the model captures directional movement rather than reverting to a flat average.
- **Known seasonality** — a series with an engineered December spike (mirroring V1's own `MONTH_MULTIPLIER` in `seed_synthetic_data.py:54-59`); verifies `is_december`/`is_summer` features are actually being used effectively.
- **Sudden regime change** — a series with an abrupt level shift (e.g., a user's rent doubling); tests how quickly rolling-window features adapt and whether the model reacts sensibly rather than being anchored to stale history.
- **Sparse categories / zero-spend months** — directly exercises §16's eligibility logic once it's implemented.
- **Leakage detection** — a deliberately constructed series where a naive/buggy implementation would leak future information (e.g., a target month's own value accidentally present in a rolling feature); the test asserts the correct implementation does *not* achieve suspiciously perfect accuracy on such a series.
- **Rolling-feature correctness** — direct unit tests confirming `rolling_3m_avg`/`rolling_6m_avg`/`lag_1_spend` compute the expected numeric values against hand-calculated examples.

None of these establish real-user forecasting accuracy — they establish that the *mechanism* behaves as designed, which is a prerequisite for trusting any real-data evaluation built on top of it.

---

## 18. Model Artifact and Versioning Strategy

**No MLflow / model registry** — a lightweight, file-and-naming-convention approach is sufficient for this project's scale (§0's complexity-discipline principle extends to tooling, not just models).

**Categorization artifact metadata** (recorded alongside the saved artifact, exact file format is a Build Plan/implementation detail, not fixed here): model implementation name (e.g., `"kmeans_v1"`, `"tfidf_logreg_v1"`), artifact version/id (a simple incrementing string or the training date), training date, training dataset version/id (§3.2's independent dataset, once it exists, is itself versioned — e.g., `"eval_dataset_v1"`), feature configuration version (e.g., which TF-IDF settings), evaluation report reference (a path to the metrics/confusion-matrix output from §7/§8), random seed where applicable (K-Means's `random_state`; a supervised candidate's own seed if stochastic).

**Forecasting artifact/versioning:** the TRD already anticipates a `model_impl_version` free-text field on every `forecast_runs` row (TRD §4.6, §22 decision table). **This document defines how that value is produced:** it is a string identifying the *forecasting implementation and configuration* actually used to generate that run — e.g., `"rf_v1_default_hparams"` or, if a future implementation wins the bake-off, `"ridge_v1"`. Whether a **persistent trained artifact** (a saved `.pkl`, analogous to `kmeans_model.pkl`) is appropriate is evaluated per candidate: since TRD §12.3 specifies the interactive path fits fresh on the user's *current* transaction history at each `run_forecast()` call (not a static pre-trained model applied to new data), the "artifact" for forecasting is more accurately the **implementation + fixed hyperparameter configuration** (versioned as code/config) rather than a single persisted fitted model object — a fresh model is fit per run by design (TRD §12.3), so there is no long-lived trained-weights artifact to version the way `kmeans_model.pkl` is versioned for categorization. This is a genuine asymmetry between the two components, not an oversight: categorization needs a persisted fitted artifact because inference must be fast and consistent across many transactions between retrains; forecasting's TRD-mandated design deliberately refits per invocation on the latest data, so its "version" is the *recipe*, not a saved weight file.

---

## 19. Reproducibility

Before any final ML result is accepted (§20): pinned Python dependencies for the ML evaluation environment (addressing the reality check's reproducibility gap, TRD §16); fixed random seeds wherever a candidate is stochastic (K-Means, any bootstrap/random-split step, RF's own `random_state`); an immutable definition of the merchant-grouped split (§6) — e.g., a saved list of which merchant identities fall into TRAIN/VALIDATION/TEST, not a re-randomized split each run; a training script per candidate; an evaluation script producing the metrics in §7/§13; saved metrics output (plain text/JSON/CSV, not a bespoke format) and the confusion matrix for categorization candidates; the resulting artifact/config per §18; and enough written documentation (a short README alongside the evaluation output) that a clean environment plus the permitted evaluation dataset can reproduce the reported numbers. None of these scripts/files are created by this document (§23 defers their creation).

---

## 20. Final Model Acceptance Gates

### Categorizer
- [ ] Evaluation dataset protocol documented (§3.2) and the actual dataset acquired.
- [ ] Merchant-grouped, category-stratified TRAIN/VALIDATION/TEST split frozen (§6) and saved immutably (§19).
- [ ] K-Means baseline evaluated under this protocol **with strict TRAIN-only fitting and cluster→category mapping** (§6.1) — not merely re-reported from V1's original synthetic evaluation, and not using VALIDATION/TEST labels for mapping construction.
- [ ] TF-IDF + Logistic Regression evaluated.
- [ ] TF-IDF + Linear SVM evaluated (§5).
- [ ] Primary (macro F1) and secondary (accuracy, per-category F1) metrics recorded for every candidate.
- [ ] Confusion matrix produced for every candidate.
- [ ] Structured error analysis completed (§8).
- [ ] Final untouched-test evaluation performed exactly once, on the selected candidate only.
- [ ] Artifact/version metadata recorded (§18).
- [ ] No synthetic-only metric presented as real-world performance anywhere in the resulting report.

### Forecaster
- [ ] Naive and seasonal-naive baselines evaluated under the corrected calendar-month-boundary temporal protocol (§12).
- [ ] Hyperparameter-search fold-assignment weakness corrected using explicit calendar-month boundaries — not row-sorted `TimeSeriesSplit` — before any tuned result is trusted (§12).
- [ ] RF re-evaluated under the corrected protocol, using the **same hyperparameter configuration the TRD's interactive `train_and_predict()` path actually ships** (§11), not re-reported from V1's original, partially flawed evaluation or its diagnostic walk-forward defaults.
- [ ] Ridge candidate evaluated (§11).
- [ ] Primary (WAPE) and secondary (MAE, RMSE, MAPE) metrics recorded, both aggregate and per-category, **and separately at each of +1/+2/+3 for WAPE and MAE** (§13.1) — not combined-only.
- [ ] Model-selection review explicitly checked for a horizon where good +1 performance masks poor +2/+3 performance (§14).
- [ ] The multi-step forecasting strategy actually evaluated (§11.1) is the same strategy the shipped implementation uses — no unevaluated strategy is called final.
- [ ] Per-category/per-window/per-horizon stability reviewed (§14).
- [ ] Where an untouched final temporal period was feasible, it was reserved and not consulted during candidate/hyperparameter selection (§12.1); where it was not feasible, results are explicitly labeled "temporal validation performance," not "test performance."
- [ ] History-length behavior evaluated where the available data permits (§15) — recorded as a finding, not silently applied as a product change.
- [ ] Per-category sparsity behavior documented against the candidate rules in §16 (rule not necessarily finalized, but the evaluation itself must occur).
- [ ] Implementation/configuration version metadata recorded (§18).
- [ ] No synthetic-only metric presented as real-world performance anywhere in the resulting report.

---

## 21. Resume / Interview Claim Policy

**Allowed only after the corresponding §20 gate has passed:** the final model family actually selected; held-out (FINAL UNTOUCHED TEST) categorization performance from the independent dataset, **labeled by which data tier (§3.2 Tier A/B/C) it came from**; temporal-validation (or, where feasible, untouched-final-period test, §12.1) forecasting performance from the corrected protocol, **reported per-horizon (+1/+2/+3) as well as combined** (§13.1); a stated improvement over a *named, actually-evaluated* baseline; an accurately described dataset scope, provenance, and tier (e.g., "evaluated on N independently labeled transactions from a curated benchmark (Tier B)" or "...from real de-identified transaction history (Tier A)" — never blurring the two, and never implying a larger or more diverse dataset than what was actually used).

**Not allowed, ever, regardless of how the ML Track concludes:**
- "90% real-world accuracy" — this number is V1's synthetic diagnostic (§2), never real-world.
- "<15% MAPE" from the diagnostic heuristic-category result (§2) — this bypassed K-Means entirely and is not a production-shaped number.
- Describing synthetic evaluation, or Tier B curated/constructed benchmark evaluation, as real-bank (Tier A) performance under any phrasing.
- Describing a Tier C public dataset as "real-world" without having actually verified its provenance supports that characterization.
- Claiming the 29.4%/15.7% gap as a validated causal decomposition of categorization-attributable error (§2).
- Claiming production-grade performance without a passed §20 gate to point to.
- Claiming superiority over a baseline that was not actually run under the same protocol (e.g., claiming "beats naive" without having actually evaluated naive under §12's corrected calendar-month-boundary protocol).
- Claiming forecast accuracy without specifying which horizon(s) it applies to, or citing only a combined/+1 number when +2/+3 performance was materially worse (§13.1/§14).
- Describing a temporal-validation-only result (§12.1) as "test" or "held-out" performance when no untouched final period was actually feasible.

**Every numeric resume claim must cite a specific, reproducible evaluation artifact/report** (§19) — a claim with no traceable report behind it is not permitted, matching the discipline the PRD/TRD already apply to acceptance criteria.

---

## 22. Implementation Phasing Relationship

- **Phase ML-A — App Bootstrap:** integrate current K-Means + RF behind the TRD's `CategorizationService`/`ForecastService` contracts (already specified, TRD §7.3/§7.4) so FastAPI/React construction proceeds immediately. **Not** a scientific acceptance of either model — purely enables product development in parallel.
- **Phase ML-B — Scientific Evaluation:** execute §3's data acquisition, §6/§12's corrected splitting/validation protocols, §5/§11's bake-offs, §8's error analysis, §15/§16's history/sparsity experiments. May run fully in parallel with application-track Build Plan phases (per the earlier planning discussion's "parallel ML track").
- **Phase ML-C — Model Selection:** apply §14's forecast rule and §7's categorization metric to choose final implementations — including the explicit possibility that the current baseline (K-Means, or even naive-over-RF) is kept because nothing beat it (§0, §14).
- **Phase ML-D — Final Integration:** if a candidate other than the current implementation wins, swap it in behind the **unchanged** `CategorizationService`/`ForecastService` contracts (TRD §11.4/§12.2's replacement-boundary design exists exactly for this), then rerun unit/repository/API integration tests (TRD §17) and reproduce the final evaluation numbers from a clean environment (§19) before calling the project scientifically final.

**No product-facing API/schema/frontend contract changes as a result of ML-D** — this is a direct consequence of the TRD's model-agnostic boundary design (TRD §11.4, §12.2), not a new constraint invented here.

---

## 23. Data / Experiment Artifacts To Be Created Later

**Not created by this document.** A minimal future organization, sized to avoid turning the repository into an MLOps platform:

```
ml/
    training/          # one script per candidate (categorization + forecasting)
    evaluation/         # metric computation, confusion matrix, error-analysis output
    experiments/         # history-length (§15) and sparsity (§16) experiment scripts

data/
    evaluation/          # the independent labeled dataset (§3.2) and its frozen split definition (§19)

models/
    (existing kmeans_model.pkl / rf_model.pkl location; a future winning categorizer artifact
     would be versioned alongside, e.g. tfidf_logreg_v1.pkl, without deleting the V1 baseline
     artifact needed for comparison reporting)

reports/
    ml/                 # the evaluation report(s) §19/§21 require for reproducibility and resume claims
```

This mirrors the TRD's own restraint (no MLOps platform, TRD §19) applied to the ML side specifically.

---

## 24. Open Questions

**BLOCKING BEFORE APP IMPLEMENTATION:** none. Phase ML-A (§22) explicitly does not require any of this document's experiments to have run — the app integrates the current K-Means/RF baseline immediately, per the dual-track principle (§0).

**BLOCKING BEFORE FINAL MODEL SELECTION (i.e., before §20's gates can be marked passed):**
1. Which acquisition path (§3.2: personal de-identified data, manually curated realistic samples, or a vetted public dataset) will actually produce the independent categorization evaluation dataset — none exists yet, and this is a prerequisite for every categorization gate in §20.
2. Execution of the corrected calendar-month-boundary hyperparameter-search protocol (§12) — until run, no tuned RF or Ridge result can be trusted as leakage-safe.
3. The actual bake-off runs themselves (§5, §11) — no candidate has been evaluated under this document's protocols yet; this document defines the experiments, it does not report their results.

**NON-BLOCKING / EXPERIMENT-DEPENDENT:**
1. The exact per-category sparsity rule (§16) — the TRD's existing contract (`is_available`/`unavailable_reason`) already covers product behavior in the meantime.
2. Whether 12 months remains the right overall eligibility threshold (§15) — a finding for a possible *future* PRD amendment, not something blocking current work.
3. Whether a confidence/calibration score is ever surfaced in the product — not required by the frozen TRD/PRD regardless of what a winning categorizer happens to support.
4. The exact error-analysis sample size (§8) — scales with how much independent data ends up being collected, itself an open question (#1 above).
5. Whether nested temporal cross-validation is ultimately needed for forecasting (§12) — contingent on how ambiguous the simpler protocol's results turn out to be.

---

## Required Self-Audit

**A. K-Means TRAIN-only fitting/mapping check** — §6.1 explicitly restricts scaler/vectorizer/K-Means fitting and cluster→category mapping construction to TRAIN labels only; VALIDATION/TEST are transform-and-predict-only against the already-frozen pipeline.

**B. VALIDATION/FINAL TEST leakage into mapping check** — §6.1's "explicitly prohibited" list directly forbids VALIDATION/TEST labels from influencing preprocessing fitting, K-Means fitting, mapping construction, feature selection, or hyperparameter choice.

**C. Real vs. curated-synthetic distinction check** — §3.2's Tier A/B/C system explicitly separates naturally-occurring real data (Tier A) from independently curated/constructed benchmarks (Tier B), with Tier B explicitly barred from being described as real-world performance under any framing.

**D. Public-data-not-assumed-real check** — §3.2 Tier C explicitly states a public dataset supports a real-data claim only if its provenance/licensing genuinely establish it as naturally occurring, not merely because it is publicly available.

**E. Calendar-month-boundary check** — §12 explicitly rejects row-sorted `TimeSeriesSplit` (even after re-sorting) as insufficient, and mandates an explicit expanding-window loop keyed on calendar month, with a worked example showing month-by-month boundaries.

**F. No-split-month check** — §12's expanding-window loop moves each month's full set of category rows from VALIDATE to TRAIN as one unit when the window expands — no month's rows are ever divided between TRAIN and VALIDATION at any step.

**G. Selection-vs-final-evaluation separation check** — §12.1 explicitly separates earlier-history candidate/hyperparameter selection from a later untouched temporal period reserved for final evaluation, mirroring §6's categorization TRAIN/VALIDATION/TEST discipline, with an explicit "temporal validation performance" labeling fallback when an untouched period isn't feasible.

**H. Validation-vs-test labeling check** — §12.1 and §21 both explicitly require a temporal-validation-only result to be labeled as such, never as "test" or "held-out" performance, and explicitly forbid fabricating a final holdout merely to appear more rigorous.

**I. All-horizons-evaluated check** — §13.1 requires WAPE and MAE at +1, +2, and +3 individually, not just combined; §14 requires the model-selection review to explicitly check for a horizon where good +1 performance masks poor +2/+3 performance.

**J. Multi-step-strategy-matches-shipped-strategy check** — §11.1's binding rule explicitly states the shipped multi-step strategy must be the same one evaluated under §12/§13.1's protocol, and that the forecaster acceptance gate (§20) is not satisfied by an unevaluated strategy even if the application has already bootstrapped on it (Phase ML-A).

**K. Categorization bake-off check** — §5 is now frozen to exactly three candidates (K-Means, Logistic Regression, Linear SVM), with SGDClassifier and KNN explicitly excluded from the initial bake-off (SGDClassifier noted only as a possible future fallback).

**L. Forecast bake-off check** — §11 is now frozen to exactly four candidates (naive, seasonal naive, Random Forest, Ridge), with Lasso, moving-average, XGBoost/gradient boosting, and deep time-series models explicitly excluded from the initial bake-off.

**M. No new heavy model family check** — no LSTM, Transformer, embedding-based, XGBoost, or gradient-boosting candidate appears anywhere in the frozen bake-offs (§5, §11); both explicitly state no family is added without a future, separate ML-Spec amendment.

**N. WAPE definition check** — §13 already correctly defined `WAPE = sum(|actual - predicted|) / sum(|actual|)` prior to this amendment pass; no substantive change was needed, and the existing explanation of why WAPE/MAE are safer than raw MAPE for sparse/near-zero categories is retained unchanged.

**O. `model_impl_version` check** — §18 already described `model_impl_version` as identifying the forecasting implementation/configuration specifically (not a combined categorizer+forecaster string) prior to this amendment pass; verified consistent with the reasoning that V2 forecasting consumes effective categories (which may include user corrections), so a forecast run is not attributable to one categorizer implementation. No change was needed to comply with this amendment's item 10.

**P. Frozen PRD/TRD contracts unchanged check** — no amendment in this pass touches `CategorizationService`'s or `ForecastService`'s TRD-defined contracts, the `is_available`/`unavailable_reason` schema, the 12-month eligibility rule, or any API/schema/frontend description; all changes are confined to evaluation methodology and bake-off scope.

**Q. Phase ML-A unblocked check** — §22's Phase ML-A description is unchanged by this amendment pass; the app still bootstraps on the current K-Means/RF implementation immediately, independent of when §6.1/§12/§12.1/§13.1's stricter evaluation protocols are actually executed in ML-B.

**Carried forward from the original draft, still holding after amendment:**

**Synthetic claim check** — §2 and §21 both explicitly state 90%/15.7% are synthetic/diagnostic, never real-world; §16/§17 explicitly separate synthetic pipeline-behavior tests from real-accuracy claims.

**Baseline check** — §0, §5, §11, §14 all state K-Means/RF are baselines to be beaten or matched with justification, explicitly including the outcome where they remain the final choice because nothing beat them, or where a simpler model wins outright. No predetermined winner declared anywhere in this document.

**12-month check** — §15 explicitly states the frozen rule is not changed by this document and defines only an experiment whose findings would feed a *future* PRD amendment process, never a silent change.

**Sparsity check** — §16 explicitly states V1's `dropna` behavior is retained only as a floor, not treated as the final scientific rule, and lists multiple candidate rules requiring evidence before any one is chosen.

**Correction-loop check** — §9 explicitly prohibits automatic/online retraining on corrections and requires the full offline versioned pipeline before any retrained model can be deployed.

**Source check** — every statement about current V1 behavior in §1/§2/§12 cites a specific file and, where relevant, line numbers, verified by reading the actual files in this session (`pipeline/features.py`, `pipeline/cluster.py`, `pipeline/forecast.py`, `scripts/generate_synthetic_24mo.py`, `tests/test_pipeline.py`, `scripts/diagnose_heldout.py`, `docs/V2_PREBUILD_REALITY_CHECK.md`) rather than assumed from any authoring prompt's own suggested numbers.

**No blocking PRD/TRD contradiction was exposed by this amendment pass.**

---

*No production code, model artifacts, datasets, or other documents were modified in the production of this document. Only `docs/V2_ML_SPEC.md` was created.*
