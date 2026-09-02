# PlainCents V2 — ML-C: Model and Strategy Selection Report

**Status:** ML-C complete. Selections frozen, both permitted FINAL passes executed exactly once each.
**Commit evaluated:** `2c06181a12fb270e6e534564c98ccebd2998088c` (HEAD == origin/main at the start of this phase; this ML-C work is additive/uncommitted at time of writing, same convention ML-B used).
**Authority:** `docs/V2_ML_SPEC.md` (frozen). Where this report and the ML Spec appear to disagree, the ML Spec wins.
**Reproduce:** `python -m ml.forecasting.fold_stability`, `python -m ml.categorization.run_final`, `python -m ml.forecasting.run_final`, using the pinned environment in `requirements.txt` (`venv/`). Seed 42 throughout, inherited from ML-B.

---

## 1. ML-C purpose

ML-B answered "what does TRAIN+VALIDATION evidence show for each frozen candidate?" without selecting a winner. ML-C answers the next, narrower question: **given that frozen TRAIN+VALIDATION evidence, which single categorization candidate, which single forecasting candidate/configuration, and which multi-step strategy should PlainCents actually select** — before, and only before, opening either component's single permitted FINAL pass. ML-C does not integrate anything into production (`CategorizationService`/`ForecastService` are untouched, verified in §23) and does not begin ML-D.

## 2. Exact ML-B evidence inherited

- `reports/ml/ML_B_EXPERIMENT_REPORT.md` and `reports/ml/PRE_EXPERIMENT_REPORT.md` (narrative reports).
- `reports/ml/results/categorization_results.json` (per-candidate VALIDATION/TRAIN-diagnostic metrics, confusion matrices).
- `reports/ml/results/categorization_error_analysis.json` (per-candidate, per-row structured §8 error log).
- `reports/ml/results/forecasting_metrics.json` (pooled VALIDATION metrics by candidate/strategy, combined/by-horizon/by-category).
- `reports/ml/results/forecasting_predictions_long.csv` (1,872 prediction-level rows across 14 origins — the artifact ML-C's Part B fold-stability analysis is computed from).
- `reports/ml/results/history_length_sensitivity.json`, `reports/ml/results/sparsity_analysis.json`.
- `reports/ml/results/experiment_log.jsonl` (9 experiments, all `SUCCESS`).
- `data/evaluation/tier_b_split_v1.json` (frozen merchant→partition assignment), `data/evaluation/tier_b_benchmark.csv`.

**Independent verification performed (Part A, before any selection work):** every headline number quoted from `ML_B_EXPERIMENT_REPORT.md` was cross-checked directly against its underlying JSON artifact (categorization macro F1/accuracy for all 3 candidates, forecasting combined and per-horizon WAPE for all 6 candidate/strategy pairs, history-length sensitivity table, experiment log status). **No disagreement was found between the ML-B report and its machine-readable artifacts** — every figure matched to the precision reported (e.g. K-Means VALIDATION macro F1 `0.056589147286821705` in `categorization_results.json` vs. "0.0566" in the report; Naive combined WAPE `0.1902958523908706` in `forecasting_metrics.json` vs. "0.1903" in the report). Per the ML-C brief's STOP rule, this means Part A's verification requirement is satisfied with no blocking discrepancy to report.

## 3. Evidence tiers (unchanged from ML-B, restated per ML-C's evidence-tier discipline)

| Component | Tier | Dataset | Never described as |
|---|---|---|---|
| Categorization | **Tier B** — independently curated/constructed benchmark (ML Spec §3.2) | `data/evaluation/tier_b_benchmark.csv`, 228 rows / 81 merchant groups, single author | Real-world / Tier A performance |
| Forecasting | **Synthetic** (ML Spec §3.1) | `data/raw/synthetic_24mo.csv` run through the production K-Means artifact (read-only) | Tier B; real-world; "temporal validation" for the FINAL result specifically (an untouched period *was* feasible and *was* used, §21) |

No claim in this report treats either tier as real-world (Tier A) evidence, and the two tiers are never blended into one blanket claim.

---

## 4. Categorization VALIDATION comparison (re-verified)

| Candidate | VALIDATION macro F1 | VALIDATION accuracy | TRAIN accuracy (diagnostic) |
|---|---|---|---|
| K-Means (TRAIN-only isolated) | 0.0566 | 12.0% | 42.1% |
| TF-IDF + Logistic Regression | **0.2552** | **32.0%** | 78.2% |
| TF-IDF + Linear SVM | 0.2405 | 26.0% | 82.0% |

Random-chance accuracy with 8 roughly-balanced categories ≈ 12.5% — K-Means's 12.0% is statistically indistinguishable from guessing. Source: `reports/ml/results/categorization_results.json`, independently re-verified in this phase (§2).

## 5. §7/§8 per-category/error gate

**§7 requirement:** any category with F1 substantially below the others triggers mandatory §8 error analysis before that candidate can be considered for selection.

| Candidate | VALIDATION categories at F1 = 0.0 (of 8) |
|---|---|
| K-Means | 6: Entertainment, Healthcare, Other, Rent & Utilities, Shopping, Subscriptions |
| TF-IDF + Logistic Regression | 3: Healthcare, Other, Shopping |
| TF-IDF + Linear SVM | 3: Food & Dining, Healthcare, Other |

**Gate check:** the structured §8 error analysis was completed for **all three** candidates, not only the worst — `reports/ml/results/categorization_error_analysis.json` logs every VALIDATION misclassification (all 50 rows reviewed, per §8's "review all if the total is small" rule) with `{merchant/description, amount, true_category, predicted_category, candidate_model}`. Root cause, quantified in ML-B §15 and re-confirmed here: **0 of 50 VALIDATION merchant strings share even one token with TRAIN's fitted top-50 TF-IDF vocabulary** — every VALIDATION merchant group is, by construction of the correct merchant-grouped split, one the models never saw. Category performance tracks whether that category's TRAIN vocabulary happens to be *generic* (Rent & Utilities, Subscriptions — words like "bill"/"monthly"/"membership" recur across different real merchants of the same type) vs. *merchant-proper-noun-dependent* (every other category, whose TRAIN vocabulary is dominated by specific business names that don't generalize). This is cited explicitly per the §7/§8 gate instruction, and it explains *why* K-Means's failure is structural (chance-level across nearly every category) rather than isolated to one or two categories, distinguishing it from LogReg/SVM's narrower 3-category weak spot.

Gate outcome: **satisfied for all three candidates** — none is disqualified by an unperformed error analysis. K-Means is rejected below on primary/secondary metric evidence, not on the gate.

## 6. Categorization selection

**Selected: TF-IDF + Logistic Regression** (`ml/categorization/candidates.py::TfidfLogRegCandidate`, `C=1.0, max_iter=1000, random_state=42`, TF-IDF `max_features=50, ngram_range=(1,2), sublinear_tf=True`, TRAIN-only fit per §6.1 isolation).

**Primary metric (macro F1) and secondary metric (accuracy) both favor Logistic Regression** over Linear SVM (0.2552 vs. 0.2405; 32.0% vs. 26.0%). This is a real but not enormous margin (~6% relative on macro F1), so it was treated as a close call per the ML-C brief's close-call rule rather than accepted uncritically:

- **Per-category failure pattern:** both candidates have exactly 3/8 categories at F1=0.0 on VALIDATION, but *different* categories (LogReg: Healthcare/Other/Shopping; SVM: Food & Dining/Healthcare/Other) — neither pattern is more concentrated or more broadly distributed than the other; this does not favor either candidate.
- **Complexity/implementation asymmetry:** `LinearSVC` (the sklearn class backing the SVM candidate) does not produce calibrated probabilities without an added `CalibratedClassifierCV` step; `LogisticRegression` produces genuine class probabilities natively at no extra cost. Not required by the product today (TRD §11.2), but a real, uncontested asymmetry if confidence scoring is ever added.
- **Runtime/artifact burden:** both fit in well under a second at this data scale (ML-B §30) and require an equivalent 2-object artifact (vectorizer + model) — no differentiator.

No evidence favors SVM enough to override the primary-metric leader. **Decision: TF-IDF + Logistic Regression.**

### 6.1 Rejected categorization candidates

- **K-Means:** near-chance VALIDATION performance (12.0% accuracy vs. ~12.5% random baseline; macro F1 0.057) once merchant-leakage was corrected — a ~30-point TRAIN(42.1%)/VALIDATION(12.0%) accuracy gap is a textbook generalization-failure signature. §7/§8 gate satisfied (root cause identified and quantified, §5 above). Rejected on primary-metric evidence.
- **TF-IDF + Linear SVM:** beaten on both primary and secondary VALIDATION metrics by Logistic Regression, with no offsetting complexity/runtime/maintainability advantage identified (§6 close-call analysis above).

---

## 7. Forecasting pooled VALIDATION comparison (re-verified)

**Combined (all horizons pooled), primary metric WAPE** — `reports/ml/results/forecasting_metrics.json`, independently re-verified against `ML_B_EXPERIMENT_REPORT.md` §21 (exact match to reported precision):

| Candidate / Strategy | WAPE | MAE | n |
|---|---|---|---|
| **Naive** | **0.1903** | 34.82 | 312 |
| Ridge — last-known-history | 0.2237 | 40.93 | 312 |
| Ridge — recursive | 0.2395 | 43.83 | 312 |
| Random Forest — last-known-history | 0.2423 | 44.34 | 312 |
| Random Forest — recursive | 0.2565 | 46.93 | 312 |
| Seasonal Naive | 0.2631 | 47.14 | 216 (eligibility-limited) |

Naive has the lowest combined WAPE of every candidate.

## 8. §14 eligibility analysis

§14(a) requires a candidate to outperform Naive on WAPE **separately at +1, +2, and +3** — not only combined — by a margin not plausibly explained by fold-to-fold noise.

**Pooled per-horizon WAPE** (re-verified against `ML_B_EXPERIMENT_REPORT.md` §22):

| Candidate / Strategy | +1 WAPE | +2 WAPE | +3 WAPE |
|---|---|---|---|
| Naive | 0.2020 | 0.1904 | 0.1765 |
| Seasonal Naive | 0.2631 | 0.2631 | 0.2631 |
| RF — last-known-history | 0.2426 | 0.2354 | 0.2495 |
| RF — recursive | 0.2426 | 0.2524 | 0.2771 |
| Ridge — last-known-history | 0.2020 | 0.2235 | 0.2493 |
| Ridge — recursive | 0.2020 | 0.2347 | 0.2887 |

**Every non-Naive candidate fails §14(a):**
- **Ridge — last-known-history**: ties Naive at +1 (numerically identical WAPE/MAE — a documented coincidence), then loses at +2 (0.2235 vs 0.1904) and +3 (0.2493 vs 0.1765). This is precisely the "good +1 masking poor +2/+3" pattern §13.1/§14 require checking for.
- **RF — last-known-history**: loses to Naive at every single horizon.
- Both **recursive** variants fail more severely — recursive error compounding degrades +2/+3 further still (§10 below).
- **Seasonal Naive**: loses at every horizon (constant 0.2631 vs. Naive's declining 0.2020/0.1904/0.1765).

No candidate clears the §14(a) bar at all three horizons. Per §14's explicitly anticipated outcome — *"if RF or Ridge does not reliably clear naive... the scientifically correct decision is to ship the simpler model... even the naive baseline itself, if nothing beats it"* — **no candidate is eligible to replace Naive.**

## 9. Forecasting fold-level stability (ML-C Part B — new analysis)

ML-B pooled its per-fold predictions into the tables above. ML-C computed genuine per-origin metrics directly from the frozen prediction-level artifact (`reports/ml/results/forecasting_predictions_long.csv`, no new split created) via `ml/forecasting/fold_stability.py`, persisted to `reports/ml/ML_C_FOLD_STABILITY.json`. This directly tests whether §14(a)'s aggregate result is stable across the 14 origins or driven by a handful of them.

**Per-origin, per-horizon win rate against Naive** (out of origins where both have a defined WAPE):

| Candidate / Strategy | +1 | +2 | +3 |
|---|---|---|---|
| RF — last-known-history | 3/14 (21%) | 6/13 (46%) | 3/12 (25%) |
| RF — recursive | 3/14 (21%) | 6/13 (46%) | 1/12 (8%) |
| Ridge — last-known-history | 5/14 (36%) | 5/13 (38%) | 2/12 (17%) |
| Ridge — recursive | 5/14 (36%) | 4/13 (31%) | 1/12 (8%) |

**Every candidate loses to Naive in the majority of individual origins at every horizon** (best case: RF-last-known-history at +2, still under 50%), and every candidate's win rate *worsens* at +3 (recursive strategies collapse hardest: 8%). This is direct evidence that §14(a)'s pooled result is **not** an artifact of a few unlucky origins — it is the norm across the 14-origin VALIDATION window. Naive also beats Seasonal Naive on **11/11** origins where Seasonal Naive is eligible (100%).

Per-origin WAPE spread for Naive itself: mean 0.1912, median 0.1821, std 0.0407, range [0.1249, 0.2579] across the 14 origins — a moderate, not wild, spread, meaning Naive's own performance is reasonably consistent rather than propped up by one exceptional origin.

**Horizon ranking stability:** at +1 and +2 the WAPE ranking (best→worst) is identical: `naive, ridge-lkh, ridge-recursive, rf-lkh, rf-recursive, seasonal_naive`. At +3 it reshuffles among the non-Naive candidates (`naive, ridge-lkh, rf-lkh, seasonal_naive, rf-recursive, ridge-recursive`) — Naive stays first at every horizon; the recursive strategies fall furthest behind specifically at +3, consistent with compounding error.

## 10. +1 comparison

Naive (0.2020) is tied by Ridge-last-known-history (0.2020, numerically identical prediction) and beaten by neither RF variant. At the individual-origin level, Ridge-lkh only wins 36% of origins at +1 despite the tied pooled number — the tie is a pooled/aggregate coincidence, not a per-origin dominance.

## 11. +2 comparison

Naive (0.1904) is not beaten by any candidate on pooled WAPE. RF-last-known-history's per-origin win rate peaks here (46%) — still under half.

## 12. +3 comparison

Naive (0.1765, its *best* horizon) is not beaten by any candidate; every candidate's per-horizon WAPE is markedly worse than at +1/+2 relative to Naive, and win rates against Naive collapse to 8–25%. This is the horizon at which recursive-strategy error compounding is most visible (§9 above, §20 below).

## 13. History-length findings (ML Spec §15, re-verified)

| Truncated TRAIN history | RF WAPE | Naive WAPE (reference) |
|---|---|---|
| 6 months | n/a — RF produces zero valid predictions (structural floor) | 0.2054 |
| 9 months | 0.3111 | 0.2054 |
| 12 months | 0.1846 | 0.2054 |
| 18 months | **0.1459** | 0.2054 |

RF only overtakes Naive at 18 months, in a **3-origin sample** (the only origins in the 24-month dataset with ≥18 months of prior history after reserving 3 for FINAL). This is a directional finding, not a robust basis for changing the selection: at the data volumes actually available across the 14-origin VALIDATION window, RF still loses combined and at every horizon (§7–§9). Noted for any future re-evaluation once more history accumulates, not acted on here.

## 14. Sparsity findings (ML Spec §16, re-verified)

"Other" is entirely zero-spend across the 21-month development region under this run's K-Means-derived labels (WAPE undefined/NaN for every candidate) — this affects all six candidate/strategy combinations identically and does not change the ranking. All other 7 categories are "dense" (20–21/21 nonzero months) or "intermittent" (Entertainment, 20/21). None of the four candidate sparsity-eligibility rules named in ML Spec §16 was adopted (per ML-B's explicit deferral); this remains an open item, not a blocker to forecaster selection.

## 15. Forecasting selection

**Selected: Naive** (`predicted = lag_1_spend`, no fitting, no hyperparameters — `ml/forecasting/baselines.py::naive_predict`).

Rationale: no candidate in the frozen four-candidate set clears the §14(a) eligibility bar at all three horizons (§8), and the fold-level stability review (§9) confirms this is a stable, majority-of-origins result at every horizon, not fold noise. Runtime/complexity strictly favor Naive as well: it is O(1) (a single lookup), strictly simpler than the currently-shipped Random Forest (`n_estimators=100, max_depth=10, min_samples_leaf=5`, refit per call) — selecting Naive is a genuine simplification, not merely "not worse."

## 16. Rejected forecasting candidates

- **Seasonal Naive:** loses to Naive at every horizon and 11/11 eligible origins; its own ≥13-month eligibility floor also limits its evaluated sample (216/312 rows).
- **Random Forest — last-known-history:** fails §14(a) at all three horizons on pooled WAPE; wins <50% of individual origins at every horizon. Currently the production forecaster — beaten by both Naive and Ridge under the corrected calendar-boundary protocol.
- **Random Forest — recursive:** fails more severely; recursive error compounding is a measured, monotonic cost.
- **Ridge — last-known-history:** ties Naive at +1 only (a documented numerical coincidence), then fails at +2/+3 — exactly the §13.1/§14 "good +1 masking poor +2/+3" failure.
- **Ridge — recursive:** fails more severely than last-known-history for the same recursive-compounding reason.

## 17. Per-horizon-selection consideration

ML Spec §14 permits a per-horizon decision (e.g. a different candidate at +1 vs. +2/+3) but prefers simplicity when one candidate passes at all horizons. Here, **no** horizon has evidence that a different candidate is superior to Naive — Naive is the pooled-WAPE minimum or tied-minimum at every individual horizon (§10–§12) and wins the majority of origins at every horizon in the fold-stability review (§9). There is therefore no evidentiary basis for per-horizon complexity, and introducing it would violate §14's simplicity preference with nothing to show for it. **Decision: a single strategy (Naive) applies uniformly to all three horizons.**

## 18. TRD compatibility implications

- **Categorizer:** TF-IDF + Logistic Regression fits `CategorizationService.predict(transaction) -> {predicted_category}` identically to K-Means — no schema/API change (ML Spec §5, verified drop-in compatible). It also genuinely produces class probabilities if a confidence field is ever added (not required now, TRD §11.2) — a capability K-Means's cluster-lookup does not have.
- **Forecaster:** Naive fits `ForecastService.run_forecast()`'s single-fit-per-call contract (TRD §12.3) trivially — it requires no fitting at all, only a lookup of the latest per-category monthly total, strictly cheaper than the currently-shipped Random Forest fit. **Explicit Naive-vs-RF/Ridge comparison for the TRD §12.3 interactive path:** all three are computationally trivial at this data scale (ML-B §30: RF/Ridge per-fold fitting completes in seconds for 14 folds × 2 models), so runtime is not the deciding factor for any of them — Naive wins on eligibility (§8), not on being the only one fast enough.

## 19. Multi-step strategy applicability/comparison

Strategy A (last-known-history) vs. B (recursive) was only a meaningful, evaluated axis for Random Forest and Ridge — both rejected (§16). For the selected candidate (Naive), the strategy distinction is not applicable: Naive always predicts the single most recent observed value for every horizon, and a "recursive" re-application of "repeat the last observation" produces the identical value at every horizon as the non-recursive version (documented in `ml/forecasting/baselines.py`'s own module docstring and ML-B §20).

**A/B evidence for the rejected candidates is preserved, not discarded:** for both RF and Ridge, Strategy B is monotonically worse than Strategy A at every horizon it diverges at (+2, +3) — e.g. Ridge: last-known-history +3 WAPE 0.2493 vs. recursive +3 WAPE 0.2887; RF: 0.2495 vs. 0.2771. Direct, quantified evidence that feeding predictions back into features accumulates error rather than correcting it, at this data scale. Retained for any future re-evaluation if RF or Ridge becomes competitive at a larger data scale.

## 20. Selected strategy

**N/A.** Recorded explicitly as not applicable in `reports/ml/ML_C_SELECTION_RECORD.json`, per the ML-C brief's instruction not to force an A/B choice onto a forecaster for which the distinction doesn't apply.

## 21. Complexity/runtime/maintainability tradeoffs

| | Categorizer (selected) | Forecaster (selected) |
|---|---|---|
| Fit cost | <1s at this data scale, TRAIN-only | none (no fitting) |
| Inference cost | one sparse matrix-vector product | one array lookup |
| Artifact | {vectorizer, model} — 2 objects, comparable to K-Means's existing 4-object artifact | none (no persisted artifact; a "recipe" only, matching ML Spec §18's asymmetry for forecasting) |
| Hyperparameters | `C=1.0, max_iter=1000` (both defaults) | none |
| vs. currently shipped | K-Means: comparable complexity, different (better-evidenced) model family | Random Forest: strictly simpler — Naive selection is a genuine reduction in production complexity |

## 22. Frozen pre-final selection record

`reports/ml/ML_C_SELECTION_RECORD.json`, written and frozen **after** Parts C (categorization selection), D (forecast model selection), and E (multi-step strategy selection) were all complete, and **before** Part F opened either FINAL component (confirmed by the file's own `note_on_writing_order` field and by the git-diff/file-timestamp record of this session). Contains: selected candidates, exact configurations, exact VALIDATION evidence cited, fold-stability evidence, §7/§8 and §14 gate applications, complexity/runtime considerations, evidence-tier limitations, rejected alternatives and why, git commit, and the required declaration:

> "Selection was finalized using TRAIN + VALIDATION evidence only. Neither component's FINAL data had been evaluated."

---

## 23. Categorization FINAL result

`reports/ml/results/final_categorization.json`, produced by `ml/categorization/run_final.py` — refuses to run unless the selection record names `tfidf_logreg` (verified: `tests/ml/test_ml_c_selection.py`).

**Label: "Tier B curated benchmark — held-out FINAL_TEST"** (never "real-world" or "temporal validation").

| Partition | n rows | n merchant groups |
|---|---|---|
| FINAL_TEST | 45 | 17 |

| Metric | VALIDATION (for reference) | **FINAL_TEST** |
|---|---|---|
| Macro F1 | 0.2552 | **0.4405** |
| Accuracy | 32.0% | **42.2%** |

Per-category FINAL F1 (support in parentheses): Other 1.000 (5), Healthcare 0.571 (5), Transport 0.364 (9), Food & Dining 0.345 (8), Subscriptions 0.400 (4), Shopping 0.400 (5), Rent & Utilities 0.222 (5), Entertainment 0.222 (4). **No category is at F1=0.0 on FINAL_TEST**, unlike VALIDATION's 3 zero-F1 categories.

`§18` metadata recorded in the artifact: `dataset_id`, `evidence_tier`, `split_definition_ref`, `selected_candidate`, `preprocessing_recipe` (exact TF-IDF/LogReg config), `model_impl_version: "tfidf_logreg_v1"`, `git_commit: 2c06181a12fb270e6e534564c98ccebd2998088c`, `evaluation_timestamp_utc`, and explicit `no_refitting_using_final_labels` / `no_mapping_using_final_labels` / `no_error_driven_model_modification_after_seeing_final` flags, all `true`.

## 24. Forecasting FINAL result

`reports/ml/results/final_forecasting.json`, produced by `ml/forecasting/run_final.py` — refuses to run unless the selection record names candidate `naive` and strategy `N/A`.

**Label: "Untouched temporal-test performance on reserved synthetic months"** (never "Tier B", "real-world", or "temporal validation" — an untouched period *was* feasible here and *was* used).

Reserved period: 2024-10, 2024-11, 2024-12. Trained on 2023-01 through 2024-09 (21 months), chronology preserved (`chronology_preserved: true`). 24 predictions (8 categories × 3 horizons).

| Metric | VALIDATION Naive (for reference, pooled) | **FINAL (reserved period)** |
|---|---|---|
| Combined WAPE | 0.1903 | **0.1887** |
| Combined MAE | 34.82 | **35.78** |
| +1 WAPE | 0.2020 | **0.1938** |
| +2 WAPE | 0.1904 | **0.1469** |
| +3 WAPE | 0.1765 | **0.2217** |

Notable per-category FINAL result: Rent & Utilities WAPE spiked to 0.765 (n=3, small sample) — the largest single-category contributor to FINAL's combined error; "Other" remains WAPE-undefined (all-zero actuals in this window too, consistent with the dev-region finding). Every category has only n=3 FINAL observations (one per horizon) — genuinely small-sample, disclosed rather than smoothed over.

`§18` metadata recorded: `dataset_id`, `evidence_tier`, `reserved_period`, `train_months_used`, `selected_candidate`, `selected_strategy: "N/A"`, `preprocessing_recipe`, `model_impl_version: "naive_v1"`, `git_commit`, `evaluation_timestamp_utc`, and an explicit list of the 5 rejected candidate/strategy combinations that were **not** evaluated on this reserved period.

## 25. Validation vs. FINAL comparison

**Forecasting:** FINAL combined WAPE (0.1887) is close to and slightly better than VALIDATION's pooled Naive WAPE (0.1903) — broadly consistent, no material degradation. Per-horizon, +1 and +2 are somewhat better on FINAL than VALIDATION (0.1938 vs 0.2020; 0.1469 vs 0.1904) while +3 is somewhat worse (0.2217 vs 0.1765) — plausibly explained by a single small-sample category (Rent & Utilities, WAPE 0.765 at n=3) dominating the +3 pool, not by any systematic +3-specific failure mode; every horizon has only n=8 FINAL observations, so no individual-horizon difference of this size is treated as more than directional.

**Categorization:** FINAL macro F1 (0.4405) and accuracy (42.2%) are **materially better** than VALIDATION (0.2552 / 32.0%) — a large jump for a 45-row FINAL_TEST partition. Plausible explanations, none confirmed: (a) small-sample variance — 45 rows / 17 merchant groups is a small evaluation set, and per-category support is as low as 4–5 rows for several categories, so a handful of easier merchant groups landing in FINAL_TEST by the frozen (seed-42) random assignment could swing the aggregate substantially; (b) FINAL_TEST's specific merchant groups may happen to include more categories with generic, TRAIN-vocabulary-adjacent language (the same mechanism identified in §5/ML-B §15) than VALIDATION's draw did. **This difference is reported, not explained away, and does not change the selection** (§22/§26 — FINAL is measurement, not tuning, and the ML Spec has no catastrophic-failure *or* catastrophic-improvement re-selection rule).

## 26. No reopening of model selection

Per the ML-C brief and ML Spec §20/§21: FINAL is a one-time measurement pass, not a tuning signal. The selection recorded in `ML_C_SELECTION_RECORD.json` was not revisited after either FINAL result was seen — both FINAL results (one moving in the "worse than expected" direction for forecasting's +3 horizon specifically, one moving in the "better than expected" direction for categorization overall) are reported exactly as computed. There is no catastrophic-failure rule in the frozen ML Spec, and none was invented here in either direction.

## 27. Evidence-tier limitations (restated precisely)

- **Categorization:** every number in this report and its artifacts describes performance on a Tier B (independently curated/constructed) 228-row, single-author benchmark. Never real-world (Tier A) evidence. FINAL_TEST specifically is 45 rows / 17 merchant groups — a small evaluation set; per-category FINAL support is as low as 4 rows.
- **Forecasting:** every number describes pipeline/mechanism behavior on `synthetic_24mo.csv` run through the production K-Means artifact. Never real-user spending forecast accuracy. The FINAL reserved period is 24 months of a single synthetic history's most recent 3 months — genuinely small-sample (n=8 per horizon, n=3 per category).

## 28. Exact claims ML-C supports

- Under a leakage-safe, merchant-grouped/category-stratified Tier B benchmark evaluation, TF-IDF + Logistic Regression achieves 44.1% macro F1 / 42.2% accuracy on a held-out FINAL_TEST partition (45 rows, 17 merchant groups) never consulted during model selection.
- Under the corrected calendar-month-boundary temporal protocol, a Naive (last-observed-month) forecaster achieves 0.1887 combined WAPE on a reserved, chronologically-later 3-month synthetic period never consulted during model selection, and no evaluated Random Forest or Ridge configuration (in either multi-step strategy) reliably beat Naive on VALIDATION at all three forecast horizons.
- Both selections were made from TRAIN+VALIDATION evidence only, with the selection frozen in a versioned, git-commit-stamped artifact before either FINAL partition/period was ever evaluated.

## 29. Claims ML-C still does NOT support

- That either result generalizes to real user data (both are Tier B/synthetic, §27).
- That TF-IDF + Logistic Regression's FINAL_TEST improvement over VALIDATION (0.44 vs 0.26 macro F1) reflects the model's "true" performance rather than small-sample variance in a 45-row partition — this has not been disambiguated and is not claimed as resolved.
- That Naive would remain the best forecaster at a materially larger data scale — §13's history-length finding (RF only overtakes Naive at 18 months, on 3 origins) is directional, not confirmatory, and is explicitly not treated as grounds to prefer RF now.
- That any specific numeric threshold ("beats naive by X%") was used for §14 eligibility — per the frozen spec's explicit instruction not to invent one, this was assessed qualitatively from the fold-level evidence in §9.
- That the categorization or forecasting selection is production-integrated — neither is (§30, §33).

## 30. Implications for ML-D

- ML-D would integrate `TfidfLogRegCandidate`'s exact TRAIN-fit recipe (`ml/categorization/candidates.py::TfidfLogRegCandidate`, `C=1.0, max_iter=1000, random_state=42`) behind the unchanged `CategorizationService.predict()` contract, producing/persisting a `{vectorizer, model}` artifact analogous to `kmeans_model.pkl`.
- **Flagged discrepancy for ML-D to resolve, not fixed here:** `TfidfLogRegCandidate` fits on merchant text (TF-IDF) only — it does **not** use `amount`/`day_of_week`/`is_weekend`, unlike K-Means's `build_feature_matrix`. This is a genuine feature-set difference between the current production categorizer and the ML-C-selected one; ML-D should decide (with its own evidence, not invented here) whether to extend the LogReg candidate to consume those features or ship it as evaluated (text-only).
- ML-D would replace the `n_estimators=100, max_depth=10...` Random Forest call inside `pipeline.forecast.train_and_predict()` with a Naive lookup — a strict simplification, removing the RandomForestRegressor dependency from the interactive path entirely for forecasting. `model_impl_version` would change from `"rf_v1_default_hparams"` to `"naive_v1"` (TRD §4.6/ML Spec §18).
- Both replacements are, by construction (ML Spec §5/§11's own compatibility analysis, reconfirmed in §18 above), drop-in behind the existing service contracts — no schema/API/frontend change is implied.

## 31. Open issues for ML-E

- Whether the categorization FINAL/VALIDATION gap (§25) reflects genuine small-sample noise or a real, generalizable pattern — would benefit from a larger or repeated Tier B benchmark before ML-E makes any FINAL-based claim.
- Whether a larger Tier B benchmark (or Tier A data) changes the LogReg-vs-SVM close call (§6) or the K-Means rejection.
- Whether more than 24 months of forecasting history would let RF (or another candidate) clear §14 at a data scale the current dataset cannot test (§13).
- Whether the "Other" category's persistent all-zero-spend behavior (an artifact of this specific K-Means run's cluster→category mapping, not a property of the category itself) should be revisited once the production categorizer changes (§30) — a new categorizer may route some spend to "Other," changing the sparsity picture for any future forecaster evaluation.
- Precisely how ML-E should phrase the resume/interview-safe versions of §28's claims, respecting §21's tier and horizon-specificity requirements.

---

# Interview & Deep-Dive Notes (ML-C additions)

This extends ML-B's working record — not a certified claim document (ML-E's job).

### Why the selected categorizer won
Logistic Regression won on the frozen primary metric (macro F1) and the secondary metric (accuracy) against its closest competitor (Linear SVM), and no complexity/runtime/probability-output consideration favored SVM enough to override that. The win margin (0.2552 vs 0.2405) is real but modest — this was treated explicitly as a close call, and the close-call analysis (§6) is preserved so a reader can judge for themselves whether they'd have weighed it the same way, rather than presenting the choice as more obviously decisive than the evidence supports.

### Why alternatives were rejected
K-Means's rejection is not close — it is statistically indistinguishable from chance once the exact leakage V1's own evaluation had (evaluating on merchants the mapping step had already seen) was removed. This is the single most important scientific finding to carry forward from ML-B into ML-C: a categorizer's headline number can look excellent (V1's 90%) purely because of a specific, nameable measurement flaw, and correcting that one flaw — nothing else — collapsed the number to noise. Linear SVM's rejection is a genuine close call, decided on the primary metric with no countervailing evidence, not a landslide.

### Why the selected forecaster won
Naive won because *nothing beat it*, not because it was expected to. §14 is explicitly an eligibility filter, not a "pick the lowest average WAPE" rule — a candidate has to clear Naive separately at +1, +2, and +3, with a margin not plausibly explained by fold noise. Every other candidate failed that test at least at +2/+3, several failed at every horizon, and the fold-level review (ML-C's own Part B contribution) showed those failures are the norm across origins, not a few unlucky ones.

### Why the selected strategy is N/A, not a forced choice
Once Naive was selected, "which multi-step strategy" stopped being a real question — Naive has no meaningful A/B distinction (repeating a fixed value recursively is definitionally identical to not doing so). Recording "N/A" rather than defaulting to "A" (which would have been trivial to write and easy to miss as meaningless) keeps the selection record honest about what was and wasn't actually evaluated for the shipped forecaster.

### Why §14 is an eligibility rule, not simply "lowest average WAPE wins"
If ML-C had used "lowest combined WAPE" alone, the conclusion (Naive wins) would have been the same here — but for the wrong reason, and it would have missed Ridge's dangerous +1-ties-Naive-then-loses-badly pattern, which a combined-only view would hide behind a deceptively close aggregate. The eligibility framing is what makes "Ridge looks fine at a glance" falsifiable per-horizon rather than assumed.

### What fold stability revealed
The pooled/aggregate numbers already told the main story (Naive wins), but the fold-level review answered a different, necessary question: *is that win fragile?* No — every rejected candidate loses to Naive in the majority of the 14 origins at every horizon, and the margin widens (not narrows) at +3. If the fold-level review had instead shown RF winning in, say, 12 of 14 origins at +2 despite losing on pooled WAPE (which can happen if a few large-magnitude origins dominate the pooled sum), that would have been a genuinely different, more nuanced finding worth surfacing — it did not happen here.

### What surprised us
Ridge's +1 prediction being numerically *identical* to Naive's (not just similar) is a specific, quantifiable coincidence, not a vague "roughly similar" — plausible mechanism: with only a few dozen TRAIN rows per fold, a linear model's fitted coefficients collapse toward "weight ≈1 on `lag_1_spend`, ≈0 on everything else," which is mathematically Naive. The categorization FINAL result being *better* than VALIDATION (rather than worse, which is the more commonly discussed failure direction) was the other surprise — worth flagging because it's tempting to only prepare a story for "FINAL disappoints," and this run is a reminder that small-sample partitions can swing either direction.

### Why simpler models may beat more complex ones
At this data scale (a few dozen TRAIN rows per categorization fold; a few dozen months per forecasting fold), a model's capacity to fit non-linear interactions or high-dimensional decision boundaries is not the bottleneck — data volume is. Random Forest's non-linear splits and SVM's margin-maximization both have room to overfit noise rather than signal when there are only a handful of examples per class/category, while Naive/Ridge/Logistic-Regression's comparative simplicity acts as an implicit regularizer. This is the concrete, measured instantiation of ML Spec §0's abstract principle, not a restatement of the principle on faith.

### How leakage changed the K-Means story
V1's headline 90% accuracy and ML-B/ML-C's measured ~12–44% (VALIDATION/FINAL) are not two different models — they are the *same* K-Means/mapping mechanism evaluated with and without merchant leakage. The lesson generalizes past this one model: a "held-out" evaluation set that shares underlying entities (merchants, users, whatever the natural grouping unit is) with training data measures memorization, not generalization, and the gap this can produce is not a small correction — it was ~78 percentage points of accuracy here.

### Why FINAL could not be used for model selection
If FINAL had been consulted while choosing between LogReg and SVM, or between Naive and Ridge, the resulting "selection" would really be reporting FINAL's own noise back to itself — the FINAL number would no longer be an honest estimate of out-of-sample performance for whichever candidate got picked, because the picking process would have used exactly the data being held out to estimate performance. This is why the selection record (§22) is written and frozen with a machine-checkable "before either FINAL was evaluated" declaration, and why the FINAL evaluators themselves (`ml/*/run_final.py`) structurally refuse to run without that record naming their own candidate.

### Why a poor FINAL result cannot trigger retuning
The same logic in reverse: if a disappointing FINAL number could send you back to try a different candidate or hyperparameter, FINAL stops being a held-out estimate and becomes just another validation fold that happened to run last — the next number you report would again not be honest. The frozen spec has no catastrophic-failure exception, and this report does not invent one even though it would have been easy to rationalize one for the +3 forecasting horizon's modest degradation.

### What it means if VALIDATION and FINAL differ
Both directions occurred in this single ML-C pass (forecasting: roughly consistent, with a directional +3 wrinkle explained by one small-sample category; categorization: a large, unexplained-but-disclosed improvement) — this is itself informative: it demonstrates the reports are not being selectively framed to make the story look cleaner than it is. A future reader should treat both differences as evidence of small-sample volatility at these evaluation-set sizes (45 categorization rows, 24 forecasting predictions), not as evidence about the underlying real-world quality of either model, because neither evaluation set is real-world data (§27).

### What the evidence still cannot establish
Real-world accuracy for either component (§29); whether the categorization FINAL/VALIDATION gap would persist or vanish with a larger benchmark; whether Naive remains best once genuinely multi-year transaction history exists; whether a categorizer that also used amount/day-of-week features (like the current K-Means, unlike the selected LogReg) would do meaningfully better or worse than the text-only LogReg evaluated here (§30's flagged ML-D question).

### Likely interviewer challenges
- *"You're shipping the simplest possible forecaster — doesn't that feel like giving up?"* — No: §14 is explicitly designed so that "ship the simplest thing that isn't beaten" is a legitimate scientific outcome, not a failure to try hard enough; the fold-level evidence (§9) shows the more complex candidates lose more often than they win, at every horizon, across 14 independent origins.
- *"Your categorization FINAL result jumped from 26–32% to 42–44% — doesn't that mean your VALIDATION estimate was just wrong?"* — It means the evaluation set (45–50 rows) is small enough that both numbers carry real sampling uncertainty; neither is "the" true number, and the honest answer is that more data (§27/§31) is needed to narrow that uncertainty, not that one number should be preferred over the other.
- *"Why not just retune Ridge/RF's hyperparameters instead of giving up on them?"* — Out of scope for ML-C by the frozen spec (no new hyperparameter search is permitted here, §0 of the ML-C brief) and not obviously going to help: the failure mode observed (losing at +2/+3, recursive compounding) looks like a data-scale limitation rather than a mistunable one, based on the monotonic degradation pattern in §9/§19.

---

*Reproduce every ML-C number in this report with `python -m ml.forecasting.fold_stability`, `python -m ml.categorization.run_final`, and `python -m ml.forecasting.run_final` from a clean checkout at the commit named at the top of this file, using `requirements.txt`'s pinned environment (`venv/`). `reports/ml/ML_C_SELECTION_RECORD.json` records the frozen pre-FINAL selection; `reports/ml/results/final_categorization.json` and `reports/ml/results/final_forecasting.json` record the two one-time FINAL passes.*
