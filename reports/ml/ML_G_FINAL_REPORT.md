# ML-G — Final Production Hardening Report

**Starting HEAD:** `88fe5ae` — *Fix HITL override semantics and demo re-entry*

This phase set out to fix a reported production failure — real RBC and
Scotiabank imports collapsing into one category — and to finish the product
around it. Everything below is drawn from the repository: committed
evaluation records, test output, and diagnostics run against the actual
database, not from recollection.

---

## A. Starting state

**Architecture.** FastAPI + SQLite backend (routes → services →
repositories, one shared connection, numbered SQL migrations); React 19 +
TypeScript + Vite + Tailwind + shadcn/Radix + Recharts + TanStack Query
frontend; a `pipeline/` package carrying V1's ingestion and forecasting code;
an `ml/` package holding evaluation harnesses; `reports/ml/` holding frozen
selection records.

**Bank support.** RBC, Scotiabank, TD, CIBC — strict column-fingerprint
detection, fail-closed. BMO and National Bank visible but disabled.

**Production categorizer before this phase.** `models/tfidf_logreg_v2.pkl`:
word TF-IDF, `max_features=200`, `ngram_range=(1,2)`, `sublinear_tf`, feeding
`LogisticRegression(C=1.0)`. Fit on `deployment_benchmark_v1` TRAIN (96 rows
/ 41 merchant groups). Frozen in `ML_F_SELECTION_RECORD.json`.

**Previous dataset.** `deployment_benchmark_v1`: 190 rows, 73 merchant
groups, split 60/20/20 by merchant group.

**Previous metrics (sealed FINAL_TEST, `ML_F_SELECTION_RECORD.json`).**
accuracy 0.308, **macro-F1 0.174**, n=39. Four of eight categories scored
F1 = 0.000 (Entertainment, Healthcare, Other, Transport).

**Previous forecaster.** 3-month rolling mean
(`ml/forecasting/baselines.py::rolling_mean_predict`, window=3), selected in
ML-F on pooled validation WAPE. Eligibility gated at 6 completed months.

**Analytics before this phase.** Dashboard only: current-vs-previous month
totals, current-month category breakdown, a 6-month total-spend trend, and
five recent transactions.

---

## B. Problems found

### B1. Preview and Confirm disagreed — confirmed, and it was structural

`IngestionService.parse_and_stage()` staged the raw model output.
`commit_import()` then *independently* applied (a) structural-ambiguity
routing to "Other" and (b) a correction-memory lookup. Two code paths, one
decision. The category shown in the Preview table was therefore **not** the
category persisted on confirm — and the divergence was concentrated on
exactly the rows where being wrong is most visible: ambiguous rows, and
merchants the user had already corrected.

### B2. Food & Dining collapse — root cause measured, not inferred

Probing the shipped artifact with 18 realistic deployment-shaped strings:

- **11 of 18 produced an all-zero feature vector.**
- `LogisticRegression.predict` on an all-zero row returns
  `argmax(intercept_)` — one fixed class, deterministically, for every
  evidence-free input.
- On this artifact that class was **Food & Dining**.

This is a representation-coverage bug. No hyperparameter could have fixed it.
Observed collapses included `NETFLIX COM`, `TIM HORTONS 4521`, `SHELL 1234`,
`CVS PHARMACY`, `UBER TRIP`, `WALMART SUPERCENTRE`, `STARBUCKS`,
`ADOBE SUBSCRIPTION`, `LOBLAWS` — all zero-feature, all Food & Dining.

### B3. The corpus made generalization impossible by construction

`deployment_benchmark_v1` gave each merchant group a **single** merchant
name, and no descriptive word was shared between groups. The split is
merchant-group isolated — correctly — so a held-out group's words had
**never** appeared in TRAIN. A bag-of-words or char-n-gram model had
literally no feature in common with the row it was being asked to classify.
The reported 0.174 was the honest ceiling of that design.

### B4. Vocabulary capacity was a real but secondary problem

`max_features=200` over a 96-row TRAIN partition. Raising the cap alone
(measured this phase as candidate G2) helps — 0.361 → 0.505 validation
macro-F1 — but does not approach what the representation change achieves.

### B5. Structural ambiguity was over-routing legitimate rows

`ambiguity.py` matched three bare regexes (`\bE-?TRANSFER\b`,
`\bABM WITHDRAWAL\b`, `\bATM WITHDRAWAL\b`) against raw text. So
`E-TRANSFER SENT MAPLEWOOD DINER REF44120` — payment boilerplate wrapped
around a perfectly usable merchant identity — was routed to "Other" and
**never shown to the classifier at all**. Measured on the new benchmark's
FINAL_TEST partition, that rule fired on **27 of 195 (13.8%)** of legitimate
rows. This is the "over-route everything to Other" failure the brief warns
against, and it silently destroyed categorization for an entire payment rail.

### B6. Correction memory was inert on real data

`find_latest_confirmed_category(merchant, bank_source)` matched the **exact**
`merchant` string. Real bank descriptions embed a per-transaction card
suffix, store number or reference code:

```
VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY
VISA DEBIT PURCHASE - 9137 CAREWELL PHARMACY
CAREWELL PHARMACY #0284
```

Three distinct strings, one merchant. The exact match essentially never
fired. A user could correct the same merchant every month forever and the
system would never learn.

### B7. Train/serve skew was baked into the artifact contract

The payload was `{vectorizer, model, model_impl_version, metadata}`.
`CategorizationService` vectorized the raw `merchant` column with **no text
normalization**, regardless of what the winning recipe had been fit with. A
normalizing candidate could therefore never have been served correctly even
if it had won — and ML-F's candidate C was exactly such a candidate.

### B8. Boilerplate could act as a category shortcut (introduced and caught
*within* this phase)

The first v2 corpus draft assigned each merchant archetype its own template
mix (card rails for restaurants, PREAUTH for utilities, ONLINE for
subscriptions). Measurement caught it immediately: stripping boilerplate made
validation macro-F1 **fall** (0.375 → 0.299) — the signature of a model
leaning on transaction-method words rather than merchant identity. Recorded
here because it changed the corpus design, and because a benchmark inflated
by that shortcut would have overstated the final numbers.

### B9. Smaller findings

- `ForecastService.MONTHS_REQUIRED` (6) and `pipeline.forecast`'s gate (6)
  were duplicated constants with no test binding them together.
- Demo seed reported `months_required = MONTHS_OF_HISTORY` (12) rather than
  the gate the app actually enforces.
- Import copy said "Valid rows" and "Credits excluded" — parser facts, not
  product ones.
- The How It Works page displayed superseded ML-F numbers as current.

---

## C. Categorization techniques tried

All fit on `deployment_benchmark_v2` TRAIN only (580 rows / 119 merchant
groups); all scored on VALIDATION (195 rows / 40 groups). Every candidate
carried a written hypothesis before it was run
(`ML_G_SELECTION_RECORD.json > hypotheses`). Classical ML only — no neural
networks, embeddings, transformers, LLMs or external lookups.

| # | Technique | Representation | Model | Val macro-F1 | Zero-feature | Verdict |
|---|---|---|---|---|---|---|
| G6 | **Word + char union, wide** | word 1–2 unbounded ∪ char_wb 2–6 ×8000, normalized | LogReg C=1 | **0.6659** | 0.0% | **SELECTED** |
| G14 | Same, class-balanced | as G6 | LogReg, `class_weight=balanced` | 0.6636 | 0.0% | Within tie epsilon; simpler model kept |
| G13 | Same, C=4 | as G6 | LogReg C=4 | 0.6527 | 0.0% | Regularization was not the bottleneck |
| G5 | Word + char union, narrow | word 1–2 ∪ char_wb 3–5 ×3000 | LogReg C=1 | 0.6461 | 0.0% | Union confirmed; wider char range better |
| G10 | Complement NB | as G5 | ComplementNB α=0.3 | 0.6404 | 0.0% | Competitive; no usable margin for abstention |
| G3 | Word only, normalized | word 1–2 unbounded, normalized | LogReg C=1 | 0.6315 | **40.5%** | Rejected on coverage, not score — see below |
| G8 | Narrow union, C=4, balanced | as G5 | LogReg C=4 balanced | 0.6087 | 0.0% | — |
| G11 | Multinomial NB | as G5 | MultinomialNB α=0.3 | 0.6077 | 0.0% | Control for G10; confirms complement helps |
| G7 | Narrow union, balanced | as G5 | LogReg balanced | 0.6032 | 0.0% | — |
| G12 | Unigram + char | word 1–1 ∪ char_wb 3–5 | LogReg balanced | 0.5968 | 0.0% | Bigrams do carry signal after all |
| G9 | Linear SVM | as G5 | LinearSVC balanced | 0.5891 | 0.0% | Behind LogReg; no calibrated probabilities |
| G4 | Char only | char_wb 3–5 ×3000, normalized | LogReg C=1 | 0.5524 | 0.0% | Robust but blunt without word head nouns |
| G2 | Word only, unbounded | word 1–2 unbounded | LogReg C=1 | 0.5054 | 2.6% | Vocabulary cap was real but secondary |
| G1 | **Control: the shipped ML-F recipe** | word 1–2 ×200 | LogReg C=1 | 0.3608 | 5.6% | Isolates corpus effect from model effect |

**G3 is the most instructive rejection.** Ranked sixth on macro-F1 — a
respectable score — while producing **no features at all** for 40.5% of
held-out rows. Every one of those receives `argmax(intercept_)`: the exact
production failure this phase existed to remove. Selecting on accuracy alone
would have shipped the same bug with a better-looking number. This is why
representation coverage is reported alongside every candidate.

**G1 is the control that makes the rest interpretable.** The shipped recipe,
refit on the new corpus, scores 0.361 against its own 0.174 on the old one.
Better data roughly doubled it; the representation change roughly doubled it
again.

### Selection protocol

1. Freeze one merchant-group-isolated, category-stratified split.
2. Fit on TRAIN only; measure on VALIDATION only.
3. Select on VALIDATION macro-F1, tie-broken within 0.01 toward the simpler
   representation, then LogReg > LinearSVC > Naive Bayes.
4. **Then** fit the abstention policy, on VALIDATION only — it could not
   change which model won.
5. Evaluate the frozen (model + policy) on the sealed FINAL_TEST **once**.
6. Report older corpora as continuity only.

### The abstention policy

Two rules, both deterministic:

- **Evidence:** zero active features → abstain. Unconditional.
- **Margin:** top-vs-runner-up probability gap < `0.02` → abstain.

Abstaining means `predicted_category = "Other"`, `confirmed_category = NULL`
— a system decision, never a human one.

*Margin, not absolute confidence, and that choice is evidence-driven.* The
selected union is globally under-confident (mean top score 0.34 across eight
classes). At a top-score threshold of 0.20 it abstained on 28% of validation
rows to rescue 31 wrong predictions at a cost of 24 correct ones — a bad
trade. The margin is scale-relative and separates "confidently first" from
"tied with the runner-up" without depending on calibration.

*Selection was deliberately not a macro-F1 optimization.* Abstaining routes
rows to "Other", mechanically diluting that class's precision, so macro-F1
penalizes abstention even where abstaining is obviously right — optimizing it
would select "never abstain" for the wrong reason. The pre-registered rule
instead took the largest threshold that abstains on ≤15% of validation rows
**and** rescues ≥2× as many wrong predictions as the correct ones it costs.

At `min_margin = 0.02` on VALIDATION: 24 rows abstained (12.3%), **17 wrong
predictions avoided, 7 correct ones given up** (2.4:1). Cost: 0.042 macro-F1.
Both numbers are reported, and the cost is stated on the How It Works page.

---

## D. Dataset changes

| | v1 (ML-F) | v2 (ML-G) |
|---|---|---|
| Rows | 190 | 970 categorical + 32 ambiguous |
| Merchant groups | 73 | 199 |
| TRAIN | 96 rows / 41 groups | 580 rows / 119 groups |
| VALIDATION | 41 rows / 16 groups | 195 rows / 40 groups |
| FINAL_TEST | 39 rows / 16 groups | 195 rows / 40 groups |

**The change that mattered: head-noun redundancy.** Real merchant descriptors
generalize because businesses in a category share a head noun — PHARMACY /
DENTAL / CLINIC / OPTICAL; TRANSIT / TAXI / PARKING / FUEL; HYDRO / INTERNET
/ MOBILE. A human places an unseen "X PHARMACY" from that noun, not from
having memorized X. v1 gave each head noun to exactly one group, so the
split removed it entirely. v2 carries each head noun across 2–4 *different*
fabricated brands, so a held-out "CEDARVALE PHARMACY" has unseen brand words
but a seen head noun.

**This is not label leakage.** The head noun is a genuine property of a
merchant's name, not a copy of the label. No category name from the taxonomy
("Healthcare", "Rent & Utilities", …) appears in any description — asserted
by `tests/ml/test_mlg_categorization.py::test_no_category_label_leaks_into_any_description`.

**One shared template pool for every merchant.** See B8. Boilerplate is now
spread evenly across all eight categories and carries no category signal.

**Honest difficulty retained.** ~2 groups per category are deliberately
brand-only, with no descriptive word ("ZENOVARA", "KESSLIN & CO"). No text
classifier can place these. They are in the corpus so the reported numbers
are not flattering, and so abstention has real cases to be measured on.

**Split methodology and leakage checks.** Unchanged and reused
(`ml/common/splitting.py`): merchant-grouped, category-stratified, seeded.
Verified structurally by `verify_split_isolation`, and asserted by
`test_split_isolates_merchant_groups_across_every_partition`. Head-noun
redundancy itself is asserted by
`test_corpus_shares_head_nouns_across_merchant_groups`, so a future edit
cannot silently revert the corpus to v1's failure mode.

**Why FINAL_TEST is legitimately new.** The v2 corpus is a different set of
merchant groups; the v1 assignment cannot be applied to it at all. The v2
FINAL_TEST had never been evaluated before step 5 of the protocol. The v1
split file and results are untouched on disk as ML-F evidence.

**Privacy.** Every merchant name is fabricated. No literal or paraphrased
private transaction description, account number, balance or personal name
appears anywhere in the corpus or its builder. Only the *structural*
templates derive from the real-export audit, and those were already
established in v1.

---

## E. Final categorizer

**Pipeline.** cleaned `merchant` → `normalize_deployment_text_v2` (strips
transaction-method boilerplate, reference/transaction-id tokens, bare numeric
card/store suffixes) → `FeatureUnion(word TF-IDF 1–2 grams unbounded,
min_df=1, sublinear_tf; char_wb TF-IDF 2–6 grams, max_features=8000,
sublinear_tf)` → `LogisticRegression(C=1.0, max_iter=2000, random_state=42)`
→ abstention policy.

**Artifact.** `models/categorizer_v3.pkl`, `model_impl_version =
tfidf_word_char_logreg_v3`, vocabulary 8,524, fit on TRAIN only (580 rows /
116 merchant groups after ambiguous rows are excluded). Built by
`scripts/build_production_categorizer.py`, which refuses to run unless
`ML_G_SELECTION_RECORD.json` names a reconstructible winner **and** records
an abstention policy.

**The artifact now carries its own decision contract** — `normalizer_name`,
`min_margin`, `abstain_category`, `categories` — so what is served is what
was evaluated. `test_production_artifact_matches_the_frozen_selection`
asserts artifact and record agree.

**Metrics.**

| | Value |
|---|---|
| VALIDATION macro-F1 | 0.6659 |
| Sealed FINAL_TEST macro-F1, model only | **0.5931** |
| Sealed FINAL_TEST macro-F1, with policy | **0.5762** |
| Sealed FINAL_TEST accuracy, with policy | 0.5949 |
| Previous sealed FINAL_TEST macro-F1 | 0.1741 |
| Zero-feature rate on FINAL_TEST | **0.0%** |
| Mean active features per row | 54.6 |

Per-class on the sealed test (with policy):

| Category | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Food & Dining | 0.711 | 0.914 | 0.800 | 35 |
| Subscriptions | 1.000 | 0.650 | 0.788 | 20 |
| Healthcare | 0.727 | 0.640 | 0.681 | 25 |
| Transport | 0.531 | 0.680 | 0.596 | 25 |
| Shopping | 0.565 | 0.520 | 0.542 | 25 |
| Rent & Utilities | 0.818 | 0.360 | 0.500 | 25 |
| Other | 0.267 | 0.600 | 0.369 | 20 |
| Entertainment | 1.000 | 0.200 | 0.333 | 20 |

No class is at zero. Subscriptions and Entertainment have perfect precision
with poor recall — when the model commits it is right, but it often declines.
"Other" has the inverse shape because every abstention lands there, which is
the trade the policy makes deliberately.

**Continuity (recipe refit on older corpora, never used to select):**
Tier B — VALIDATION 0.292 / FINAL_TEST 0.583. deployment v1 — VALIDATION
0.210 / FINAL_TEST 0.108. The low v1 numbers are the corpus diagnosis
restated: the recipe is not magic, and v1 left nothing to generalize from.

### Production diagnostics on real data

`scripts/diagnose_production_inputs.py` runs read-only over the actual
database and reports aggregates only — no description, amount, merchant or
date is ever printed. Against the developer's real RBC/Scotiabank data
(29 stored rows):

| Metric | Value |
|---|---|
| Zero-feature rows | **0 (0.0%)** |
| Weak (1–2 features) | 0 (0.0%) |
| Mean active features | 26.5 |
| Structurally routed to Other | 7 (24.1%) |
| Abstained (low margin) | 6 of 22 model-eligible (27.3%) |
| Largest served category | Other, 48.3% |
| Food & Dining share | 41.4% |
| Rows with a stable merchant key | 22 (75.9%) |

**No accuracy claim is made or possible** — these rows have no ground-truth
labels. What is claimed is narrow and verifiable: the zero-feature pathology
is gone, and Food & Dining is no longer a universal sink.

**Remaining weakness, stated plainly.** The mean top-vs-second margin on this
real sample is 0.035 — low. Real Canadian brand names ("TIM HORTONS",
"PRESTO", "PETRO-CANADA") are not descriptive, and a classifier trained on a
fabricated corpus of descriptive names cannot reliably place them. Food &
Dining at 41% on 29 rows is a small sample, but it is also honest: unseen
brand names are the dominant residual weakness, and correction memory —
now actually functional — is the mechanism that addresses them.

---

## F. Human-in-the-loop

**One decision path.** `backend/services/category_decision.py`, ordered:

1. structural ambiguity → "Other"
2. model + abstention → category or "Other"
3. correction memory → `confirmed_category`

`decide()` and `decide_batch()` are asserted to agree row-for-row
(`test_decide_batch_matches_decide_row_by_row`), because two implementations
of one decision is the bug this module removes.

**Preview** runs the shared path and persists its complete result to
`staged_transactions` (migration 004: `merchant_key`, `remembered_category`,
`decision_source`, `model_category`). It is read-only: it creates no
transactions, seeds no memory, and does not change app mode — asserted by
`test_preview_creates_no_transactions_and_seeds_no_memory`.

**Confirm** re-validates rather than re-decides. Two things *are* re-checked
live, because they can genuinely change between preview and confirm:
duplicate status, and correction memory (the user may have corrected that
merchant in between — the newer human decision wins).

`test_every_previewed_row_stores_the_category_it_showed` asserts
`predicted_category`, `confirmed_category` **and** `effective_category` agree
between the staged preview and the stored transaction, for every row.

**HITL semantics preserved.** Steps 1 and 2 write only `predicted_category`.
Auto-"Other" can never seed correction memory, and a row nobody looked at can
never read as a manual override.

**Stable merchant identity.** `backend/services/merchant_identity.py`:
identity tokens surviving boilerplate/reference-noise removal, sorted, joined,
scoped by bank. Deterministic; no fuzzy matching, no edit distance, no
embeddings, no external lookup. Two merchants match only if their token
*sets* are equal.

Critically: a description naming nothing yields **no key at all** (`None`), so
generic transfers can never collapse into one shared memory entry and teach
the system a category for "transfers in general" —
`test_generic_transfers_do_not_collide_in_memory`. Bank isolation is
asserted by `test_correction_memory_is_isolated_per_bank`.

Migration 003 adds `transactions.merchant_key` with an index;
`backend/db/backfill.py` populates it idempotently at startup for rows that
predate the column.

**Other-routing rewritten.** Two-part rule: a structural marker must be
present **and** no merchant-identity token may survive normalization.
Coverage 100%, false-positive routing **0.0%** (was 13.8%).

---

## G. Forecast

**Method:** unchanged — 3-month rolling mean, the ML-F selection. Not
reopened; the repository confirms it is the intended production method.

**Minimum: 3 completed months.** Every runtime gate changed together:

- `backend/services/forecast_service.py::MONTHS_REQUIRED` — 6 → 3
- `pipeline/forecast.py::MONTHS_REQUIRED` (new named constant, used by
  `aggregate_monthly()`) — 6 → 3
- `backend/services/demo_seed_data.py` — reported 12, now mirrors the real gate

`test_both_month_gates_agree` asserts the two constants are equal, because a
silent divergence produces a 500 from a request the service layer already
accepted.

**Tests:** 0 / 1 / 2 months unavailable, 3 months available
(`test_eligibility_boundary_at_three_months`); generation rejected below 3
and nothing persisted; generation succeeds at exactly 3.

**Numerical:** `300, 450, 600 → 450.00`, identically across all three
horizons (`test_three_month_mean_is_exactly_the_mean_of_the_last_three_months`),
plus a determinism test.

**Effective category:** `test_forecast_aggregates_on_effective_category`
proves a correction moves the forecast, and that a category with no spend
reports `is_available: false` rather than a fabricated $0.

**Stale → refresh:** correcting a category marks the run stale; regenerating
clears it, produces a new run id, and reflects the correction.

**Wording, exactly.** Three months is the **mathematical minimum** of the
selected method — one full window. It is **not** established that three
months forecasts as accurately as 6/9/12/18; those experiments never tested
a three-month history. This is stated in the code comments, the UI cold-start
copy, the How It Works forecast section, the PRD/TRD amendments, and the
"claims we do not make" list.

---

## H. Analytics

### Considered

| Graph | User question | Decision | Reason |
|---|---|---|---|
| Category trend over time | Which categories are growing? | **Implemented** | Highest-value question the stored data can answer |
| Top merchants | Where specifically does my money go? | **Implemented** | Actionable; only became viable once `merchant_key` existed |
| Month-over-month movers | Why did I spend more this month? | **Implemented** | Additive decomposition — an explanation, not another chart |
| Cumulative pace vs. prior month | Am I on track? | **Implemented** | The one genuinely *current* question a dashboard should answer |
| Forecast vs. actual | Were past forecasts any good? | **Implemented, honestly gated** | Real snapshots exist over time; see below |
| Daily/weekly pattern heatmap | When do I spend? | Rejected | Weak actionability for personal finance; would be decoration |
| Transaction-size distribution | How are my purchases sized? | Rejected | Interesting once, rarely twice |
| Recurring/subscription detection | What am I subscribed to? | Deferred | Needs recurrence inference the data does not yet support honestly |
| Category volatility | Which categories are unpredictable? | Rejected | Statistically thin at 3–12 months of history |
| Merchant concentration index | Is my spending concentrated? | **Folded into Top merchants** | One number (`top_n_share_pct`) rather than its own chart |
| Forecast error over time | Is the forecast improving? | Deferred | Needs many more genuine snapshots than any user will have yet |

### Implemented

| Graph | Page | Source | Aggregation | Time control | Empty state |
|---|---|---|---|---|---|
| Spending pace | Dashboard | `/api/analytics/spend-pace` | Cumulative by day-of-month, current vs previous | fixed (this vs last month) | "Not enough history yet" |
| What changed this month | Dashboard | `/api/analytics/category-movers` | Per-category Δ, sums to total Δ | fixed | "Nothing to compare yet" |
| Category trend | Transactions → Insights | `/api/analytics/category-trend` | Monthly sum per category, zero-filled | 6 / 12 / 24 months | "No spending in this period" |
| Top merchants | Transactions → Insights | `/api/analytics/top-merchants` | Grouped by `merchant_key` | 3 / 6 / 12 months | "No merchants in this period" |
| Forecast accuracy | Forecast | `/api/analytics/forecast-accuracy` | Genuine snapshots vs actuals, WAPE | derived | "No forecast history yet" + why |

**Every one groups by `effective_category`.** A correction propagates into
every chart the moment it is saved; all analytics queries share one
invalidation key so they can never disagree on screen.

**Forecast-vs-actual is honest by construction.** A prediction counts only if
its run was generated **strictly before** the first day of the month it
predicted, for a month that has since completed. Old predictions are never
recomputed against present-day data and presented as history —
`test_forecast_accuracy_ignores_hindsight_predictions` asserts exactly that.
Until a real snapshot exists the card says so, and names why. It resolves on
its own: generate a forecast, let a month pass, real evidence appears.
Forecast runs are retained indefinitely, so the snapshots do accumulate.

**Current-month spend beyond today is `null`, not zero** — a genuine gap in
the line rather than a flat tail reading as "spent nothing since".

---

## I. UI/UX

**Pages changed:** Dashboard (2 new cards), Transactions (tabbed list /
insights), Forecast (accuracy card, honest cold-start copy), Import (rewritten
preview), How It Works (rebuilt).

**New components:** 5 analytics cards + shared chart primitives; 7 How It
Works sections; a tab bar; a segmented control.

**Import copy** — the requested product-semantics fix, plus the follow-through:

| Before | After |
|---|---|
| Valid rows | **Spend rows to import** (net of duplicates, emphasized) |
| Duplicates | **Already imported** |
| Credits excluded | **Credits / inflows skipped** |
| Unparseable | Unparseable *(kept — genuinely malformed)* |
| "Confirm import" | **"Import 8 transactions"** |

The preview table now shows *why* each category was chosen: a remembered
correction renders as `Healthcare → Shopping · Your category`; a system
"Other" is annotated "no merchant name" or "low confidence".

**Accessibility.** ARIA tabs with roving arrow-key focus and one tab stop;
`radiogroup`/`radio` for segmented controls (arrow-key navigation for free);
`role="img"` + `aria-label` on every decorative bar so a screen reader gets
the number; `<caption class="sr-only">` and `scope="col"` on the preview
table; visible focus rings throughout; `role="alert"` on the import error.

**Motion.** Framer Motion, subtle and fast (0.18–0.4s): staggered card
entrances, tab crossfades, bars growing from zero, walkthrough slides.
**Every animation checks `useReducedMotion()`** and degrades to a static
final state — including Recharts' `isAnimationActive`.

**WebGL/Vanta: considered and deliberately declined.** A GPU hero on the How
It Works page would have added a dependency and a GPU budget to add
atmosphere to a page whose job is to be read and trusted. A two-stop CSS
radial gradient achieves the same visual lift for nothing. The brief's own
guidance applies: if the simpler implementation looks equally good, use it.

**Responsive.** Verified in Playwright at 390×844 across Dashboard,
Transactions, Forecast, Import and How It Works, asserting
`scrollWidth - clientWidth ≤ 2px` — wide content scrolls inside its own
container, never the page.

**One layout bug found and fixed by inspection in the browser.** The sticky
section rail sat 24px below the viewport edge, leaving a strip of scrolled
content visible above it. `AppShell`'s `<main>` is the scroll container and
carries `p-6`, and a sticky offset resolves against that padded content box —
so `top-0` is not the top. Fixed with a negative offset matching the
container's padding, and the reason documented at the call site.

**Dependencies added: none.**

---

## J. How It Works

Rebuilt from tab panels to **one scrolled narrative with a sticky rail**. The
sections build on each other — premise → walkthrough → pipeline → evidence →
limitations — and tabs actively hid that ordering by making every section
look like a peer you might skip.

1. **What is PlainCents?** — a plain-language premise, and an equally
   prominent "what it deliberately doesn't do" column. For a finance tool
   that is half the answer, not fine print.
2. **Using the app** — a 10-step stepper through the real workflow, with
   schematic screen miniatures. Schematic on purpose: a screenshot is stale
   the first time a button moves.
3. **Video walkthrough** — a real player, wired and waiting. No recording
   exists in the repository, and the component does not pretend otherwise: it
   probes `/media/plaincents-walkthrough.mp4` on mount (HEAD + content-type,
   so the SPA fallback's HTML 200 isn't mistaken for a video) and shows an
   explicit "not recorded yet" state naming the drop-in path. Poster support
   is optional and detected the same way. Served from this app; no third
   party.
4. **Categorization** — the existing pipeline diagram plus a new interactive
   journey: pick one of four row shapes and follow it from raw bank text
   through normalization, ambiguity check and classifier to two
   *side-by-side, differently-coloured* columns — `predicted_category` and
   `confirmed_category` — and the effective value below. The system/human
   split is the visual point.
5. **Your corrections** — a four-beat timeline over one merchant: prediction
   → correction → reuse on a different card number → change of mind. The
   model's answer visibly survives every step.
6. **Forecasting** — the arithmetic done live, with three presets. The "one
   big month" preset exists to show the method's weakness rather than hide
   it.
7. **Evidence** — leads with the *sealed test* score, not the score the model
   was chosen on. States on-card that the corpus is fabricated. All 11
   configurations, their hypotheses and their outcomes behind disclosures.
8. **Limitations** — seven plainly-stated limits, plus six claims PlainCents
   explicitly does **not** make, rendered rather than hidden.

Stale ML-F components and data modules were **deleted**, not left unrendered:
an unrendered module full of superseded numbers is a trap for whoever opens
the directory next. The decisions they documented remain in
`reports/ml/ML_C_*` and `ML_F_*`, which is where historical evidence belongs.

---

## K. Tests

| Suite | Result |
|---|---|
| `pytest` | **448 passed** (was 344) |
| `tsc -b` (typecheck) | **clean** |
| `oxlint` | **exit 0** (17 warnings, all pre-existing shadcn conventions) |
| `vitest` | **70 passed** / 14 files (was 60) |
| `vite build` | **clean** |
| Playwright | **25 passed** / 6 flows (was 13 / 5) |

New coverage: the shared decision path and its batch/single equivalence;
Preview↔Confirm agreement on all three category columns; Preview read-only;
correction memory across card-suffix variants, bank isolation, collision
safety, auto-Other never seeding; structural routing coverage *and*
false-positive rate; artifact-matches-selection; corpus head-noun redundancy
and label-leak checks; abstention policy in both directions; forecast
0/1/2/3-month boundary, arithmetic, determinism, gate agreement,
effective-category aggregation, stale→refresh; all five analytics endpoints
(aggregation, effective category, additivity, date ranges, demo/real
isolation, empty states); the re-categorization script's never-touch-a-
correction guarantee; How It Works structure, walkthrough, video placeholder,
deep links; and a mobile no-horizontal-overflow sweep.

**Two test-infrastructure notes, stated rather than buried.** Vitest's 5s
default timeout was raised to 20s: several tests pass comfortably in
isolation but intermittently timed out under full-suite parallel load on this
machine. And the Playwright e2e global setup was pointed at the new artifact
path and taught to find `.venv` as well as `venv` — it had been silently
provisioning the superseded model.

---

## L. Final product

**Categorization:** bank CSV → detection → normalization → structural
ambiguity check → word+char TF-IDF → logistic regression → abstention →
correction memory → effective category. One path, used by both Preview and
Confirm.

**Forecasting:** effective-category monthly totals → 3-month rolling mean →
three horizons, available after three completed months, stale-marked on any
change that moves the inputs.

**Analytics:** spending pace, month-over-month movers, category trend, top
merchants, forecast accuracy — all on `effective_category`, all live SQL, all
with real empty states.

**Remaining limitations** (all surfaced in the product, not just here):
fabricated evaluation corpus; unseen non-descriptive brand names remain the
weak point; no ground truth on private exports; no online learning; no income
tracking; no bank connection; the forecast is a simple average.

### One thing to run before smoke testing

Existing transactions keep the decision made when they were imported — that
is correct, and re-writing them silently on startup would not be. On the
developer's real database, 17 of 29 stored rows would change under the new
pipeline. To adopt it on rows never corrected:

```bash
python -m scripts.recategorize_stored_transactions          # preview
python -m scripts.recategorize_stored_transactions --apply  # commit
```

It rewrites `predicted_category` and refreshes `merchant_key`, **never**
`confirmed_category`, and marks any existing forecast stale.

### Recommended screenshots

1. Dashboard — spending pace + what-changed side by side
2. Transactions → Insights — category trend (stacked) + top merchants
3. Import preview — the stat row and a remembered-correction row
4. How It Works — the premise hero
5. How It Works → Categorization — the two-column system/human split
6. How It Works → Evidence — the three headline metrics
7. Forecast — populated, with the accuracy card's honest empty state

### Manual Render smoke checklist

1. Deploy; confirm startup logs show migrations 001–004 applied and the model
   loaded with `model_impl_version=tfidf_word_char_logreg_v3`,
   `normalizer=normalize_deployment_text_v2`, `min_margin=0.02`.
2. `GET /api/health` → categorization status `loaded`.
3. Dashboard in EMPTY → onboarding, no charts drawn from nothing.
4. Load demo → pace and movers populate; Insights tab populates; Forecast
   shows a run and the accuracy card's honest empty state.
5. Clear demo → EMPTY; load demo again → works (re-entry).
6. Import a real CSV per bank (RBC, Scotiabank, TD, CIBC). Confirm the
   preview says "Spend rows to import" and "Credits / inflows skipped".
7. **Note a preview row's category, confirm, and check it stored that exact
   category.**
8. Correct a category on Transactions; confirm the charts move and the
   forecast goes stale.
9. Import a later statement containing that merchant; confirm the preview
   shows "Your category" before you confirm.
10. Confirm a generic e-transfer / ATM row imported as Other with no manual
    override flag.
11. Regenerate the forecast; confirm staleness clears.
12. How It Works: every rail link scrolls; walkthrough steps; video shows the
    placeholder; forecast presets recompute.
13. Resize to a phone width on every page; confirm no horizontal scrollbar.
14. Run `python -m scripts.diagnose_production_inputs` and confirm a 0%
    zero-feature rate.

---

## Final question

> Is PlainCents now the strongest defensible version of the product within
> its classical-ML spending-intelligence premise and ready for final Render
> smoke testing and permanent engineering freeze?

**YES.**

The reported production failure was diagnosed to a specific mechanism,
fixed at all three of its causes, and the fix is measured on held-out data
(0.174 → 0.593 sealed macro-F1, 0% zero-feature) and on the real inputs that
exposed it. Preview and Confirm cannot disagree because they are one code
path. Correction memory works on real bank text for the first time.
Forecasting is gated at its true mathematical minimum, with the reason stated
and the stronger claim explicitly refused. The analytics answer real
questions from real stored data and refuse to draw a chart they cannot
support. Every claim on the How It Works page traces to a committed
evaluation record, and the page states as prominently what the product cannot
do as what it can.

The remaining weakness — unseen, non-descriptive brand names — is inherent to
classical text classification on a privacy-safe corpus, is disclosed in the
product rather than hidden, and is precisely what the human-in-the-loop layer
exists to absorb.
