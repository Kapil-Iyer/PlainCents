/**
 * Spending forecasting evidence, transcribed from committed ML reports
 * only. Every number below has a source comment naming the exact file it
 * came from.
 *
 * ML-F AMENDMENT: Naive lag-1 (ML-C's selection) has been superseded by a
 * 3-month rolling mean, found to beat Naive by a meaningful margin in a
 * small, pre-registered re-evaluation that added rolling-mean/EWMA
 * candidates to the same walk-forward harness ML-C used. The ML-C sources
 * below remain valid historical evidence for the decision they document
 * (retained, not deleted); reports/ml/ML_F_SELECTION_RECORD.json is now the
 * primary source of truth for the production recipe.
 *
 * Source of truth (read-only references):
 *   - reports/ml/ML_F_SELECTION_RECORD.json (current production recipe)
 *   - reports/ml/results/ml_f_forecasting_metrics.json (full bake-off)
 *   - reports/ml/results/ml_f_final_forecasting.json (sealed FINAL_TEST)
 *   - reports/ml/results/ml_f_history_length_sensitivity.json
 *   - reports/ml/ML_C_SELECTION_RECORD.json (historical — superseded ML-C decision)
 *   - reports/ml/ML_C_EXPERIMENT_REPORT.md (historical)
 */

export interface ForecastingCandidate {
  id: string;
  label: string;
  /** Multi-step strategy, where the evidence distinguishes one. Naive and
   * the ML-F-added rolling-mean/EWMA baselines have no meaningful strategy
   * axis (same value reused at every horizon) — represented as null here,
   * rendered as "N/A" in the UI, never collapsed into the other variants'
   * rows. */
  strategy: "last-known-history" | "recursive" | null;
  /** Pooled VALIDATION WAPE by horizon — exact 4-decimal values as given in
   * reports/ml/results/ml_f_forecasting_metrics.json > by_candidate_strategy
   * > <key> > by_horizon. */
  validationWape: { h1: number; h2: number; h3: number };
  selected: boolean;
  rejectionReason?: string;
}

export const FORECASTING_CANDIDATES: ForecastingCandidate[] = [
  {
    id: "rolling_mean_3",
    label: "Rolling mean (3 months)",
    strategy: null,
    validationWape: { h1: 0.1575, h2: 0.171, h3: 0.1745 },
    selected: true,
  },
  {
    id: "ewma_0.5",
    label: "EWMA (α=0.5)",
    strategy: null,
    validationWape: { h1: 0.163, h2: 0.166, h3: 0.1663 },
    selected: false,
    rejectionReason:
      "Marginally lower pooled WAPE than the 3-month rolling mean (0.1650 vs 0.1672) — within the pre-registered tie-break margin. Rolling mean is kept as the simpler choice: a plain average needs no smoothing-factor decision the way EWMA's alpha does.",
  },
  {
    id: "ewma_0.3",
    label: "EWMA (α=0.3)",
    strategy: null,
    validationWape: { h1: 0.1613, h2: 0.17, h3: 0.1722 },
    selected: false,
    rejectionReason: "Within the tied cluster of rolling-mean/EWMA candidates; rolling mean is simpler.",
  },
  {
    id: "ewma_0.7",
    label: "EWMA (α=0.7)",
    strategy: null,
    validationWape: { h1: 0.1741, h2: 0.1691, h3: 0.1664 },
    selected: false,
    rejectionReason: "Still beats Naive, but the weakest of the three tested alpha values.",
  },
  {
    id: "rolling_mean_6",
    label: "Rolling mean (6 months)",
    strategy: null,
    validationWape: { h1: 0.1638, h2: 0.1737, h3: 0.1769 },
    selected: false,
    rejectionReason: "A shorter (3-month) window did slightly better — more responsive to recent spend.",
  },
  {
    id: "naive",
    label: "Naive (lag-1)",
    strategy: null,
    validationWape: { h1: 0.202, h2: 0.1904, h3: 0.1765 },
    selected: false,
    rejectionReason:
      "Previously selected (ML-C). Beaten by a meaningful, non-noise margin on pooled VALIDATION WAPE by every rolling-mean/EWMA candidate re-evaluated in ML-F — the current production forecaster since ML-D, now superseded.",
  },
  {
    id: "ridge_last_known_history",
    label: "Ridge",
    strategy: "last-known-history",
    validationWape: { h1: 0.202, h2: 0.2235, h3: 0.2493 },
    selected: false,
    rejectionReason: "Loses to every simple baseline at +2/+3 — unchanged finding from ML-C.",
  },
  {
    id: "ridge_recursive",
    label: "Ridge",
    strategy: "recursive",
    validationWape: { h1: 0.202, h2: 0.2347, h3: 0.2887 },
    selected: false,
    rejectionReason: "Fails more severely than Ridge (last-known-history) — recursive error compounding.",
  },
  {
    id: "random_forest_last_known_history",
    label: "Random Forest",
    strategy: "last-known-history",
    validationWape: { h1: 0.2426, h2: 0.2354, h3: 0.2495 },
    selected: false,
    rejectionReason: "Loses to every simple baseline at every horizon — unchanged finding from ML-C, kept for continuity.",
  },
  {
    id: "random_forest_recursive",
    label: "Random Forest",
    strategy: "recursive",
    validationWape: { h1: 0.2426, h2: 0.2524, h3: 0.2771 },
    selected: false,
    rejectionReason: "Fails more severely than Random Forest (last-known-history); recursive error compounding.",
  },
  {
    id: "seasonal_naive",
    label: "Seasonal Naive",
    strategy: null,
    validationWape: { h1: 0.2631, h2: 0.2631, h3: 0.2631 },
    selected: false,
    rejectionReason:
      "Loses to every other candidate at every horizon, and its own >=13-month-history eligibility floor limits its evaluated sample to 216/312 rows.",
  },
];

/**
 * Selection rationale — condensed from
 * reports/ml/ML_F_SELECTION_RECORD.json > forecasting_selection.
 */
export const FORECASTING_SELECTION_RATIONALE = [
  "A small, pre-registered bake-off added 3-month/6-month rolling-mean and EWMA (α ∈ {0.3, 0.5, 0.7}) baselines to the exact same walk-forward harness ML-C used (same 24-month synthetic grid, same 14 expanding-window origins, same 3-month reserved FINAL period) — Ridge/Random Forest/Seasonal Naive were re-run unchanged, for continuity, not re-tuned.",
  "3-month rolling mean beat Naive by a meaningful margin on pooled VALIDATION WAPE (0.1672 vs 0.1903 — about 12% relative), clearing the pre-registered “meaningful, not noise” bar.",
  "Stable across truncated history lengths exactly like Naive was: pooled WAPE is identical at 6, 9, 12, and 18 months of truncated TRAIN history, because a 3-month rolling average, like a lag-1 value, never depends on how much OLDER history exists once enough recent months are present.",
  "Tie-broken toward simplicity: an EWMA variant scored marginally lower WAPE but within the pre-registered tie-break margin, and a plain average has one fewer hyperparameter to justify (no smoothing-factor choice) — the smallest defensible change from Naive that still captures a real improvement.",
];

/**
 * Reserved-period FINAL result for the selected candidate only.
 * Source: reports/ml/results/ml_f_final_forecasting.json.
 */
export const FORECASTING_FINAL_RESULT = {
  resultLabel: "Untouched temporal-test performance on reserved synthetic months",
  combinedWapePct: 17.84,
  byHorizonWapePct: { h1: 26.97, h2: 7.32, h3: 19.6 },
  reservedMonths: ["2024-10", "2024-11", "2024-12"],
  nPredictions: 24,
  evidenceTier: "Synthetic" as const,
  notToBeDescribedAs: ["Tier B", "real-world", "temporal validation"],
  limitation:
    "Fully synthetic dataset (data/raw/synthetic_24mo.csv, 779 synthetic transactions, 24 months), categorized read-only by the production K-Means artifact. Never real spending data — this is a mechanism/behavior check, not a real-world accuracy figure. A modest improvement over the prior Naive FINAL_TEST result (18.87% -> 17.84% combined WAPE), consistent with (though smaller than) the VALIDATION-stage gain.",
};

export const FORECASTING_STRATEGY_NA_NOTE =
  "The selected forecaster (3-month rolling mean) has no meaningful multi-step strategy distinction, for the same reason Naive didn't: the predicted value never depends on a prior *prediction*, so a 'recursive' re-application produces the identical value at every horizon as the non-recursive version. Strategy A (last-known-history) vs. B (recursive) was only a meaningful, evaluated axis for Random Forest and Ridge — both rejected above. Recorded as N/A for the simple baselines rather than forcing an inapplicable choice.";

export const FORECASTING_PIPELINE_EXPLANATION =
  "monthly category history → temporal evaluation/model → +1/+2/+3 forecasts";

/**
 * ML-F history-length sensitivity: the selected recipe's VALIDATION WAPE is
 * unchanged whether TRAIN history is truncated to 6, 9, 12, or 18 months —
 * the evidence the 6-month product eligibility threshold rests on (an
 * availability/UX floor, not an accuracy claim).
 * Source: reports/ml/results/ml_f_history_length_sensitivity.json.
 */
export const FORECASTING_HISTORY_LENGTH_SENSITIVITY = {
  truncationLengthsMonths: [6, 9, 12, 18],
  rollingMean3Wape: [0.1484, 0.1484, 0.1484, 0.1484],
  naiveWape: [0.2054, 0.2054, 0.2054, 0.2054],
  note: "Identical WAPE at every truncation length for both the winner and Naive — a 3-month rolling average, like a lag-1 value, is invariant to how much history exists beyond what it actually uses.",
};

export const FORECASTING_ELIGIBILITY = {
  monthsRequired: 6,
  previousMonthsRequired: 12,
  mathematicalMinimum: 3,
  note: "6 months is a product-history usability floor, not an accuracy claim — the selected 3-month rolling mean mathematically needs only 3 months per category and performs identically at 6/9/12/18 months of history. Lowered from 12 (PRD §21 / TRD §12.5 amendment).",
};

/**
 * FINAL-period training window immediately preceding the 3 reserved months.
 * Unchanged by ML-F — same reserved period, same training window, only the
 * model producing the prediction changed.
 * Source: reports/ml/ML_E_FINAL_ML_REPORT.md §24 ("Reserved period
 * 2024-10/11/12, trained on 2023-01 through 2024-09 (21 months)").
 */
export const FORECASTING_FINAL_TRAIN_WINDOW = {
  start: "2023-01",
  end: "2024-09",
  months: 21,
} as const;
