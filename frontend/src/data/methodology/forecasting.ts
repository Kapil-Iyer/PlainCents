/**
 * Spending forecasting evidence, transcribed from committed ML reports
 * only. Every number below has a source comment naming the exact file it
 * came from.
 *
 * Source of truth (read-only references):
 *   - reports/ml/ML_C_SELECTION_RECORD.json
 *   - reports/ml/ML_C_EXPERIMENT_REPORT.md
 *   - reports/ml/ML_E_CLAIM_MATRIX.json
 *   - reports/ml/results/final_forecasting.json
 */

export interface ForecastingCandidate {
  id: string;
  label: string;
  /** Multi-step strategy, where the evidence distinguishes one. Naive has
   * no meaningful strategy axis (ML_C_SELECTION_RECORD.json >
   * multi_step_strategy_selection: "selected_strategy": "N/A") — represented
   * as null here, rendered as "N/A" in the UI, never collapsed into the
   * other variants' rows. */
  strategy: "last-known-history" | "recursive" | null;
  /** Pooled VALIDATION WAPE by horizon — exact 4-decimal values as given in
   * ML_C_SELECTION_RECORD.json > forecasting_selection >
   * section_14_eligibility_filter_applied.pooled_validation_wape_by_horizon. */
  validationWape: { h1: number; h2: number; h3: number };
  selected: boolean;
  rejectionReason?: string;
}

export const FORECASTING_CANDIDATES: ForecastingCandidate[] = [
  {
    id: "naive",
    label: "Naive (lag-1)",
    strategy: null,
    validationWape: { h1: 0.202, h2: 0.1904, h3: 0.1765 },
    selected: true,
  },
  {
    id: "seasonal_naive",
    label: "Seasonal Naive",
    strategy: null,
    validationWape: { h1: 0.2631, h2: 0.2631, h3: 0.2631 },
    selected: false,
    rejectionReason:
      "Loses to Naive at every horizon, and in 11/11 origins where eligible (100%). Its own >=13-month-history eligibility floor also limits its evaluated sample to 216/312 rows.",
  },
  {
    id: "ridge_last_known_history",
    label: "Ridge",
    strategy: "last-known-history",
    validationWape: { h1: 0.202, h2: 0.2235, h3: 0.2493 },
    selected: false,
    rejectionReason:
      "Ties Naive at +1 (numerically identical, a documented coincidence), then loses at +2 and +3 — the 'good +1 masking poor +2/+3' failure pattern the ML Spec's Section 14 explicitly requires checking for.",
  },
  {
    id: "ridge_recursive",
    label: "Ridge",
    strategy: "recursive",
    validationWape: { h1: 0.202, h2: 0.2347, h3: 0.2887 },
    selected: false,
    rejectionReason:
      "Fails more severely than Ridge (last-known-history) — recursive error compounding makes +2/+3 worse still.",
  },
  {
    id: "random_forest_last_known_history",
    label: "Random Forest",
    strategy: "last-known-history",
    validationWape: { h1: 0.2426, h2: 0.2354, h3: 0.2495 },
    selected: false,
    rejectionReason:
      "Loses to Naive at every single horizon on pooled WAPE, and wins fewer than half of individual VALIDATION origins at every horizon (best case 46% at +2). Was the production forecaster prior to this evaluation.",
  },
  {
    id: "random_forest_recursive",
    label: "Random Forest",
    strategy: "recursive",
    validationWape: { h1: 0.2426, h2: 0.2524, h3: 0.2771 },
    selected: false,
    rejectionReason:
      "Fails more severely than Random Forest (last-known-history); recursive error compounding is measured and monotonic (+1 -> +2 -> +3).",
  },
];

/**
 * Selection rationale — condensed from ML_C_SELECTION_RECORD.json >
 * forecasting_selection.section_14_eligibility_filter_applied /
 * fold_level_stability_evidence / per_horizon_selection_decision.
 */
export const FORECASTING_SELECTION_RATIONALE = [
  "ML Spec Section 14(a): a candidate only becomes eligible to replace Naive if it beats Naive on WAPE separately at +1, +2, AND +3 — not just in combination. No candidate in the evaluated set cleared this bar at all three horizons.",
  "Fold-stability check (14 expanding-window VALIDATION origins): every RF/Ridge variant loses to Naive in the majority of individual origins at every horizon — the losses are not a few unlucky folds, they're the norm, and they worsen at +3 for every candidate (recursive strategies collapse hardest, down to an 8% win rate).",
  "Naive is O(1) — a single lookup, no fitting — strictly simpler and faster than the previously-shipped Random Forest (refit per call). Per the ML Spec's explicit anticipated outcome, selecting the simpler baseline when nothing reliably beats it is the scientifically correct decision, not a fallback.",
  "A single strategy (Naive) applies uniformly to all three horizons — no per-horizon complexity was introduced, since Naive is at least as good as every other candidate at every individual horizon.",
];

/**
 * Reserved-period FINAL result for the selected candidate only.
 * Source: reports/ml/results/final_forecasting.json.
 * by_horizon percentages are that file's fractional WAPE values (h1
 * 0.1938370565988991, h2 0.14686835200085907, h3 0.22166005251357798)
 * multiplied by 100 and rounded to 2 decimals — a direct, non-cherry-picked
 * derivation, not an independently sourced number.
 */
export const FORECASTING_FINAL_RESULT = {
  resultLabel: "Untouched temporal-test performance on reserved synthetic months",
  combinedWapePct: 18.87,
  byHorizonWapePct: { h1: 19.38, h2: 14.69, h3: 22.17 },
  reservedMonths: ["2024-10", "2024-11", "2024-12"],
  nPredictions: 24,
  evidenceTier: "Synthetic" as const,
  notToBeDescribedAs: ["Tier B", "real-world", "temporal validation"],
  limitation:
    "Fully synthetic dataset (data/raw/synthetic_24mo.csv, 779 synthetic transactions, 24 months), categorized read-only by the production K-Means artifact. Never real spending data — this is a mechanism/behavior check, not a real-world accuracy figure.",
};

export const FORECASTING_STRATEGY_NA_NOTE =
  "The selected forecaster (Naive) has no meaningful multi-step strategy distinction: it always predicts the single most recent observed value, so a 'recursive' re-application produces the identical value at every horizon as the non-recursive version. Strategy A (last-known-history) vs. B (recursive) was only a meaningful, evaluated axis for Random Forest and Ridge — both rejected above. Recorded as N/A for Naive rather than forcing an inapplicable choice.";

export const FORECASTING_PIPELINE_EXPLANATION =
  "monthly category history → temporal evaluation/model → +1/+2/+3 forecasts";

/**
 * FINAL-period training window immediately preceding the 3 reserved months.
 * Added in Phase 11C-B to support the temporal-validation timeline visual
 * (frontend previously had no structured export for the training window —
 * only the already-frozen `reservedMonths` above).
 * Source: reports/ml/ML_E_FINAL_ML_REPORT.md §24 ("Reserved period
 * 2024-10/11/12, trained on 2023-01 through 2024-09 (21 months)"). No new
 * claim — a direct restatement of the same sentence already backing
 * FORECASTING_FINAL_RESULT above.
 */
export const FORECASTING_FINAL_TRAIN_WINDOW = {
  start: "2023-01",
  end: "2024-09",
  months: 21,
} as const;
