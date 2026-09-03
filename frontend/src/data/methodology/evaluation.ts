/**
 * Evaluation methodology facts. Sourced from:
 *   - docs/V2_ML_SPEC.md (§6: split protocol; "merchant-grouped,
 *     category-stratified split" is the spec's own chosen-protocol wording —
 *     used verbatim below, not invented).
 *   - reports/ml/ML_C_SELECTION_RECORD.json (declaration / TRAIN+VALIDATION
 *     only until FINAL was opened; ML-C Part B fold-stability).
 *   - reports/ml/ML_E_CLAIM_MATRIX.json ("Merchant groups do not overlap
 *     TRAIN/VALIDATION/FINAL", SUPPORTED).
 */
export const SPLIT_ROLES = [
  {
    id: "train",
    label: "TRAIN",
    description: "Used to fit the vectorizer/model. Never consulted for comparison or selection.",
  },
  {
    id: "validation",
    label: "VALIDATION",
    description: "Used to compare candidates and pick a winner. This is the only evidence selection decisions were based on.",
  },
  {
    id: "final_test",
    label: "FINAL_TEST",
    description: "Sealed until after the selection was finalized. Evaluated exactly once, for the selected candidate only — never used to pick between candidates.",
  },
];

export const MERCHANT_ISOLATION_EXPLANATION =
  "Chosen protocol: merchant-grouped, category-stratified split. All transactions sharing the same normalized merchant identity are assigned to exactly one of TRAIN / VALIDATION / FINAL_TEST — never split across them — so a model cannot succeed merely by recognizing a merchant string it already saw during training. Within that grouping constraint, category balance is preserved as closely as feasible across the three partitions. Verified structurally (empty pairwise intersections) both in the evaluation runner and as a defense-in-depth assertion in the production model build script.";

export const SEALED_FINAL_TEST_DISCIPLINE =
  "Selection (which categorization model, which forecaster, whether a per-horizon strategy was needed) was finalized using TRAIN + VALIDATION evidence only. Neither component's FINAL data had been evaluated at the time the selection record was written. FINAL_TEST was opened exactly once, afterward, for the already-selected candidate — never used to choose between candidates, and never re-run after seeing its result.";

export const TEMPORAL_VALIDATION_EXPLANATION =
  "Forecasting validation uses an expanding-window protocol: 14 chronological origins, each predicting +1/+2/+3 months forward from a training window that only ever grows forward in time — never a random shuffle of months. Per-origin, per-horizon win rates against the Naive baseline were computed directly from these 14 origins to confirm the pooled-aggregate result wasn't an artifact of a few lucky/unlucky folds.";
