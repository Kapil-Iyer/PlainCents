/**
 * Transaction categorization evidence, transcribed from committed ML
 * reports only. Every number below has a source comment naming the exact
 * file it came from — do not hand-edit a value without updating the cited
 * source file first.
 *
 * Source of truth (read-only references, never re-derived from memory):
 *   - reports/ml/ML_C_SELECTION_RECORD.json
 *   - reports/ml/ML_C_EXPERIMENT_REPORT.md
 *   - reports/ml/ML_E_CLAIM_MATRIX.json
 *   - reports/ml/results/final_categorization.json
 */

export interface CategorizationCandidate {
  id: string;
  label: string;
  /** VALIDATION accuracy, as a 0-100 percentage, rounded to 1 decimal —
   * matches the rounding ML_E_CLAIM_MATRIX.json's own claims use. */
  validationAccuracyPct: number;
  /** VALIDATION macro F1, rounded to 4 decimals — matches the precision
   * ML_C_SELECTION_RECORD.json's close_call_analysis_logreg_vs_svm and
   * ML_E_CLAIM_MATRIX.json use for this exact figure. */
  validationMacroF1: number;
  selected: boolean;
  rejectionReason?: string;
}

/**
 * VALIDATION bake-off — all three benchmarked candidates.
 * Source: ML_C_SELECTION_RECORD.json > categorization_selection >
 * primary_validation_evidence.values / secondary_validation_evidence.accuracy.
 */
export const CATEGORIZATION_CANDIDATES: CategorizationCandidate[] = [
  {
    id: "kmeans",
    label: "K-Means",
    validationAccuracyPct: 12.0,
    validationMacroF1: 0.0566,
    selected: false,
    rejectionReason:
      "Near-chance VALIDATION performance (~12.5% random-chance floor for 8 categories). A ~30-point TRAIN(42.1%)/VALIDATION(12.0%) accuracy gap is a textbook generalization-failure signature.",
  },
  {
    id: "tfidf_linear_svm",
    label: "TF-IDF + Linear SVM",
    validationAccuracyPct: 26.0,
    validationMacroF1: 0.2405,
    selected: false,
    rejectionReason:
      "Beaten on both primary (macro F1) and secondary (accuracy) VALIDATION metrics by Logistic Regression, with no offsetting complexity/runtime/maintainability advantage.",
  },
  {
    id: "tfidf_logreg",
    label: "TF-IDF + Logistic Regression",
    validationAccuracyPct: 32.0,
    validationMacroF1: 0.2552,
    selected: true,
  },
];

/**
 * Selection rationale — condensed from ML_C_SELECTION_RECORD.json >
 * categorization_selection.close_call_analysis_logreg_vs_svm /
 * alternatives_rejected / trd_compatibility.
 */
export const CATEGORIZATION_SELECTION_RATIONALE = [
  "Highest VALIDATION macro F1 (primary metric per the ML Spec) and highest accuracy (secondary metric) of the three candidates benchmarked.",
  "K-Means's near-chance VALIDATION accuracy was traced, via the mandatory error analysis, to a structural cause: 0 of 50 VALIDATION merchant strings share even one TRAIN top-50 vocabulary token — not an isolated failure.",
  "Against Linear SVM, the margin (macro F1 0.2552 vs 0.2405) was treated as a close call, not decisive on its own given the small sample — but Logistic Regression also produces calibrated class probabilities natively, while Linear SVM would need an extra calibration step to do the same. No factor favored SVM enough to override the primary-metric lead.",
  "Fits CategorizationService's predict(transaction) -> {predicted_category} contract as a drop-in replacement for the previous K-Means implementation — no schema or API change.",
];

/**
 * FINAL_TEST — held-out result for the selected candidate only.
 * Source: reports/ml/results/final_categorization.json,
 * cross-checked against ML_E_CLAIM_MATRIX.json's "42.2% accuracy" /
 * "0.4405 macro F1" claims (SUPPORTED_WITH_QUALIFICATION).
 */
export const CATEGORIZATION_FINAL_RESULT = {
  resultLabel: "Tier B curated benchmark — held-out FINAL_TEST",
  accuracyPct: 42.2,
  macroF1: 0.4405,
  nRows: 45,
  nMerchantGroups: 17,
  evidenceTier: "Tier B" as const,
  notToBeDescribedAs: ["real-world performance", "Tier A performance", "temporal validation performance"],
  limitation:
    "n=45 rows / 17 merchant groups, an independently curated (not real bank export) benchmark. A single held-out measurement, not a distribution — per-category F1 ranges from 0.22 to 1.0 across only 4–9 support rows per category.",
};

export const CATEGORIZATION_PIPELINE_EXPLANATION =
  "merchant description → TF-IDF → classifier → predicted category";
