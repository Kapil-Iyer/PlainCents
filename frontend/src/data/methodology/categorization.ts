/**
 * Transaction categorization evidence, transcribed from committed ML
 * reports only. Every number below has a source comment naming the exact
 * file it came from — do not hand-edit a value without updating the cited
 * source file first.
 *
 * ML-F AMENDMENT: the ML-C-era production recipe (word TF-IDF,
 * max_features=50, fit on the Tier B benchmark's TRAIN partition) has been
 * superseded. reports/ml/ML_F_SELECTION_RECORD.json is now the primary
 * source of truth for the production categorization recipe; the ML-C
 * sources below remain valid historical evidence for the DECISION they
 * document (retained, not deleted), but no longer describe what ships.
 *
 * Source of truth (read-only references, never re-derived from memory):
 *   - reports/ml/ML_F_SELECTION_RECORD.json (current production recipe)
 *   - reports/ml/results/deployment_categorization_results.json (full bake-off)
 *   - reports/ml/results/deployment_winner_tier_b_continuity.json (continuity)
 *   - reports/ml/ML_C_SELECTION_RECORD.json (historical — superseded ML-C decision)
 *   - reports/ml/ML_C_EXPERIMENT_REPORT.md (historical)
 *   - reports/ml/ML_E_CLAIM_MATRIX.json (historical)
 */

export interface CategorizationCandidate {
  id: string;
  label: string;
  /** VALIDATION accuracy, as a 0-100 percentage, rounded to 1 decimal. */
  validationAccuracyPct: number;
  /** VALIDATION macro F1, rounded to 4 decimals. */
  validationMacroF1: number;
  selected: boolean;
  rejectionReason?: string;
}

/**
 * VALIDATION bake-off on the new sanitized deployment-oriented benchmark
 * (ml/data/build_deployment_benchmark.py; 190 rows / 73 merchant groups,
 * hand-curated/fabricated merchants modeled on real-export structure — NOT
 * real-world transactions). TRAIN=96 rows/41 groups, VALIDATION=41 rows/16
 * groups, FINAL_TEST=39 rows/16 groups (categorical rows only — 14 rows
 * across all partitions are separately flagged is_ambiguous and excluded
 * from these numbers; see CATEGORIZATION_AMBIGUOUS_ROUTING below).
 * Source: reports/ml/results/deployment_categorization_results.json > candidates.
 */
export const CATEGORIZATION_CANDIDATES: CategorizationCandidate[] = [
  {
    id: "a_baseline",
    label: "A — Baseline (word TF-IDF, 50 features)",
    validationAccuracyPct: 26.8,
    validationMacroF1: 0.2826,
    selected: false,
    rejectionReason:
      "The exact ML-C production structure, refit on the new deployment TRAIN data instead of Tier B. Beaten by simply raising the vocabulary cap — the smallest, most defensible next step, tried first.",
  },
  {
    id: "b_100",
    label: "B — Word TF-IDF, 100 features",
    validationAccuracyPct: 31.7,
    validationMacroF1: 0.312,
    selected: false,
    rejectionReason: "Improves on the baseline but not as much as a further-enlarged vocabulary.",
  },
  {
    id: "b_200",
    label: "B — Word TF-IDF, 200 features",
    validationAccuracyPct: 41.5,
    validationMacroF1: 0.3854,
    selected: true,
  },
  {
    id: "b_400_unbounded",
    label: "B — Word TF-IDF, 400 / unbounded features",
    validationAccuracyPct: 41.5,
    validationMacroF1: 0.3854,
    selected: false,
    rejectionReason:
      "Numerically tied with 200 features (this small corpus's vocabulary is already exhausted by 200) — 200 is kept as the smaller, equally-performing choice, per the pre-registered simplicity tie-break.",
  },
  {
    id: "c_normalized",
    label: "C — Boilerplate-normalized text + word TF-IDF",
    validationAccuracyPct: 24.4,
    validationMacroF1: 0.294,
    selected: false,
    rejectionReason:
      "Stripping bank transaction-method boilerplate (“VISA DEBIT PURCHASE”, “POS PURCHASE”, card-suffix digits) underperformed simply enlarging the untouched vocabulary at this corpus size — those tokens turned out to carry some category signal in this dataset that outweighed the noise they also added.",
  },
  {
    id: "d_char_ngram",
    label: "D — Character n-grams (3-5), TF-IDF",
    validationAccuracyPct: 34.1,
    validationMacroF1: 0.3385,
    selected: false,
    rejectionReason:
      "Robust to truncation/suffix noise as intended, and beat the plain baseline — but still behind simply enlarging the word vocabulary on this corpus.",
  },
  {
    id: "e_word_char",
    label: "E — Word + character TF-IDF (combined)",
    validationAccuracyPct: 19.5,
    validationMacroF1: 0.2798,
    selected: false,
    rejectionReason:
      "Run because Candidate D showed a meaningful gain over the baseline (pre-registered trigger) — but combining representations diluted rather than compounded the signal on this small corpus.",
  },
  {
    id: "f_svm",
    label: "F — Linear SVM (winning representation)",
    validationAccuracyPct: 34.1,
    validationMacroF1: 0.3445,
    selected: false,
    rejectionReason:
      "Confirmation pass on Candidate B's representation — did not beat Logistic Regression by a meaningful margin, so LogReg is kept per the pre-registered tie-break (native class probabilities, already the shipped architecture).",
  },
];

/**
 * Selection rationale — condensed from reports/ml/ML_F_SELECTION_RECORD.json
 * > winner_selection_reasoning / what_improved / what_still_cannot_be_inferred.
 */
export const CATEGORIZATION_SELECTION_RATIONALE = [
  "Highest VALIDATION macro F1 among all 8 candidates benchmarked (A-F, with B run at 4 vocabulary sizes), selected BEFORE the sealed FINAL_TEST was touched — same pre-registered-rule discipline as ML-C.",
  "Structurally the simplest change from the previous production recipe: the same word TF-IDF + Logistic Regression architecture, just a larger vocabulary (200 vs. 50 features) fit on a new TRAIN partition that includes deployment-shaped bank boilerplate instead of only Tier B's cleaner synthetic text.",
  "Character n-grams (D) and deterministic boilerplate normalization (C) — both motivated by the real-export audit's findings — did not beat a larger plain word vocabulary at this corpus's size; that is reported as a size-limited finding, not a rejection of either technique.",
  "A deterministic, non-ML rule now catches generic e-transfer/ATM/ABM rows (no spending-purpose signal by construction) upstream of the classifier — see the ambiguous-row routing note below.",
];

/**
 * FINAL_TEST — held-out result for the selected candidate only, on the new
 * deployment-oriented benchmark. Source:
 * reports/ml/results/deployment_categorization_results.json > final_test.
 */
export const CATEGORIZATION_FINAL_RESULT = {
  resultLabel: "Sanitized deployment-oriented benchmark — held-out FINAL_TEST",
  accuracyPct: 30.8,
  macroF1: 0.1742,
  nRows: 39,
  nMerchantGroups: 15,
  evidenceTier: "Sanitized deployment-oriented" as const,
  notToBeDescribedAs: ["real-world performance", "Tier A performance", "temporal validation performance"],
  limitation:
    "n=39 rows / 15 merchant groups — a small, hand-curated corpus, not real bank data. Held-out performance is modest and reported as-is, not inflated: this dataset's own small size is itself the dominant limitation the ML-F-A audit identified (training-data coverage), and a bigger corpus is the most direct next lever, not attempted in this phase per its explicit scope cap.",
};

/**
 * Continuity check: the SAME winning recipe (never re-selected using this
 * benchmark) evaluated on the original Tier B benchmark.
 * Source: reports/ml/results/deployment_winner_tier_b_continuity.json.
 */
export const CATEGORIZATION_TIER_B_CONTINUITY = {
  validationMacroF1: 0.3385,
  finalTestMacroF1: 0.5067,
  note: "Tier B was NOT used to select this recipe — evaluated only for continuity after the winner was frozen on the deployment benchmark. Performance here is comparable to (slightly above) the original ML-C Tier B FINAL_TEST result (0.4405), i.e. no regression from the vocabulary/training-data change.",
};

/**
 * Deterministic ambiguous-row routing (backend/services/ambiguity.py):
 * generic Interac e-transfer sends and ABM/ATM withdrawals carry no
 * spending-purpose signal by construction and are routed to "Other" by a
 * fixed rule, independent of the classifier — never scored into the
 * macro-F1 numbers above. Source:
 * reports/ml/results/deployment_categorization_results.json > final_test_ambiguous_routing.
 */
export const CATEGORIZATION_AMBIGUOUS_ROUTING = {
  nAmbiguousFinalTest: 2,
  coverage: 1.0,
  note: "100% of the benchmark's structurally-ambiguous rows (generic e-transfer / ABM / ATM) were correctly routed to “Other” by the deterministic rule, independent of and prior to the ML classifier.",
};

export const CATEGORIZATION_PIPELINE_EXPLANATION =
  "merchant description → TF-IDF → classifier → predicted category (+ personalized correction memory, + deterministic ambiguous-row routing)";
