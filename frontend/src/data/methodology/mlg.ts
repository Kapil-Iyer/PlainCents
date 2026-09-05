/**
 * ML-G categorization evidence, transcribed from committed reports only.
 *
 * SOURCE OF TRUTH (never re-derive a number from memory; if one of these
 * files changes, update this file from it):
 *   - reports/ml/ML_G_SELECTION_RECORD.json          (the frozen selection)
 *   - reports/ml/results/mlg_categorization_results.json (the full bake-off)
 *
 * The ML-C / ML-F data files in this directory are retained as historical
 * evidence of the decisions they document. They no longer describe what
 * ships.
 */

export interface MlgCandidate {
  id: string;
  label: string;
  hypothesis: string;
  validationMacroF1: number;
  validationAccuracyPct: number;
  /** Share of held-out rows the vectorizer produced NO features for. */
  zeroFeatureRatePct: number;
  selected: boolean;
  outcome: string;
}

/**
 * Every classical-ML configuration tried in this phase, ranked by VALIDATION
 * macro-F1. Source: ML_G_SELECTION_RECORD.json > candidates_evaluated,
 * ranked_by_validation_macro_f1, hypotheses.
 */
export const MLG_CANDIDATES: MlgCandidate[] = [
  {
    id: "G6",
    label: "Word + character TF-IDF union (char 2–6) → Logistic Regression",
    hypothesis:
      "Word features carry merchant identity and category head nouns; character features survive mid-word truncation. A union of both should beat either alone, and a wider character range should catch shorter stems like PHARM or DENT.",
    validationMacroF1: 0.6659,
    validationAccuracyPct: 67.2,
    zeroFeatureRatePct: 0,
    selected: true,
    outcome: "Selected — highest validation macro-F1, and no held-out row was left without features.",
  },
  {
    id: "G14",
    label: "Same union, class_weight='balanced'",
    hypothesis:
      "Balancing class weights should lift recall on the smaller categories, which is what macro-F1 rewards.",
    validationMacroF1: 0.6636,
    validationAccuracyPct: 67.2,
    zeroFeatureRatePct: 0,
    selected: false,
    outcome: "Statistically indistinguishable from the winner; the simpler unweighted model was kept.",
  },
  {
    id: "G13",
    label: "Same union, C = 4",
    hypothesis:
      "With high-dimensional sparse text and few rows per class, the default C = 1 may be over-regularized.",
    validationMacroF1: 0.6527,
    validationAccuracyPct: 66.2,
    zeroFeatureRatePct: 0,
    selected: false,
    outcome: "Slightly worse. The default regularization was not the bottleneck.",
  },
  {
    id: "G5",
    label: "Word + character TF-IDF union (char 3–5)",
    hypothesis: "The core union hypothesis, at the narrower character n-gram range.",
    validationMacroF1: 0.6461,
    validationAccuracyPct: 64.6,
    zeroFeatureRatePct: 0,
    selected: false,
    outcome: "Confirmed the union works; the wider character range was measurably better.",
  },
  {
    id: "G10",
    label: "Complement Naive Bayes on the union",
    hypothesis:
      "ComplementNB is designed for imbalanced text classification and costs almost nothing to fit.",
    validationMacroF1: 0.6404,
    validationAccuracyPct: 63.1,
    zeroFeatureRatePct: 0,
    selected: false,
    outcome: "Competitive but behind, and it offers no calibrated margin for the abstention rule.",
  },
  {
    id: "G3",
    label: "Word TF-IDF only, unbounded vocabulary, normalized",
    hypothesis:
      "Stripping bank boilerplate should free vocabulary capacity for merchant-identity terms.",
    validationMacroF1: 0.6315,
    validationAccuracyPct: 60.5,
    zeroFeatureRatePct: 40.5,
    selected: false,
    outcome:
      "Rejected on a number accuracy alone would have hidden: 40.5% of held-out rows produced NO features at all. Every one of those gets whatever class the model's intercept favours — the exact failure this phase set out to fix.",
  },
  {
    id: "G11",
    label: "Multinomial Naive Bayes on the union",
    hypothesis:
      "Control for ComplementNB — confirms any gain comes from the complement formulation, not from NB generally.",
    validationMacroF1: 0.6077,
    validationAccuracyPct: 60.0,
    zeroFeatureRatePct: 0,
    selected: false,
    outcome: "Behind ComplementNB, as expected on imbalanced classes.",
  },
  {
    id: "G9",
    label: "Linear SVM on the union",
    hypothesis: "LinearSVC is the classical strong baseline for sparse text.",
    validationMacroF1: 0.5891,
    validationAccuracyPct: 57.4,
    zeroFeatureRatePct: 0,
    selected: false,
    outcome:
      "Behind logistic regression here, and it produces no probabilities — the abstention rule would have had to work off uncalibrated distances.",
  },
  {
    id: "G4",
    label: "Character TF-IDF only",
    hypothesis:
      "Character n-grams alone survive truncated and run-together descriptions where whole word tokens do not exist.",
    validationMacroF1: 0.5524,
    validationAccuracyPct: 53.8,
    zeroFeatureRatePct: 0,
    selected: false,
    outcome: "Robust but blunt on its own — it misses the word-level head nouns that carry the category.",
  },
  {
    id: "G2",
    label: "Word TF-IDF only, unbounded vocabulary",
    hypothesis: "Perhaps the 200-term cap, not the word representation itself, was the problem.",
    validationMacroF1: 0.5054,
    validationAccuracyPct: 49.2,
    zeroFeatureRatePct: 2.6,
    selected: false,
    outcome: "A large gain over the old recipe, but well short of adding character features.",
  },
  {
    id: "G1",
    label: "The previous production recipe (word TF-IDF, 200 features)",
    hypothesis:
      "Control: refit the exact shipped recipe on the new corpus, to separate how much of the failure was the model from how much was the data.",
    validationMacroF1: 0.3608,
    validationAccuracyPct: 33.9,
    zeroFeatureRatePct: 5.6,
    selected: false,
    outcome:
      "The control that makes the rest interpretable: better data alone roughly doubled it, and the representation change roughly doubled it again.",
  },
];

/** Source: ML_G_SELECTION_RECORD.json > dataset. */
export const MLG_DATASET = {
  id: "deployment_benchmark_v2",
  description:
    "A sanitized, hand-authored corpus of Canadian-bank-style transaction descriptions. Every merchant name is fabricated for the file; the boilerplate shapes (card-purchase prefixes, mid-word truncation, pre-authorized-payment references) are modelled on real export structure, never copied from real transactions.",
  trainRows: 580,
  validationRows: 195,
  finalTestRows: 195,
  trainGroups: 119,
  validationGroups: 40,
  finalTestGroups: 40,
  previousRows: 190,
  previousGroups: 73,
} as const;

/** Source: ML_G_SELECTION_RECORD.json > sealed_final_test_*. */
export const MLG_FINAL_TEST = {
  macroF1ModelOnly: 0.5931,
  macroF1WithPolicy: 0.5762,
  accuracyWithPolicy: 0.5949,
  rows: 195,
  merchantGroups: 40,
  previousMacroF1: 0.1741,
  zeroFeatureRatePct: 0,
  perCategory: [
    { category: "Food & Dining", precision: 0.711, recall: 0.914, f1: 0.8, support: 35 },
    { category: "Subscriptions", precision: 1.0, recall: 0.65, f1: 0.788, support: 20 },
    { category: "Healthcare", precision: 0.727, recall: 0.64, f1: 0.681, support: 25 },
    { category: "Transport", precision: 0.531, recall: 0.68, f1: 0.596, support: 25 },
    { category: "Shopping", precision: 0.565, recall: 0.52, f1: 0.542, support: 25 },
    { category: "Rent & Utilities", precision: 0.818, recall: 0.36, f1: 0.5, support: 25 },
    { category: "Other", precision: 0.267, recall: 0.6, f1: 0.369, support: 20 },
    { category: "Entertainment", precision: 1.0, recall: 0.2, f1: 0.333, support: 20 },
  ],
} as const;

/** Source: ML_G_SELECTION_RECORD.json > abstention_policy. */
export const MLG_ABSTENTION = {
  minMargin: 0.02,
  abstainRatePct: 12.3,
  wrongRescued: 17,
  correctCost: 7,
  macroF1Cost: 0.0415,
} as const;

/** Source: ML_G_SELECTION_RECORD.json > structural_ambiguity_routing.FINAL_TEST. */
export const MLG_AMBIGUITY_ROUTING = {
  coveragePct: 100,
  falsePositiveRatePct: 0,
  previousFalsePositiveRatePct: 13.8,
} as const;

/**
 * The honest limits. Every item is something the evidence above genuinely
 * cannot rule out — none of it is hedging for its own sake.
 */
export const MLG_LIMITATIONS = [
  {
    title: "The evaluation corpus is fabricated, not real",
    body: "Every merchant in it was invented for the benchmark. The numbers on this page describe performance on that corpus and support no claim about real-world accuracy.",
  },
  {
    title: "Brand names with no descriptive word cannot be placed from text",
    body: "A description reading only ZENOVARA gives a text classifier nothing to work with. Those cases are deliberately left in the benchmark so the reported scores aren't flattering — and in the app they are exactly what abstention and your own corrections are for.",
  },
  {
    title: "Your real statements have no answer key",
    body: "A private bank export carries no category labels, so no accuracy figure can be computed on your own data — only diagnostics like how much of each description the model could actually read.",
  },
  {
    title: "The model never learns from your corrections automatically",
    body: "A correction is stored and reused for that merchant, on that bank. It never retrains the model. The model is fit offline and is identical on every request.",
  },
  {
    title: "PlainCents does not track income",
    body: "Credits and deposits are recognized and skipped on import. This is a spending tool, and every total on every screen is outflow only.",
  },
  {
    title: "There is no bank connection",
    body: "PlainCents reads CSV files you export yourself. It never connects to a bank, and no credentials are ever requested or stored.",
  },
  {
    title: "The forecast is a simple historical average",
    body: "It is the mean of your last three months per category. It knows nothing about a holiday, a move, or a one-off purchase, and it is only as good as the categories underneath it.",
  },
] as const;
