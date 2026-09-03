/**
 * Human-in-the-loop terminology, verified directly against the backend
 * implementation (not paraphrased from memory):
 *   - db/migrations/001_initial_v2.sql: predicted_category (NOT NULL, set
 *     once), confirmed_category (nullable, set by a correction).
 *   - db/migrations (v_transactions_effective view):
 *     effective_category = COALESCE(confirmed_category, predicted_category);
 *     is_manual_override = (confirmed_category IS NOT NULL).
 *   - reports/ml/ML_E_CLAIM_MATRIX.json: "PlainCents preserves user
 *     corrections separately from model predictions" (SUPPORTED) and
 *     "PlainCents automatically retrains from user corrections"
 *     (NOT_SUPPORTED).
 */
export const HUMAN_IN_LOOP_STEPS = [
  {
    id: "predicted",
    label: "predicted_category",
    description: "Written once, at import/creation time, by the categorization model. Never overwritten.",
  },
  {
    id: "correction",
    label: "optional user correction",
    description: "You can override any prediction from Transactions. This writes confirmed_category — it never touches predicted_category.",
  },
  {
    id: "confirmed",
    label: "confirmed_category",
    description: "Present only if you've corrected this transaction. Nullable.",
  },
  {
    id: "effective",
    label: "effective_category = COALESCE(confirmed_category, predicted_category)",
    description: "What every chart, filter, and forecast actually uses — your correction wins whenever one exists.",
  },
] as const;

export const HUMAN_IN_LOOP_FACTS = [
  "The original model prediction is preserved forever, even after a correction — it's never overwritten, only superseded for display/analytics purposes.",
  "A confirmed category becomes authoritative for every downstream read (dashboard, forecasting, exports) the moment it exists.",
  "Corrections do NOT trigger automatic retraining. The categorization model is fit offline, ahead of time, and is never refit at request time or in response to a correction.",
];
