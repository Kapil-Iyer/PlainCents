/**
 * Claim-safety qualifiers, transcribed from reports/ml/ML_E_CLAIM_MATRIX.json.
 * Anything listed in NOT_SUPPORTED_CLAIMS must never be rendered as an
 * asserted product claim anywhere in the How It Works page (or elsewhere).
 * The SUPPORTED_WITH_QUALIFICATION strings below are the "safest_wording"
 * text from that file, verbatim — they carry their own limitation and are
 * meant to be shown on-card, not hidden in a tooltip.
 */

export const NOT_SUPPORTED_CLAIMS = [
  "PlainCents categorizes real-world bank transactions at 42.2% accuracy.",
  "PlainCents forecasts real-world personal spending at approximately 18.9% WAPE.",
  "TD CSV support is verified against a real TD export.",
  "PlainCents uses real bank transaction data for ML evaluation.",
  "PlainCents automatically retrains from user corrections.",
  "V2 categorization improved over V1.",
] as const;

export const CATEGORIZATION_EVIDENCE_QUALIFIER =
  "42.2% accuracy on a curated Tier B benchmark (not real-world bank data).";

export const FORECASTING_EVIDENCE_QUALIFIER =
  "18.9% WAPE on a synthetic 3-month reserved evaluation period — a mechanism/behavior check, not a real-world accuracy figure.";

export const TD_IMPORT_QUALIFIER =
  "TD import is tested against synthetic fixtures shaped like TD's publicly documented export columns — not field-verified against a real export.";

export const RETRAINING_QUALIFIER =
  "There is no online learning: a correction only ever writes confirmed_category. The categorization model artifact is fit offline and never refit at request time or on correction — verified directly by tests asserting predict()/predict_batch() never call .fit()/.fit_transform().";
