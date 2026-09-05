/**
 * Claims PlainCents deliberately does NOT make.
 *
 * This list is not decoration. Each entry is a sentence that sounds
 * reasonable, that the evidence in this repository does NOT support, and
 * that a reader could easily infer from the numbers shown elsewhere on the
 * How It Works page if nobody said otherwise. They are rendered on the page
 * for exactly that reason.
 *
 * Rule: nothing here may ever appear as an asserted claim anywhere in the
 * product. If a future phase produces evidence for one of these, move it out
 * of this list in the same change that adds the evidence — never before.
 *
 * ML-G update: the figures quoted below were refreshed from
 * reports/ml/ML_G_SELECTION_RECORD.json. The claims themselves are unchanged
 * in kind — a better model does not make a fabricated benchmark into
 * real-world evidence.
 */

export const NOT_SUPPORTED_CLAIMS = [
  "PlainCents categorizes real-world bank transactions at 58% accuracy.",
  "PlainCents is more accurate than your bank's own categorization.",
  "PlainCents forecasts real-world personal spending to within 18%.",
  "Three months of history forecasts as accurately as six or twelve.",
  "PlainCents learns from your corrections and retrains itself.",
  "Bank CSV support is verified against real exports from every supported bank.",
] as const;

export const CATEGORIZATION_EVIDENCE_QUALIFIER =
  "0.58 macro-F1 on a sanitized, hand-authored benchmark of Canadian-bank-style descriptions with fabricated merchant names — measured on merchants held out of training entirely. It is not a real-world accuracy figure, and no real-world figure can be computed: private bank exports carry no category labels to check against.";

export const FORECASTING_EVIDENCE_QUALIFIER =
  "The forecast method (a three-month average per category) was selected on a synthetic 24-month grid using walk-forward validation. That is a mechanism check, not a real-world accuracy figure.";

export const THREE_MONTH_MINIMUM_QUALIFIER =
  "Three completed months is the mathematical minimum for a three-month average — exactly one full window. It has not been shown to forecast as accurately as six, nine or twelve months; the history-length experiments never tested a three-month history at all.";

export const BANK_IMPORT_QUALIFIER =
  "Import is tested against synthetic fixtures shaped like each bank's documented export columns, plus a read-only structural audit of real RBC and Scotiabank exports. Individual banks change their export format without notice.";

export const RETRAINING_QUALIFIER =
  "There is no online learning. A correction only ever writes your category for that merchant on that bank; the model artifact is fit offline and is byte-identical on every request — verified by tests asserting the inference path never calls fit().";
