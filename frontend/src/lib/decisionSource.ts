/**
 * Shared "why is this categorized the way it is" copy for decision_source,
 * used identically in Preview (ImportPreviewCard.tsx) and confirmed
 * transaction views (CategoryBadge.tsx) -- one source of truth so the two
 * never drift, the same way backend/services/category_decision.py is the
 * one decision path both Preview and Confirm call.
 *
 * "ambiguous_e_transfer" (backend/services/e_transfer_policy.py) is
 * deliberately distinguished from "structural_other" here: the former is a
 * purposeless E-Transfer (a recipient name survives, but no spending
 * purpose does); the latter is text that names nothing at all (a bare ATM
 * withdrawal, a generic transfer). Both serve `effective_category = "Other"`
 * when not manually corrected, but the reason a customer sees should not
 * be the same for the two.
 */
export type DecisionSource =
  | "model"
  | "structural_other"
  | "low_confidence_other"
  | "gazetteer"
  | "ambiguous_e_transfer"
  | null
  | undefined;

export interface DecisionSourceNote {
  /** Short label for a compact badge/caption. */
  label: string;
  /** Longer explanation, suitable for a title/tooltip. */
  explanation: string;
}

export function describeDecisionSource(source: DecisionSource): DecisionSourceNote | null {
  switch (source) {
    case "structural_other":
      return {
        label: "no merchant name",
        explanation: "No merchant name in this description, so there's nothing to categorize",
      };
    case "ambiguous_e_transfer":
      return {
        label: "E-Transfer",
        explanation: "Purpose could not be determined from the bank description",
      };
    case "low_confidence_other":
      return {
        label: "low confidence",
        explanation: "Not confident enough to guess — you can set this yourself",
      };
    case "gazetteer":
      return {
        label: "recognized merchant",
        explanation: "Recognized public merchant/service, not a model guess",
      };
    default:
      return null;
  }
}

/**
 * The model's best non-abstained guess, shown as ADVISORY information only
 * when the system abstained to "Other" for lack of confidence -- never a
 * calibrated confidence percentage (decision margin is an abstention-policy
 * threshold, not a 0-100% probability claim; see backend/services/
 * categorization_service.py). One source of truth for "should a suggestion
 * chip appear at all", shared by ImportPreviewCard (display-only -- no
 * transaction id exists yet to act on) and CategoryBadge (display + a
 * one-click "Use" that becomes a normal human correction via the existing
 * PATCH confirmed_category path -- never a second write path).
 *
 * Gating, in order:
 *   - only for decision_source === "low_confidence_other" -- structural_other
 *     and ambiguous_e_transfer never call the model at all (CategoryDecision
 *     .model_category is already None on those paths), so there is nothing
 *     to suggest; a plain "model"/"gazetteer" decision, or any row a human
 *     has already corrected (is_manual_override), isn't a suggestion
 *     candidate either -- callers gate is_manual_override themselves, the
 *     same way they already do for describeDecisionSource's caption.
 *   - model_category must be present (a pre-migration row, or a caller that
 *     hasn't loaded it, has nothing to suggest)
 *   - model_category !== "Other" -- suggesting the same category the row is
 *     already showing is not a suggestion
 *   - model_category !== the served predicted_category -- guards the same
 *     no-op case defensively even though a low_confidence_other row's
 *     predicted_category is always "Other" today
 */
export function getCategorySuggestion(params: {
  decisionSource: DecisionSource;
  modelCategory?: string | null;
  predictedCategory?: string | null;
}): string | null {
  const { decisionSource, modelCategory, predictedCategory } = params;
  if (decisionSource !== "low_confidence_other") return null;
  if (!modelCategory) return null;
  if (modelCategory === "Other") return null;
  if (modelCategory === predictedCategory) return null;
  return modelCategory;
}
