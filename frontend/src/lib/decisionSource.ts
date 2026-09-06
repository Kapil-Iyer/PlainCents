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
