import { Sparkles, UserCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { describeDecisionSource } from "@/lib/decisionSource";
import type { TransactionResponse } from "@/types/transaction";

/**
 * TRD §9.8: predicted vs. corrected category must be visually distinguishable,
 * not tooltip-only — a dashed/muted badge with a sparkle icon for a
 * model prediction, a solid badge with a check icon once the user has
 * confirmed/corrected it.
 *
 * A SYSTEM-decided (not manually overridden) row additionally gets the same
 * short secondary caption Preview already shows (@/lib/decisionSource) --
 * most notably distinguishing a genuine miscellaneous "Other" from an
 * "Other" that's actually a purposeless E-Transfer whose purpose couldn't
 * be read from the bank description. A human correction (is_manual_override)
 * is always the authoritative answer, so no caption is shown once one
 * exists — decision_source only explains the SYSTEM's original reasoning.
 */
export function CategoryBadge({ transaction }: { transaction: TransactionResponse }) {
  if (transaction.is_manual_override) {
    return (
      <Badge variant="confirmed" title="Confirmed by you">
        <UserCheck className="mr-1 h-3 w-3" />
        {transaction.effective_category}
      </Badge>
    );
  }

  const note = describeDecisionSource(transaction.decision_source);

  return (
    <span className="flex flex-wrap items-center gap-1.5">
      <Badge variant="predicted" title="Predicted by the categorization model">
        <Sparkles className="mr-1 h-3 w-3" />
        {transaction.effective_category}
      </Badge>
      {note && (
        <span className="text-xs text-muted-foreground" title={note.explanation}>
          ({note.label})
        </span>
      )}
    </span>
  );
}
