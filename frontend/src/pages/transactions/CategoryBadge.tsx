import { Sparkles, UserCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { TransactionResponse } from "@/types/transaction";

/**
 * TRD §9.8: predicted vs. corrected category must be visually distinguishable,
 * not tooltip-only — a dashed/muted badge with a sparkle icon for a
 * model prediction, a solid badge with a check icon once the user has
 * confirmed/corrected it.
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
  return (
    <Badge variant="predicted" title="Predicted by the categorization model">
      <Sparkles className="mr-1 h-3 w-3" />
      {transaction.effective_category}
    </Badge>
  );
}
