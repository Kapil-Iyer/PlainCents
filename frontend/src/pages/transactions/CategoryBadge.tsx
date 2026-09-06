import { Loader2, Sparkles, UserCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/shared/Toast";
import { describeDecisionSource, getCategorySuggestion } from "@/lib/decisionSource";
import { useUpdateTransaction } from "@/hooks/useTransactions";
import { ApiError } from "@/types/common";
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
 *
 * A low-confidence abstention additionally gets an ADVISORY "Suggested:
 * {model_category}" chip with a one-click "Use" accept -- never a
 * confidence percentage (decision margin is an abstention-policy threshold,
 * not a calibrated 0-100% score). Accepting it is a NORMAL human
 * correction: it writes confirmed_category through the exact same PATCH
 * path TransactionFormDialog uses, never a second write path, and it stays
 * advisory metadata (model_category) forever after -- accepting it does not
 * retroactively change what the system itself predicted.
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

  return <SystemCategoryBadge transaction={transaction} />;
}

/** Hooks live here, not in the gate above -- is_manual_override's early
 * return must stay unconditional (Rules of Hooks), the same pattern
 * DemoBanner/DemoReentryBanner already use for their own gated mounts. */
function SystemCategoryBadge({ transaction }: { transaction: TransactionResponse }) {
  const note = describeDecisionSource(transaction.decision_source);
  const suggestion = getCategorySuggestion({
    decisionSource: transaction.decision_source,
    modelCategory: transaction.model_category,
    predictedCategory: transaction.predicted_category,
  });
  const updateMutation = useUpdateTransaction();
  const { toast } = useToast();

  const handleUse = async () => {
    if (!suggestion) return;
    try {
      await updateMutation.mutateAsync({
        id: transaction.id,
        payload: { confirmed_category: suggestion },
      });
      toast({ title: `Category set to ${suggestion}` });
    } catch (err) {
      toast({
        title: "Couldn't update category",
        description: err instanceof ApiError ? err.message : "Please try again.",
        variant: "destructive",
      });
    }
  };

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
      {suggestion && (
        <span className="flex items-center gap-1 text-xs text-muted-foreground">
          Suggested: {suggestion}
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs"
            onClick={handleUse}
            disabled={updateMutation.isPending}
          >
            {updateMutation.isPending && <Loader2 className="h-3 w-3 animate-spin" />}
            Use {suggestion}
          </Button>
        </span>
      )}
    </span>
  );
}
