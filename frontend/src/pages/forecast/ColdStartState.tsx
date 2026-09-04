import { Hourglass } from "lucide-react";

import { EmptyState } from "@/components/shared/EmptyState";

interface ColdStartStateProps {
  monthsAvailable: number;
  monthsRequired: number;
}

/** TRD Section 12.5 / PRD Section 21 (ML-F amendment: 12 -> 6 unique
 * calendar months, reports/ml/ML_F_SELECTION_RECORD.json): forecasting
 * needs `monthsRequired` unique calendar months of history, read from the
 * API rather than hardcoded here. Explains how many the user has vs.
 * needs — never an error screen, since cold_start is a normal 200 status
 * (TRD Section 5.6). */
export function ColdStartState({ monthsAvailable, monthsRequired }: ColdStartStateProps) {
  const remaining = Math.max(monthsRequired - monthsAvailable, 0);
  return (
    <EmptyState
      icon={Hourglass}
      title="Not enough history yet"
      description={`Forecasting needs at least ${monthsRequired} months of transaction history. You have ${monthsAvailable} so far — keep importing statements or adding transactions, and this will unlock automatically once ${remaining === 1 ? "1 more month is" : `${remaining} more months are`} covered.`}
    />
  );
}
