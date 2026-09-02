import { Hourglass } from "lucide-react";

import { EmptyState } from "@/components/shared/EmptyState";

interface ColdStartStateProps {
  monthsAvailable: number;
  monthsRequired: number;
}

/** TRD Section 12.5 / PRD Section 21: forecasting needs 12 unique calendar
 * months of history. Explains how many the user has vs. needs — never an
 * error screen, since cold_start is a normal 200 status (TRD Section 5.6). */
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
