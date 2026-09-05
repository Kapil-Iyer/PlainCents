import { Hourglass } from "lucide-react";

import { EmptyState } from "@/components/shared/EmptyState";

interface ColdStartStateProps {
  monthsAvailable: number;
  monthsRequired: number;
}

/** Cold start is a normal product state, not an error — the API returns it
 * as a 200 (TRD Section 5.6), and this explains how many months the user has
 * versus needs.
 *
 * `monthsRequired` is read from the API rather than hardcoded, and it is
 * three: the minimum history the selected 3-month rolling-mean method needs
 * to compute one full window. The copy below says exactly that and claims
 * nothing more — three months has NOT been shown to forecast as accurately
 * as six or twelve. */
export function ColdStartState({ monthsAvailable, monthsRequired }: ColdStartStateProps) {
  const remaining = Math.max(monthsRequired - monthsAvailable, 0);
  return (
    <EmptyState
      icon={Hourglass}
      title="Not enough history yet"
      description={`Forecasting becomes available after ${monthsRequired} completed months — the minimum history the 3-month rolling-mean method needs. You have ${monthsAvailable} so far, so importing ${remaining === 1 ? "1 more month" : `${remaining} more months`} of statements will unlock it automatically.`}
    />
  );
}
