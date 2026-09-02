import { AlertTriangle } from "lucide-react";

/** TRD Section 12.4: a stored (not derived) is_stale flag — surfaced as a
 * banner, not silently hidden, so the user knows the numbers below may no
 * longer reflect their current transaction history. */
export function StaleWarning() {
  return (
    <div className="flex items-start gap-2 rounded-md border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <p>
        This forecast may be out of date — your transactions have changed since it was generated.
        Refresh it to see updated predictions.
      </p>
    </div>
  );
}
