import { useState } from "react";
import { Info, X } from "lucide-react";

const DISMISS_KEY = "plaincents:demoPriceNoticeDismissed";

/** Same try/catch-everywhere localStorage pattern as GuidedTourContext's
 * completion flag -- a private window or blocked site data must degrade to
 * "not dismissed yet", never throw. */
function readDismissed(): boolean {
  try {
    return window.localStorage.getItem(DISMISS_KEY) === "true";
  } catch {
    return false;
  }
}

function writeDismissed(): void {
  try {
    window.localStorage.setItem(DISMISS_KEY, "true");
  } catch {
    // Best-effort only -- a failed write just means this shows again next
    // visit, never a crash.
  }
}

/**
 * Answers "why does this say Jan 1, 2024?" before the user has to wonder --
 * shown only in DEMO mode, and only while at least one holding is still on
 * its seeded snapshot price (Portfolio.tsx decides that). Dismissible and
 * remembered locally, same as the guided tour's completion flag; reappears
 * for a fresh browser/profile, which is fine -- it's informational, not a
 * one-time-only notice.
 *
 * Deliberately a plain inline callout, not another ConfirmDialog/banner
 * pattern -- this is read-only information, nothing to confirm or clear.
 */
export function DemoPriceNotice() {
  const [dismissed, setDismissed] = useState(readDismissed);
  if (dismissed) return null;

  return (
    <div className="flex items-start gap-2.5 rounded-lg border border-border bg-elevated px-3 py-2.5 text-sm">
      <Info className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden />
      <p className="flex-1 leading-relaxed text-muted-foreground">
        Demo holdings start on a fixed sample price snapshot (dated Jan 1, 2024) so the demo
        works even without a market-data connection. Prices come from Yahoo Finance and are
        cached for up to an hour — select{" "}
        <span className="font-medium text-foreground">Refresh prices</span> above to request the
        latest available quotes.
      </p>
      <button
        type="button"
        onClick={() => {
          writeDismissed();
          setDismissed(true);
        }}
        aria-label="Dismiss"
        className="shrink-0 rounded p-0.5 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
