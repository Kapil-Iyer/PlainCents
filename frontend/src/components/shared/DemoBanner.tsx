import { useState } from "react";
import { FlaskConical, Loader2 } from "lucide-react";

import { ConfirmDialog } from "@/components/shared/ConfirmDialog";
import { useToast } from "@/components/shared/Toast";
import { useAppState } from "@/context/AppStateContext";
import { ApiError } from "@/types/common";

/**
 * Build Plan §2.5 / TRD §9.7 corrected mode table: DEMO -> banner shown,
 * EMPTY -> no banner (onboarding empty-state instead), REAL -> no banner.
 *
 * Phase 11B: also the DEMO-mode chrome surface for a direct "Clear demo
 * data" action — this is a frontend-only exposure of the existing
 * useClearDemo()/DELETE /api/demo/clear path already used by
 * DemoConflictDialog (TRD §14.3/§14.4), not a new capability. It renders
 * full-width regardless of viewport, so it (and How It Works, via TopNav)
 * stays reachable on mobile even though Sidebar is desktop-only.
 */
export function DemoBanner() {
  const { mode, clearDemo, isClearingDemo } = useAppState();
  const { toast } = useToast();
  const [confirmOpen, setConfirmOpen] = useState(false);

  if (mode !== "DEMO") return null;

  const handleClear = async () => {
    try {
      await clearDemo();
      toast({ title: "Demo data cleared" });
    } catch (err) {
      toast({
        title: "Couldn't clear demo data",
        description: err instanceof ApiError ? err.message : "Please try again.",
        variant: "destructive",
      });
      throw err;
    }
  };

  return (
    <div className="flex flex-wrap items-center justify-center gap-2 border-b border-warning/20 bg-warning/10 px-4 py-2 text-sm font-medium text-warning sm:gap-3">
      <span className="flex items-center gap-2">
        <FlaskConical className="h-4 w-4 shrink-0" />
        Demo Data — everything you see is sample data, not your own.
      </span>
      <button
        type="button"
        onClick={() => setConfirmOpen(true)}
        disabled={isClearingDemo}
        className="inline-flex items-center gap-1.5 rounded-md border border-warning/40 px-2 py-0.5 text-xs font-semibold text-warning transition-colors hover:bg-warning/15 disabled:pointer-events-none disabled:opacity-50"
      >
        {isClearingDemo && <Loader2 className="h-3 w-3 animate-spin" />}
        Clear demo data
      </button>

      <ConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        title="Clear demo data?"
        description="This resets the sample data and returns the app to its empty state. This can't be undone — you can always reload demo data afterward."
        confirmLabel="Clear demo data"
        onConfirm={handleClear}
      />
    </div>
  );
}
