import { Loader2, Sparkles } from "lucide-react";
import { useLocation } from "react-router-dom";

import { useToast } from "@/components/shared/Toast";
import { useAppState } from "@/context/AppStateContext";
import { ApiError } from "@/types/common";

/**
 * Companion to DemoBanner (Build Plan §2.5 / TRD §9.7's mode table, extended):
 * DEMO -> DemoBanner shown; EMPTY -> this banner, everywhere except the
 * Dashboard route (which already carries the fuller OnboardingEmptyState
 * "Load demo data" / "Import real data" pair); REAL -> neither banner.
 *
 * Root-cause fix: clearing demo from any non-Dashboard page (DemoBanner's
 * "Clear demo data" is global chrome, reachable from every route) correctly
 * flips app_state.mode to EMPTY, but only Dashboard.tsx ever rendered a
 * "Load demo data" action — every other page falls back to its own generic
 * EmptyState with no such action. A user who cleared demo from, say,
 * Forecast or Transactions had no discoverable way back into Demo mode
 * short of already knowing to click "Dashboard" in the sidebar themselves.
 * This banner closes that gap without duplicating Dashboard's own richer
 * onboarding surface.
 */
export function DemoReentryBanner() {
  const { mode, loadDemo, isLoadingDemo } = useAppState();
  const { toast } = useToast();
  const location = useLocation();

  if (mode !== "EMPTY") return null;
  if (location.pathname === "/dashboard") return null;

  const handleLoad = async () => {
    try {
      await loadDemo();
      toast({ title: "Demo data loaded" });
    } catch (err) {
      toast({
        title: "Couldn't load demo data",
        description: err instanceof ApiError ? err.message : "Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="flex flex-wrap items-center justify-center gap-2 border-b border-primary/20 bg-primary/10 px-4 py-2 text-sm font-medium text-primary sm:gap-3">
      <span className="flex items-center gap-2">
        <Sparkles className="h-4 w-4 shrink-0" />
        No data yet — load sample demo data to explore PlainCents.
      </span>
      <button
        type="button"
        onClick={handleLoad}
        disabled={isLoadingDemo}
        className="inline-flex items-center gap-1.5 rounded-md border border-primary/40 px-2 py-0.5 text-xs font-semibold text-primary transition-colors hover:bg-primary/15 disabled:pointer-events-none disabled:opacity-50"
      >
        {isLoadingDemo && <Loader2 className="h-3 w-3 animate-spin" />}
        Load demo data
      </button>
    </div>
  );
}
