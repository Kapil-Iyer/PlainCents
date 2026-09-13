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
 * Toast lives in the active child only (same pattern as DemoBanner) so
 * non-EMPTY routes never require a toast context for a null render.
 */
export function DemoReentryBanner() {
  const { mode, loadDemo, isLoadingDemo } = useAppState();
  const location = useLocation();

  if (mode !== "EMPTY") return null;
  if (location.pathname === "/dashboard") return null;

  return <DemoReentryBannerActive loadDemo={loadDemo} isLoadingDemo={isLoadingDemo} />;
}

function DemoReentryBannerActive({
  loadDemo,
  isLoadingDemo,
}: {
  loadDemo: () => Promise<unknown>;
  isLoadingDemo: boolean;
}) {
  const { toast } = useToast();

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
    <div className="flex shrink-0 flex-wrap items-center justify-center gap-2 border-b border-primary/20 bg-primary/10 px-4 py-2 text-sm font-medium text-primary sm:gap-3">
      <span className="flex items-center gap-2">
        <Sparkles className="h-4 w-4 shrink-0" />
        No data yet. Load sample demo data to explore PlainCents.
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
