import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Compass, Sparkles } from "lucide-react";
import { NavLink } from "react-router-dom";

import { getHealth } from "@/api/health";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAppState } from "@/context/AppStateContext";
import { useGuidedTour } from "@/context/GuidedTourContext";
import { cn } from "@/lib/utils";

const MODE_LABEL: Record<string, string> = {
  EMPTY: "No data yet",
  DEMO: "Demo",
  REAL: "Live",
};

export function TopNav() {
  const { mode, loadDemo, isLoadingDemo } = useAppState();
  const { start: startTour } = useGuidedTour();
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 60_000,
  });

  // The tour spotlights real charts (Spending Pace, What Changed, ...) that
  // don't exist yet in EMPTY mode -- so Replay tour loads Demo data first
  // when there's nothing loaded at all, exactly like the onboarding
  // screen's "Load demo data & start tour". In DEMO or REAL mode there's
  // already real data to spotlight, so it starts immediately.
  const handleReplayTour = async () => {
    if (mode === "EMPTY") {
      try {
        await loadDemo();
      } catch {
        // Best-effort -- still start the tour rather than leaving this
        // button silently doing nothing; most steps still have a target
        // (page headers, nav items) even without Demo data loaded.
      }
    }
    startTour();
  };

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-card px-4 sm:px-5">
      {/* Sidebar is hidden below `md` — How It Works stays reachable on
       * mobile via this compact link instead of a full nav duplication. */}
      <NavLink
        to="/how-it-works"
        className={({ isActive }) =>
          cn(
            "flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground md:hidden",
            isActive && "bg-primary/10 text-primary",
          )
        }
      >
        <Sparkles className="h-3.5 w-3.5" />
        How It Works
      </NavLink>
      <div className="hidden md:block" />
      <div className="flex items-center gap-3">
        {health?.categorization_model && health.categorization_model !== "loaded" && (
          <span className="flex items-center gap-1.5 text-xs font-medium text-warning">
            <AlertTriangle className="h-3.5 w-3.5" />
            Categorization model unavailable
          </span>
        )}
        {/* Always available, from any screen -- the tour's own Skip/Done
         * never leaves a user without a way back in (Build Plan PATCH C:
         * "provide a Replay Tour entry point"). */}
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={handleReplayTour}
          disabled={isLoadingDemo}
          className="hidden items-center gap-1.5 text-muted-foreground sm:flex"
        >
          <Compass className="h-3.5 w-3.5" />
          Replay tour
        </Button>
        <Badge
          data-tour="topnav-mode-badge"
          variant={mode === "DEMO" ? "warning" : mode === "REAL" ? "success" : "outline"}
        >
          {MODE_LABEL[mode] ?? mode}
        </Badge>
      </div>
    </header>
  );
}
