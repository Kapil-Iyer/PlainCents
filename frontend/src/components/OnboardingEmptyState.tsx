import { Compass, Loader2, Lock, Sparkles, UploadCloud } from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/shared/EmptyState";
import { useToast } from "@/components/shared/Toast";
import { useGuidedTour } from "@/context/GuidedTourContext";
import { useLoadDemo } from "@/hooks/useDemo";
import { ApiError } from "@/types/common";

interface OnboardingEmptyStateProps {
  title?: string;
  description?: string;
  /** Set to false on non-Dashboard EMPTY surfaces so the "Load demo data &
   * start tour" entry point only appears once, on the primary first-open
   * screen (PRD §10a) — it stays reachable everywhere else via TopNav's
   * Replay tour (once Demo/Real data already exists). */
  showWalkthrough?: boolean;
}

/**
 * PRD §10a: the EMPTY-mode onboarding surface every core screen falls back
 * to when there's no data yet. Offers two clearly distinct data paths --
 * import real data, or load demo data -- never blended into one action, so
 * the DEMO/REAL mutual-exclusion rule is visible in the UI itself, not just
 * enforced server-side.
 *
 * On the Dashboard (the primary first-open screen) this also offers a
 * guided spotlight tour over the real app (PATCH C) — a reviewer can
 * understand PlainCents by seeing the actual product navigate itself,
 * rather than a static mockup. The tour spotlights real charts that only
 * exist once data is loaded, so its entry point always loads Demo data
 * FIRST and starts the tour only after that succeeds -- never against an
 * empty app. Loading Demo data plain (no tour) stays available separately.
 */
export function OnboardingEmptyState({
  title = "Welcome to PlainCents",
  description = "A local-first personal finance MVP: import your own bank transactions, or load sample demo data, to see spending, forecasts, and portfolio tracking populated right away.",
  showWalkthrough = true,
}: OnboardingEmptyStateProps) {
  const loadDemoMutation = useLoadDemo();
  const { toast } = useToast();
  const { start: startTour } = useGuidedTour();

  const handleLoadDemo = async () => {
    try {
      await loadDemoMutation.mutateAsync();
      toast({ title: "Demo data loaded" });
    } catch (err) {
      toast({
        title: "Couldn't load demo data",
        description: err instanceof ApiError ? err.message : "Please try again.",
        variant: "destructive",
      });
    }
  };

  // The tour spotlights real charts (Spending Pace, What Changed, ...) that
  // simply don't exist yet in EMPTY mode -- starting it here without data
  // loaded first left every chart step falling back to an un-anchored
  // centered card. Load Demo data BEFORE starting, so every step has a real
  // element to spotlight from its very first frame.
  const handleLoadDemoAndStartTour = async () => {
    try {
      await loadDemoMutation.mutateAsync();
      startTour();
    } catch (err) {
      toast({
        title: "Couldn't load demo data",
        description: err instanceof ApiError ? err.message : "Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="flex flex-col gap-8">
      <EmptyState
        icon={Sparkles}
        title={title}
        description={description}
        action={
          <div className="flex flex-col items-center gap-3">
            <div className="flex flex-wrap justify-center gap-2">
              {showWalkthrough && (
                <Button onClick={handleLoadDemoAndStartTour} disabled={loadDemoMutation.isPending}>
                  {loadDemoMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Compass className="h-4 w-4" />
                  )}
                  Load demo data &amp; start tour
                </Button>
              )}
              <Button asChild variant={showWalkthrough ? "outline" : "default"}>
                <Link to="/import">
                  <UploadCloud className="h-4 w-4" />
                  Import real data
                </Link>
              </Button>
              <Button variant="outline" onClick={handleLoadDemo} disabled={loadDemoMutation.isPending}>
                {loadDemoMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="h-4 w-4" />
                )}
                Load demo data
              </Button>
            </div>
            <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Lock className="h-3 w-3" />
              Runs locally on your machine — no signup, no account, no data leaves your computer
              unless a Portfolio price refresh is requested.
            </p>
          </div>
        }
      />
    </div>
  );
}
