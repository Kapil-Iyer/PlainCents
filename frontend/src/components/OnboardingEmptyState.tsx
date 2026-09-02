import { Loader2, Lock, Sparkles, UploadCloud } from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/shared/EmptyState";
import { useToast } from "@/components/shared/Toast";
import { useLoadDemo } from "@/hooks/useDemo";
import { ApiError } from "@/types/common";

import { ProductWalkthrough } from "@/components/walkthrough/ProductWalkthrough";

interface OnboardingEmptyStateProps {
  title?: string;
  description?: string;
  /** Set to false on non-Dashboard EMPTY surfaces so the walkthrough only
   * appears once, on the primary first-open screen (PRD §10a). */
  showWalkthrough?: boolean;
}

/**
 * PRD §10a: the EMPTY-mode onboarding surface every core screen falls back
 * to when there's no data yet. Offers two clearly distinct paths — import
 * real data, or load demo data — never blended into one action, so the
 * DEMO/REAL mutual-exclusion rule is visible in the UI itself, not just
 * enforced server-side.
 *
 * On the Dashboard (the primary first-open screen) this also carries the
 * Phase 10 recruiter/product walkthrough — a presentation-only preview of
 * the five major product areas, so a reviewer can understand PlainCents in
 * under a minute without importing anything first. It is deliberately not
 * the same action as "Load demo data": the walkthrough never touches
 * application state.
 */
export function OnboardingEmptyState({
  title = "Welcome to PlainCents",
  description = "A local-first personal finance MVP: import your own bank transactions, or load sample demo data, to see spending, forecasts, and portfolio tracking populated right away.",
  showWalkthrough = true,
}: OnboardingEmptyStateProps) {
  const loadDemoMutation = useLoadDemo();
  const { toast } = useToast();

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

  return (
    <div className="flex flex-col gap-8">
      <EmptyState
        icon={Sparkles}
        title={title}
        description={description}
        action={
          <div className="flex flex-col items-center gap-3">
            <div className="flex flex-wrap justify-center gap-2">
              <Button asChild>
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

      {showWalkthrough && (
        <div className="flex flex-col items-center gap-3">
          <p className="text-sm font-medium text-muted-foreground">See what PlainCents does</p>
          <ProductWalkthrough />
        </div>
      )}
    </div>
  );
}
