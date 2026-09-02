import { Loader2, Sparkles, UploadCloud } from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/shared/EmptyState";
import { useToast } from "@/components/shared/Toast";
import { useLoadDemo } from "@/hooks/useDemo";
import { ApiError } from "@/types/common";

interface OnboardingEmptyStateProps {
  title?: string;
  description?: string;
}

/**
 * PRD §10a: the EMPTY-mode onboarding surface every core screen falls back
 * to when there's no data yet. Offers two clearly distinct paths — import
 * real data, or load demo data — never blended into one action, so the
 * DEMO/REAL mutual-exclusion rule is visible in the UI itself, not just
 * enforced server-side.
 */
export function OnboardingEmptyState({
  title = "Welcome to PlainCents",
  description = "Import your own transactions, or load sample demo data to see Dashboard, Forecast, and Portfolio populated right away.",
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
    <EmptyState
      icon={Sparkles}
      title={title}
      description={description}
      action={
        <div className="flex gap-2">
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
      }
    />
  );
}
