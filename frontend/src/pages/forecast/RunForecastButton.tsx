import { Loader2, RefreshCw, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";

interface RunForecastButtonProps {
  hasExistingForecast: boolean;
  isPending: boolean;
  onClick: () => void;
}

/** TRD Section 9.8: loading state required for explicit forecast generation.
 * Label reflects whether this is the first generation or a refresh of an
 * existing (possibly stale) run. */
export function RunForecastButton({ hasExistingForecast, isPending, onClick }: RunForecastButtonProps) {
  return (
    <Button onClick={onClick} disabled={isPending}>
      {isPending ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          Generating…
        </>
      ) : hasExistingForecast ? (
        <>
          <RefreshCw className="h-4 w-4" />
          Refresh forecast
        </>
      ) : (
        <>
          <Sparkles className="h-4 w-4" />
          Generate forecast
        </>
      )}
    </Button>
  );
}
