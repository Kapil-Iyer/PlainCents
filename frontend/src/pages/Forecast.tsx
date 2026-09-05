import { LineChart, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { ForecastAccuracyCard } from "@/components/analytics/ForecastAccuracyCard";
import { EmptyState } from "@/components/shared/EmptyState";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/components/shared/Toast";
import { useForecastStatus, useLatestForecast, useRunForecast } from "@/hooks/useForecast";
import { ApiError } from "@/types/common";
import { hasForecastRun } from "@/types/forecast";

import { CategoryForecastList } from "@/pages/forecast/CategoryForecastList";
import { ColdStartState } from "@/pages/forecast/ColdStartState";
import { ForecastChart } from "@/pages/forecast/ForecastChart";
import { ForecastMetadata } from "@/pages/forecast/ForecastMetadata";
import { RunForecastButton } from "@/pages/forecast/RunForecastButton";
import { StaleWarning } from "@/pages/forecast/StaleWarning";

export function ForecastPage() {
  const { toast } = useToast();
  const { data: status, isLoading: statusLoading, isError: statusError } = useForecastStatus();
  const isEligible = !!status && status.status !== "cold_start";

  const { data: latest, isLoading: latestLoading } = useLatestForecast({ enabled: isEligible });
  const runForecastMutation = useRunForecast();

  const hasRun = hasForecastRun(latest);

  const handleRun = () => {
    runForecastMutation.mutate(undefined, {
      onError: (err) => {
        if (err instanceof ApiError && err.error === "cold_start") {
          toast({
            title: "Not enough history yet",
            description: err.message,
            variant: "destructive",
          });
        } else {
          toast({
            title: "Couldn't generate the forecast",
            description: err instanceof ApiError ? err.message : "Please try again.",
            variant: "destructive",
          });
        }
      },
    });
  };

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Forecast</h1>
          <p className="text-sm text-muted-foreground">
            A 3-month spending forecast by category, generated on demand.
          </p>
          <Link
            to="/how-it-works#forecasting"
            className="mt-1 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
          >
            <Sparkles className="h-3 w-3" />
            Why this model?
          </Link>
        </div>
        {isEligible && (
          <RunForecastButton
            hasExistingForecast={hasRun}
            isPending={runForecastMutation.isPending}
            onClick={handleRun}
          />
        )}
      </div>

      {statusLoading ? (
        <ForecastSkeleton />
      ) : statusError || !status ? (
        <EmptyState
          icon={LineChart}
          title="Couldn't load the forecast"
          description="Something went wrong talking to the server. Try refreshing the page."
        />
      ) : status.status === "cold_start" ? (
        <ColdStartState monthsAvailable={status.months_available} monthsRequired={status.months_required} />
      ) : latestLoading ? (
        <ForecastSkeleton />
      ) : (
        <div className="flex flex-col gap-5">
          {status.is_stale && <StaleWarning />}

          {!hasRun ? (
            <EmptyState
              icon={LineChart}
              title="No forecast yet"
              description="Click Generate forecast to see your predicted spending for the next 3 months, by category."
            />
          ) : (
            <>
              <ForecastMetadata generatedAt={latest.generated_at} monthsAvailable={latest.months_available} />
              <ForecastChart predictions={latest.predictions} />
              <CategoryForecastList predictions={latest.predictions} />
              {/* Deliberately rendered even before any forecast history
               * exists: its empty state explains WHY there is nothing to
               * show yet, which is more useful than the card silently not
               * being there. */}
              <ForecastAccuracyCard />
            </>
          )}
        </div>
      )}
    </div>
  );
}

function ForecastSkeleton() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-4 w-40" />
      </CardHeader>
      <CardContent>
        <Skeleton className="h-48 w-full" />
      </CardContent>
    </Card>
  );
}
