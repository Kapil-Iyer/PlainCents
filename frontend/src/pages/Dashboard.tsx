import { LayoutDashboard } from "lucide-react";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import { OnboardingEmptyState } from "@/components/OnboardingEmptyState";
import { Skeleton } from "@/components/ui/skeleton";
import { useDashboardSummary } from "@/hooks/useDashboard";
import { formatMonthLabel } from "@/lib/utils";

import { CategoryMoversCard } from "@/components/analytics/CategoryMoversCard";
import { SpendPaceCard } from "@/components/analytics/SpendPaceCard";
import { CategoryBreakdown } from "@/pages/dashboard/CategoryBreakdown";
import { RecentTransactions } from "@/pages/dashboard/RecentTransactions";
import { SpendingOverview } from "@/pages/dashboard/SpendingOverview";
import { SpendingTrend } from "@/pages/dashboard/SpendingTrend";

export function DashboardPage() {
  const { data, isLoading, isError } = useDashboardSummary();

  return (
    <div className="flex flex-col gap-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-sm text-muted-foreground">
          {data
            ? `${formatMonthLabel(data.period.current)} vs. ${formatMonthLabel(data.period.previous)}`
            : "Your spending at a glance."}
        </p>
      </div>

      {isLoading ? (
        <DashboardSkeleton />
      ) : isError ? (
        <EmptyState
          icon={LayoutDashboard}
          title="Couldn't load the dashboard"
          description="Something went wrong talking to the server. Try refreshing the page."
        />
      ) : !data ? null : data.data_mode === "EMPTY" ? (
        <OnboardingEmptyState />
      ) : (
        <div className="flex flex-col gap-5">
          <SpendingOverview summary={data} />

          {/* Dashboard answers "how am I doing right now?". The two cards
           * below are the pair that actually answer it: am I on pace, and
           * what moved. Deeper category and merchant analysis lives on
           * Transactions, where the underlying rows are. */}
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            <SpendPaceCard />
            <CategoryMoversCard />
          </div>

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            <CategoryBreakdown items={data.category_breakdown} />
            <SpendingTrend points={data.spending_trend} />
          </div>

          <RecentTransactions transactions={data.recent_transactions} />
        </div>
      )}
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Card key={i}>
            <CardHeader>
              <Skeleton className="h-4 w-24" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-8 w-32" />
            </CardContent>
          </Card>
        ))}
      </div>
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <Skeleton className="h-4 w-40" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-48 w-full" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <Skeleton className="h-4 w-40" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-48 w-full" />
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader>
          <Skeleton className="h-4 w-40" />
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-6 w-full" />
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
