import { LayoutDashboard, Plus, UploadCloud } from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import { Skeleton } from "@/components/ui/skeleton";
import { useDashboardSummary } from "@/hooks/useDashboard";
import { formatMonthLabel } from "@/lib/utils";

import { CategoryBreakdown } from "@/pages/dashboard/CategoryBreakdown";
import { RecentTransactions } from "@/pages/dashboard/RecentTransactions";
import { SpendingOverview } from "@/pages/dashboard/SpendingOverview";
import { SpendingTrend } from "@/pages/dashboard/SpendingTrend";

export function DashboardPage() {
  const { data, isLoading, isError } = useDashboardSummary();

  return (
    <div className="flex flex-col gap-5">
      <div>
        <h1 className="text-xl font-semibold">Dashboard</h1>
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
        <EmptyState
          icon={LayoutDashboard}
          title="No data yet"
          description="Import a bank statement or add a transaction manually to see your spending here."
          action={
            <div className="flex gap-2">
              <Button asChild>
                <Link to="/import">
                  <UploadCloud className="h-4 w-4" />
                  Import transactions
                </Link>
              </Button>
              <Button variant="outline" asChild>
                <Link to="/transactions">
                  <Plus className="h-4 w-4" />
                  Add manually
                </Link>
              </Button>
            </div>
          }
        />
      ) : (
        <div className="flex flex-col gap-5">
          <SpendingOverview summary={data} />

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
