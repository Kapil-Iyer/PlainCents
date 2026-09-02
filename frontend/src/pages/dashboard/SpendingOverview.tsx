import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn, formatCurrency, formatMonthLabel } from "@/lib/utils";
import type { DashboardSummaryResponse } from "@/types/dashboard";

interface SpendingOverviewProps {
  summary: DashboardSummaryResponse;
}

/**
 * PRD §11.7: "current-period spending total, comparison to the prior
 * period." A spend increase is flagged (not necessarily bad, but worth
 * noticing) in warning tone; a decrease in success tone — mirroring the
 * refund/spend color convention already used in TransactionTable.
 */
export function SpendingOverview({ summary }: SpendingOverviewProps) {
  const { period, total_spend_current, total_spend_previous, change_pct } = summary;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <Card>
        <CardHeader>
          <CardTitle className="text-muted-foreground">{formatMonthLabel(period.current)}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-semibold tabular-nums">{formatCurrency(total_spend_current)}</p>
          <p className="mt-1 text-xs text-muted-foreground">Total spend this month</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-muted-foreground">{formatMonthLabel(period.previous)}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-semibold tabular-nums">{formatCurrency(total_spend_previous)}</p>
          <p className="mt-1 text-xs text-muted-foreground">Total spend last month</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-muted-foreground">Change</CardTitle>
        </CardHeader>
        <CardContent>
          <ChangeIndicator changePct={change_pct} />
          <p className="mt-1 text-xs text-muted-foreground">Vs. last month</p>
        </CardContent>
      </Card>
    </div>
  );
}

function ChangeIndicator({ changePct }: { changePct: number | null }) {
  if (changePct === null) {
    return <p className="text-2xl font-semibold text-muted-foreground">—</p>;
  }

  const isIncrease = changePct > 0;
  const isFlat = changePct === 0;
  const Icon = isFlat ? Minus : isIncrease ? ArrowUpRight : ArrowDownRight;

  return (
    <p
      className={cn(
        "flex items-center gap-1 text-2xl font-semibold tabular-nums",
        isFlat ? "text-muted-foreground" : isIncrease ? "text-warning" : "text-success",
      )}
    >
      <Icon className="h-5 w-5" />
      {Math.abs(changePct).toFixed(1)}%
    </p>
  );
}
