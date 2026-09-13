import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn, formatCurrency, formatDayRangeLabel, formatMonthLabel } from "@/lib/utils";
import type { DashboardSummaryResponse } from "@/types/dashboard";

interface SpendingOverviewProps {
  summary: DashboardSummaryResponse;
}

/**
 * PRD §11.7: "current-period spending total, comparison to the prior
 * period." A spend increase is flagged (not necessarily bad, but worth
 * noticing) in warning tone; a decrease in success tone — mirroring the
 * refund/spend color convention already used in TransactionTable.
 *
 * PRODUCT-SEMANTICS FIX: when the selected month is still in progress,
 * `change_pct` compares the current month's spend through today against
 * the previous month's spend through the SAME day-of-month
 * (`comparable_day`) — never the previous month's full total, which reads
 * as a misleading steep decline early in a month. When the user has
 * selected a fully-completed historical month instead (`is_current_incomplete`
 * is false), both sides are simply full calendar months, and the copy below
 * says so plainly rather than naming an elapsed-day range that no longer
 * applies.
 *
 * CURRENT-MONTH-DEFAULT FIX: when `current_month_has_data` is false (no
 * transactions imported for `period.current` at all — either the user
 * explicitly selected an empty month, or nothing has been imported for it
 * yet), the "Total spend" and "Change" tiles show honest "not imported yet"
 * copy instead of a $0 total and a -100%/"behind" comparison. $0 and -100%
 * would both be mathematically consistent with an empty month, but they
 * read as "you spent nothing", which is a different and false claim from
 * "nothing has been imported yet".
 */
export function SpendingOverview({ summary }: SpendingOverviewProps) {
  const {
    period,
    is_current_incomplete,
    current_month_has_data,
    total_spend_current,
    total_spend_previous,
    comparable_day,
    change_pct,
  } = summary;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <Card variant="elevated">
        <CardHeader>
          <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {formatMonthLabel(period.current)}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {current_month_has_data ? (
            <>
              <p className="text-3xl font-bold tabular-nums">{formatCurrency(total_spend_current)}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {is_current_incomplete ? "Total spend so far" : "Total spend"}
              </p>
            </>
          ) : (
            <>
              <p className="text-lg font-semibold text-muted-foreground">Not imported yet</p>
              <p className="mt-1 text-xs text-muted-foreground">
                No transactions imported for {formatMonthLabel(period.current)} yet
              </p>
            </>
          )}
        </CardContent>
      </Card>

      <Card variant="elevated">
        <CardHeader>
          <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {formatMonthLabel(period.previous)}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-3xl font-bold tabular-nums">{formatCurrency(total_spend_previous)}</p>
          <p className="mt-1 text-xs text-muted-foreground">Total spend, full month</p>
        </CardContent>
      </Card>

      <Card variant="elevated">
        <CardHeader>
          <CardTitle className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Change
          </CardTitle>
        </CardHeader>
        <CardContent>
          {current_month_has_data ? (
            <>
              <ChangeIndicator changePct={change_pct} />
              <p className="mt-1 text-xs text-muted-foreground">
                {is_current_incomplete
                  ? `Vs. ${formatDayRangeLabel(period.previous, comparable_day)}`
                  : `Vs. ${formatMonthLabel(period.previous)}`}
              </p>
            </>
          ) : (
            <>
              <p className="text-3xl font-bold text-muted-foreground">—</p>
              <p className="mt-1 text-xs text-muted-foreground">No data to compare yet</p>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ChangeIndicator({ changePct }: { changePct: number | null }) {
  if (changePct === null) {
    return <p className="text-3xl font-bold text-muted-foreground">—</p>;
  }

  const isIncrease = changePct > 0;
  const isFlat = changePct === 0;
  const Icon = isFlat ? Minus : isIncrease ? ArrowUpRight : ArrowDownRight;

  return (
    <p
      className={cn(
        "flex items-center gap-1 text-3xl font-bold tabular-nums",
        isFlat ? "text-muted-foreground" : isIncrease ? "text-warning" : "text-success",
      )}
    >
      <Icon className="h-5 w-5" />
      {Math.abs(changePct).toFixed(1)}%
    </p>
  );
}
