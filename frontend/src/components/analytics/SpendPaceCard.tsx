import { motion, useReducedMotion } from "framer-motion";
import { TrendingDown, TrendingUp } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import { useSpendPace } from "@/hooks/useAnalytics";
import {
  ChartCardSkeleton,
  LegendSwatch,
  TOOLTIP_STYLE,
} from "@/components/analytics/primitives";
import { cn, formatCurrency, formatDayRangeLabel, formatMonthLabel } from "@/lib/utils";

/**
 * "Am I ahead of or behind where I was this time last month?"
 *
 * Cumulative spend by day-of-month, selected month against the previous
 * one -- driven by the same shared analysis-month clock as the Dashboard's
 * Change KPI and Category Movers (see Dashboard.tsx). When the selected
 * month is still in progress, its line stops at today rather than running
 * flat to the end of the month — a flat tail would read as "spent nothing
 * since", when in fact those days have not happened yet. When a
 * fully-completed historical month is selected instead, both lines run to
 * their own full real length.
 */
export function SpendPaceCard({ month }: { month?: string }) {
  const { data, isLoading, isError } = useSpendPace(month);
  const reduceMotion = useReducedMotion();

  if (isLoading) return <ChartCardSkeleton title="Spending pace" />;
  if (isError || !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Spending pace</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            title="Couldn't load spending pace"
            description="Something went wrong talking to the server."
          />
        </CardContent>
      </Card>
    );
  }

  const hasData = data.current_to_date > 0 || data.previous_same_point > 0;
  const ahead = data.difference > 0;
  const Icon = ahead ? TrendingUp : TrendingDown;

  // The backend's previous_cumulative already runs to the previous month's
  // own full length (context: "was last month's pace unusual overall?"),
  // while the fair day-for-day COMPARISON only actually covers 1..comparable_day
  // (see date_windows.elapsed_window / spend_pace's own docstring) -- the
  // metrics above (`difference`, `previous_same_point`) already respect that
  // cutoff. Split the single previous-month series into two chart-only
  // series so the two are visually distinguishable rather than reading as
  // one continuous "last month" comparison: `previous_comparable` (solid
  // dashed, same as before) IS the fair comparison window; `previous_context`
  // (lighter, dotted) is everything after it, shown only so a longer prior
  // month isn't hidden, never implied to be part of the pace comparison.
  // Both include the point AT comparable_day so the two segments connect
  // with no visual gap.
  const chartPoints = data.points.map((p) => ({
    ...p,
    previous_comparable: p.day <= data.comparable_day ? p.previous_cumulative : null,
    previous_context: p.day >= data.comparable_day ? p.previous_cumulative : null,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {data.is_current_incomplete ? "Spending pace" : `Spending pace — ${formatMonthLabel(data.current_month, "short")}`}
        </CardTitle>
        <CardDescription>
          {data.is_current_incomplete
            ? `${formatDayRangeLabel(data.current_month, data.day_of_month)} vs. ${formatDayRangeLabel(data.previous_month, data.comparable_day)}`
            : `${formatMonthLabel(data.current_month)} vs. ${formatMonthLabel(data.previous_month)}, full months`}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {!hasData ? (
          <EmptyState
            title="Not enough history yet"
            description="Once you have spending in this month and the last, this compares them day by day."
          />
        ) : (
          <>
            <motion.div
              initial={reduceMotion ? false : { opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              className="flex flex-wrap items-baseline gap-x-3 gap-y-1"
            >
              <span className="text-2xl font-semibold tabular-nums">
                {formatCurrency(data.current_to_date)}
              </span>
              <span
                className={cn(
                  "inline-flex items-center gap-1 text-sm font-medium",
                  ahead ? "text-warning" : "text-success",
                )}
              >
                <Icon className="h-4 w-4" aria-hidden />
                {formatCurrency(Math.abs(data.difference))} {ahead ? "ahead of" : "behind"}
              </span>
              <span className="text-sm text-muted-foreground">
                {data.is_current_incomplete
                  ? `by day ${data.comparable_day} last month`
                  : "for the same full month last time"}
              </span>
            </motion.div>

            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartPoints} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid vertical={false} stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="day"
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                  tickLine={false}
                />
                <YAxis
                  tickFormatter={(v: number) => formatCurrency(v)}
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                  width={72}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  formatter={(value, name) => [
                    formatCurrency(Number(value)),
                    name === "current_cumulative" ? "This month" : "Last month",
                  ]}
                  labelFormatter={(day) => `Day ${day}`}
                />
                {/* Rest of last month, shown lightly for context only — not
                 * part of the pace comparison (see comment above). Drawn
                 * first so the comparable segment's dot/line stays on top
                 * at the point where they meet. */}
                <Line
                  type="monotone"
                  dataKey="previous_context"
                  stroke="hsl(var(--muted-foreground))"
                  strokeWidth={1.5}
                  strokeDasharray="1 5"
                  strokeOpacity={0.5}
                  dot={false}
                  isAnimationActive={!reduceMotion}
                  animationDuration={400}
                />
                <Line
                  type="monotone"
                  dataKey="previous_comparable"
                  stroke="hsl(var(--muted-foreground))"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={false}
                  isAnimationActive={!reduceMotion}
                  animationDuration={400}
                />
                <Line
                  type="monotone"
                  dataKey="current_cumulative"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2.5}
                  dot={false}
                  // A null past today leaves a genuine gap in the line rather
                  // than connecting through days that have not happened.
                  connectNulls={false}
                  isAnimationActive={!reduceMotion}
                  animationDuration={400}
                />
              </LineChart>
            </ResponsiveContainer>

            <div className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
              <LegendSwatch className="bg-primary" label="This month" />
              <LegendSwatch className="bg-muted-foreground" label="Last month (comparable days)" dashed />
              {data.comparable_day < (chartPoints.at(-1)?.day ?? 0) && (
                <span className="opacity-70">···· rest of last month, for context only</span>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
