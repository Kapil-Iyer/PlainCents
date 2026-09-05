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
import { cn, formatCurrency, formatMonthLabel } from "@/lib/utils";

/**
 * "Am I ahead of or behind where I was this time last month?"
 *
 * Cumulative spend by day-of-month, current month against the previous one.
 * The current-month line stops at today rather than running flat to the end
 * of the month — a flat tail would read as "spent nothing since", when in
 * fact those days have not happened yet.
 */
export function SpendPaceCard() {
  const { data, isLoading, isError } = useSpendPace();
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

  return (
    <Card>
      <CardHeader>
        <CardTitle>Spending pace</CardTitle>
        <CardDescription>
          {formatMonthLabel(data.current_month, "short")} so far vs.{" "}
          {formatMonthLabel(data.previous_month, "short")}
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
                by day {data.day_of_month} last month
              </span>
            </motion.div>

            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={data.points} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
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
                <Line
                  type="monotone"
                  dataKey="previous_cumulative"
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

            <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
              <LegendSwatch className="bg-primary" label="This month" />
              <LegendSwatch className="bg-muted-foreground" label="Last month" dashed />
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
