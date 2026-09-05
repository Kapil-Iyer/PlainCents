import { History } from "lucide-react";
import { useReducedMotion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import { useForecastAccuracy } from "@/hooks/useAnalytics";
import { formatCurrency, formatMonthLabel } from "@/lib/utils";

import { ChartCardSkeleton, TOOLTIP_STYLE } from "@/components/analytics/primitives";

/**
 * "Were my past forecasts any good?"
 *
 * This card is deliberately allowed to have nothing to show.
 *
 * A forecast-vs-actual chart is only honest if the forecast genuinely existed
 * before the month it predicted. It would have been easy to make this look
 * populated immediately by running today's model over last year's data and
 * charting the result — but that is not a record of what PlainCents
 * predicted, it is a record of what it would predict now, presented as
 * history. So the backend only counts a prediction whose run was generated
 * strictly before the predicted month began, for a month that has since
 * completed, and until such a snapshot exists this card says so plainly.
 *
 * On a fresh install that is the normal state. It resolves on its own: the
 * user generates a forecast, a month passes, and real evidence appears.
 */
export function ForecastAccuracyCard() {
  const { data, isLoading, isError } = useForecastAccuracy();
  const reduceMotion = useReducedMotion();

  if (isLoading) return <ChartCardSkeleton title="Forecast accuracy" />;
  if (isError || !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Forecast accuracy</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            title="Couldn't load forecast accuracy"
            description="Something went wrong talking to the server."
          />
        </CardContent>
      </Card>
    );
  }

  if (!data.available) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Forecast accuracy</CardTitle>
          <CardDescription>How past forecasts compared to what you actually spent</CardDescription>
        </CardHeader>
        <CardContent>
          <EmptyState
            icon={History}
            title="No forecast history yet"
            description="This fills in once you've generated a forecast and the month it covered has finished. PlainCents only compares forecasts it actually made at the time — it won't re-run today's model on old months and call that a prediction."
          />
        </CardContent>
      </Card>
    );
  }

  // Aggregate to the month level for the chart; per-category detail lives in
  // the table below it, where it can be read as numbers rather than 8 bars
  // per month.
  const byMonth = new Map<string, { month: string; predicted: number; actual: number }>();
  for (const item of data.items) {
    const row = byMonth.get(item.forecast_month) ?? {
      month: item.forecast_month,
      predicted: 0,
      actual: 0,
    };
    row.predicted += item.predicted;
    row.actual += item.actual;
    byMonth.set(item.forecast_month, row);
  }
  const rows = [...byMonth.values()]
    .sort((a, b) => a.month.localeCompare(b.month))
    .map((r) => ({
      ...r,
      predicted: Math.round(r.predicted * 100) / 100,
      actual: Math.round(r.actual * 100) / 100,
      label: formatMonthLabel(r.month, "short"),
    }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Forecast accuracy</CardTitle>
        <CardDescription>
          {data.months_covered.length} completed{" "}
          {data.months_covered.length === 1 ? "month" : "months"} with a forecast made beforehand
          {data.wape !== null && ` — off by ${(data.wape * 100).toFixed(1)}% overall`}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} stroke="hsl(var(--border))" />
            <XAxis dataKey="label" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} />
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
                name === "predicted" ? "Forecast" : "Actual",
              ]}
            />
            <Legend wrapperStyle={{ fontSize: "0.75rem" }} />
            <Bar
              dataKey="predicted"
              name="Forecast"
              fill="hsl(var(--muted-foreground))"
              radius={[3, 3, 0, 0]}
              isAnimationActive={!reduceMotion}
            />
            <Bar
              dataKey="actual"
              name="Actual"
              fill="hsl(var(--primary))"
              radius={[3, 3, 0, 0]}
              isAnimationActive={!reduceMotion}
            />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
