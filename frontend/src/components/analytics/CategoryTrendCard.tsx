import { useState } from "react";
import { useReducedMotion } from "framer-motion";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import { colorForCategory } from "@/constants/chartColors";
import { useCategoryTrend } from "@/hooks/useAnalytics";
import { formatCurrency, formatMonthLabel } from "@/lib/utils";

import {
  ChartCardSkeleton,
  SegmentedControl,
  TOOLTIP_STYLE,
} from "@/components/analytics/primitives";

const WINDOWS = [6, 12, 24] as const;
type Window = (typeof WINDOWS)[number];

type View = "stacked" | "lines";

/**
 * "Which categories are growing or shrinking, and since when?"
 *
 * Two views over the same data, because they answer genuinely different
 * questions and neither subsumes the other:
 *   Stacked  — how total spend is composed, and how that mix shifts.
 *   Lines    — how one category moves independently of the others, which a
 *              stacked chart actively hides (a band's thickness is hard to
 *              read when the bands below it are moving).
 *
 * Grouped by effective category, so correcting a transaction moves these
 * lines immediately.
 */
export function CategoryTrendCard() {
  const [months, setMonths] = useState<Window>(12);
  const [view, setView] = useState<View>("stacked");
  const { data, isLoading, isError } = useCategoryTrend(months);
  const reduceMotion = useReducedMotion();

  if (isLoading) return <ChartCardSkeleton title="Category trend" />;
  if (isError || !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Category trend</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            title="Couldn't load the category trend"
            description="Something went wrong talking to the server."
          />
        </CardContent>
      </Card>
    );
  }

  const rows = data.points.map((p) => ({
    month: p.month,
    label: formatMonthLabel(p.month, "short"),
    ...p.by_category,
  }));

  return (
    <Card>
      <CardHeader className="gap-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle>Category trend</CardTitle>
            <CardDescription>
              Monthly spend by category. Your corrections are included.
            </CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <SegmentedControl
              label="Chart type"
              value={view}
              onChange={(v) => setView(v as View)}
              options={[
                { value: "stacked", label: "Stacked" },
                { value: "lines", label: "Lines" },
              ]}
            />
            <SegmentedControl
              label="Time range"
              value={String(months)}
              onChange={(v) => setMonths(Number(v) as Window)}
              options={WINDOWS.map((w) => ({ value: String(w), label: `${w}m` }))}
            />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {data.categories.length === 0 ? (
          <EmptyState
            title="No spending in this period"
            description="Import a statement or widen the time range to see how your categories move."
          />
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            {view === "stacked" ? (
              <AreaChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
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
                  formatter={(value, name) => [formatCurrency(Number(value)), String(name)]}
                  labelFormatter={(_l, payload) =>
                    payload?.[0] ? formatMonthLabel(payload[0].payload.month) : ""
                  }
                />
                <Legend wrapperStyle={{ fontSize: "0.75rem" }} />
                {data.categories.map((category) => (
                  <Area
                    key={category}
                    type="monotone"
                    dataKey={category}
                    stackId="spend"
                    stroke={colorForCategory(category)}
                    fill={colorForCategory(category)}
                    fillOpacity={0.35}
                    strokeWidth={1.5}
                    isAnimationActive={!reduceMotion}
                    animationDuration={400}
                  />
                ))}
              </AreaChart>
            ) : (
              <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
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
                  formatter={(value, name) => [formatCurrency(Number(value)), String(name)]}
                  labelFormatter={(_l, payload) =>
                    payload?.[0] ? formatMonthLabel(payload[0].payload.month) : ""
                  }
                />
                <Legend wrapperStyle={{ fontSize: "0.75rem" }} />
                {data.categories.map((category) => (
                  <Line
                    key={category}
                    type="monotone"
                    dataKey={category}
                    stroke={colorForCategory(category)}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={!reduceMotion}
                    animationDuration={400}
                  />
                ))}
              </LineChart>
            )}
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
