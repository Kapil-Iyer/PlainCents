import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatCurrency, formatMonthLabel } from "@/lib/utils";
import type { SpendingTrendPoint } from "@/types/dashboard";

interface SpendingTrendProps {
  points: SpendingTrendPoint[];
}

/** PRD §11.7: "a spending trend over time." A trailing multi-month area
 * chart, zero-filled for months with no transactions (real information, not
 * decoration — see DashboardService's docstring for why zero-fill is honest). */
export function SpendingTrend({ points }: SpendingTrendProps) {
  const data = points.map((p) => ({ ...p, label: formatMonthLabel(p.month, "short") }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Spending trend</CardTitle>
        <CardDescription>Total spend, last {points.length} months</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity={0.35} />
                <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} stroke="hsl(var(--border))" />
            <XAxis
              dataKey="label"
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
              contentStyle={{
                background: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "0.5rem",
                fontSize: "0.75rem",
              }}
              formatter={(value) => [formatCurrency(Number(value)), "Spend"]}
              labelFormatter={(_label, payload) =>
                payload[0] ? formatMonthLabel(payload[0].payload.month) : ""
              }
            />
            <Area
              type="monotone"
              dataKey="total_spend"
              stroke="hsl(var(--primary))"
              strokeWidth={2}
              fill="url(#trendFill)"
              isAnimationActive
              animationDuration={400}
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
