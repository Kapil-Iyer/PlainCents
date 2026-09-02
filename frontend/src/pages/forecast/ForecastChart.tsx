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
import { formatCurrency } from "@/lib/utils";
import type { ForecastPrediction } from "@/types/forecast";

interface ForecastChartProps {
  predictions: ForecastPrediction[];
}

interface ChartRow {
  category: string;
  offset_1: number | null;
  offset_2: number | null;
  offset_3: number | null;
}

/** PRD Section 11.8: a chart view of the +1/+2/+3 forecast, alongside
 * CategoryForecastList's exact figures. Only available categories are
 * charted — unavailable ones have no number to plot (TRD Section 12.5). */
export function ForecastChart({ predictions }: ForecastChartProps) {
  const byCategory = new Map<string, ChartRow>();
  for (const p of predictions) {
    if (!p.is_available) continue;
    const row = byCategory.get(p.category) ?? { category: p.category, offset_1: null, offset_2: null, offset_3: null };
    if (p.month_offset === 1) row.offset_1 = p.predicted_amount;
    if (p.month_offset === 2) row.offset_2 = p.predicted_amount;
    if (p.month_offset === 3) row.offset_3 = p.predicted_amount;
    byCategory.set(p.category, row);
  }
  const data = Array.from(byCategory.values()).sort((a, b) => a.category.localeCompare(b.category));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Forecast by category</CardTitle>
        <CardDescription>Predicted spend, next 3 months</CardDescription>
      </CardHeader>
      <CardContent>
        {data.length === 0 ? (
          <EmptyState
            title="No categories available to chart"
            description="Every category needs more history before a forecast can be charted for it."
            className="border-none py-10"
          />
        ) : (
          <ResponsiveContainer width="100%" height={Math.max(data.length * 56, 200)}>
            <BarChart data={data} layout="vertical" margin={{ top: 0, right: 24, bottom: 0, left: 0 }}>
              <CartesianGrid horizontal={false} stroke="hsl(var(--border))" />
              <XAxis
                type="number"
                tickFormatter={(v: number) => formatCurrency(v)}
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
              />
              <YAxis
                type="category"
                dataKey="category"
                width={120}
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "hsl(var(--accent))" }}
                contentStyle={{
                  background: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "0.5rem",
                  fontSize: "0.75rem",
                }}
                formatter={(value, name) => [value == null ? "—" : formatCurrency(Number(value)), name]}
              />
              <Legend wrapperStyle={{ fontSize: "0.75rem" }} />
              <Bar dataKey="offset_1" name="+1 month" fill="hsl(var(--primary))" fillOpacity={1} radius={[0, 4, 4, 0]} maxBarSize={14} />
              <Bar dataKey="offset_2" name="+2 months" fill="hsl(var(--primary))" fillOpacity={0.6} radius={[0, 4, 4, 0]} maxBarSize={14} />
              <Bar dataKey="offset_3" name="+3 months" fill="hsl(var(--primary))" fillOpacity={0.35} radius={[0, 4, 4, 0]} maxBarSize={14} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
