import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import { colorForIndex } from "@/constants/chartColors";
import { formatCurrency } from "@/lib/utils";
import type { AllocationSlice } from "@/lib/portfolioMath";

interface PortfolioAllocationChartProps {
  slices: AllocationSlice[];
}

/** "What makes up my portfolio right now?" -- ticker-level allocation by
 * CURRENT MARKET VALUE only. No invented sectors/asset types: PlainCents
 * has no such metadata for a holding, so it shows exactly what it knows. */
export function PortfolioAllocationChart({ slices }: PortfolioAllocationChartProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Allocation</CardTitle>
        <CardDescription>Share of total market value, by ticker</CardDescription>
      </CardHeader>
      <CardContent>
        {slices.length === 0 ? (
          <EmptyState
            title="Nothing to allocate yet"
            description="Refresh prices to see how your holdings' market value breaks down."
            className="border-none py-10"
          />
        ) : (
          <ResponsiveContainer width="100%" height={Math.max(slices.length * 44, 160)}>
            <BarChart
              data={slices}
              layout="vertical"
              margin={{ top: 0, right: 24, bottom: 0, left: 0 }}
              barCategoryGap={10}
            >
              <CartesianGrid horizontal={false} stroke="hsl(var(--border))" />
              <XAxis
                type="number"
                tickFormatter={(v: number) => formatCurrency(v)}
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
              />
              <YAxis
                type="category"
                dataKey="ticker"
                width={72}
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
                formatter={(value, _name, entry) => [
                  `${formatCurrency(Number(value))} (${(entry.payload as AllocationSlice).pct.toFixed(1)}%)`,
                  "Market value",
                ]}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={28}>
                {slices.map((slice, i) => (
                  <Cell key={slice.ticker} fill={colorForIndex(i)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
