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
import { colorForCategory } from "@/constants/chartColors";
import { formatCurrency } from "@/lib/utils";
import type { CategoryBreakdownItem } from "@/types/dashboard";

interface CategoryBreakdownProps {
  items: CategoryBreakdownItem[];
  /** The selected analysis month's display label (e.g. "September 2026") --
   * this breakdown follows the same shared analysis-month clock as the
   * Change KPI, Spending Pace, and Category Movers (see Dashboard.tsx), so
   * its copy must name the actual month shown, not always assume "this
   * month". */
  monthLabel: string;
}

/** PRD §11.7: "a category breakdown." Horizontal bars read category names
 * (some are multi-word) more comfortably than a pie chart's small slices. */
export function CategoryBreakdown({ items, monthLabel }: CategoryBreakdownProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Spending by category</CardTitle>
        <CardDescription>{monthLabel}, by effective category</CardDescription>
      </CardHeader>
      <CardContent>
        {items.length === 0 ? (
          <EmptyState
            title="No spending yet"
            description={`Categories will appear here once you have transactions in ${monthLabel}.`}
            className="border-none py-10"
          />
        ) : (
          <ResponsiveContainer width="100%" height={Math.max(items.length * 44, 160)}>
            <BarChart
              data={items}
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
                formatter={(value, _name, entry) => [
                  `${formatCurrency(Number(value))} (${(entry.payload as CategoryBreakdownItem).pct_of_total.toFixed(1)}%)`,
                  "Spend",
                ]}
              />
              <Bar dataKey="total_spend" radius={[0, 4, 4, 0]} maxBarSize={28}>
                {items.map((item) => (
                  <Cell key={item.category} fill={colorForCategory(item.category)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
