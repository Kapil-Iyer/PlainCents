import { motion, useReducedMotion } from "framer-motion";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import { cn, formatCurrency } from "@/lib/utils";
import type { PnlSlice } from "@/lib/portfolioMath";

interface PortfolioPnlChartProps {
  slices: PnlSlice[];
  /** How many holdings exist in total, vs. how many are represented here
   * (only those with a known cost basis) -- so an incomplete chart never
   * silently looks complete. */
  totalHoldingsCount: number;
}

/**
 * "Which holdings are contributing gains or losses?" A zero-centered
 * diverging bar per holding with a KNOWN cost basis -- direction is read
 * from spatial position (a loss grows left of center, a gain grows right),
 * never from color alone. A holding with unknown cost basis is excluded
 * entirely, never rendered as a $0 (break-even) bar, which would fabricate
 * an answer PlainCents cannot honestly give.
 */
export function PortfolioPnlChart({ slices, totalHoldingsCount }: PortfolioPnlChartProps) {
  const reduceMotion = useReducedMotion();
  const excluded = totalHoldingsCount - slices.length;
  const largest = Math.max(...slices.map((s) => Math.abs(s.pnl)), 1);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Gain / loss by holding</CardTitle>
        <CardDescription>
          Unrealized P&amp;L, holdings with a known cost basis only
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {slices.length === 0 ? (
          <EmptyState
            title="Cost basis unavailable"
            description="Add an average cost to at least one holding to see which are gaining or losing."
            className="border-none py-10"
          />
        ) : (
          <>
            <div className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              <span>← Losses</span>
              <span>Gains →</span>
            </div>

            <ul className="flex flex-col gap-3">
              {slices.map((slice, i) => {
                const pct = (Math.abs(slice.pnl) / largest) * 100;
                const gain = slice.pnl > 0;
                return (
                  <li key={slice.ticker} className="flex flex-col gap-1">
                    <div className="flex items-baseline justify-between gap-3 text-sm">
                      <span className="truncate font-medium">{slice.ticker}</span>
                      <span
                        className={cn(
                          "shrink-0 tabular-nums",
                          slice.pnl === 0
                            ? "text-muted-foreground"
                            : gain
                              ? "text-success"
                              : "text-destructive",
                        )}
                      >
                        {gain ? "+" : slice.pnl < 0 ? "−" : ""}
                        {formatCurrency(Math.abs(slice.pnl))}
                      </span>
                    </div>
                    <div
                      className="relative h-2.5 overflow-hidden rounded-sm bg-muted"
                      role="img"
                      aria-label={`${slice.ticker} ${gain ? "gained" : "lost"} ${formatCurrency(Math.abs(slice.pnl))}`}
                    >
                      <span
                        aria-hidden
                        className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-border"
                      />
                      <motion.span
                        className={cn(
                          "absolute top-0 h-full rounded-sm",
                          gain ? "bg-success left-1/2" : "bg-destructive right-1/2",
                        )}
                        initial={reduceMotion ? { width: `${pct / 2}%` } : { width: 0 }}
                        animate={{ width: `${pct / 2}%` }}
                        transition={{ duration: 0.35, delay: reduceMotion ? 0 : i * 0.04 }}
                      />
                    </div>
                  </li>
                );
              })}
            </ul>

            {excluded > 0 && (
              <p className="text-xs text-muted-foreground">
                Cost basis unavailable for {excluded} other {excluded === 1 ? "holding" : "holdings"} —
                excluded above, not shown as break-even.
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
