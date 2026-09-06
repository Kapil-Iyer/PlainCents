import { motion, useReducedMotion } from "framer-motion";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState } from "@/components/shared/EmptyState";
import { useCategoryMovers } from "@/hooks/useAnalytics";
import { colorForCategory } from "@/constants/chartColors";
import { cn, formatCurrency, formatDayRangeLabel, formatMonthLabel } from "@/lib/utils";

import { ChartCardSkeleton } from "@/components/analytics/primitives";

const MAX_ROWS = 6;

/**
 * "Why did I spend more (or less) than last month?"
 *
 * A zero-centered diverging bar per category, sized by its share of the
 * LARGEST single movement so the bars stay comparable to each other.
 * Direction is read from spatial position — a decrease grows LEFT of the
 * visible zero tick, an increase grows RIGHT of it — never from color
 * alone, with an explicit "Less spending / More spending" axis label pair
 * above the list stating what each side means. The per-category changes
 * sum exactly to the headline total change, which is what makes this an
 * explanation of the month rather than a second, unrelated chart.
 *
 * Rendered as bars in the DOM rather than a charting library: each row is
 * one number and one label, the whole point is reading them as a ranked
 * list, and a real list is more accessible than an SVG chart here.
 *
 * `month` ("YYYY-MM") is the shared analysis-month clock (see
 * Dashboard.tsx) -- omitted, this defaults to the current calendar month.
 */
export function CategoryMoversCard({ month }: { month?: string }) {
  const { data, isLoading, isError } = useCategoryMovers(month);
  const reduceMotion = useReducedMotion();

  const title = data && !data.is_current_incomplete
    ? `What changed in ${formatMonthLabel(data.current_month, "short")}`
    : "What changed so far";

  if (isLoading) return <ChartCardSkeleton title="What changed so far" />;
  if (isError || !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>What changed so far</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState
            title="Couldn't load this month's changes"
            description="Something went wrong talking to the server."
          />
        </CardContent>
      </Card>
    );
  }

  const movers = data.movers.filter((m) => m.change !== 0).slice(0, MAX_ROWS);
  const largest = Math.max(...movers.map((m) => Math.abs(m.change)), 1);
  const up = data.total_change > 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>
          {data.is_current_incomplete
            ? `${formatDayRangeLabel(data.current_month, data.comparable_day)} vs. ${formatDayRangeLabel(data.previous_month, data.comparable_day)} — each category's contribution`
            : `${formatMonthLabel(data.current_month)} vs. ${formatMonthLabel(data.previous_month)}, full months — each category's contribution`}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {movers.length === 0 ? (
          <EmptyState
            title="Nothing to compare yet"
            description="Once you have spending in two consecutive months, this breaks the change down by category."
          />
        ) : (
          <>
            <div className="flex items-baseline gap-2">
              <span
                className={cn(
                  "text-2xl font-semibold tabular-nums",
                  up ? "text-warning" : "text-success",
                )}
              >
                {up ? "+" : "−"}
                {formatCurrency(Math.abs(data.total_change))}
              </span>
              <span className="text-sm text-muted-foreground">
                {up ? "more" : "less"} than{" "}
                {data.is_current_incomplete ? "the same point last month" : "last month"}
              </span>
            </div>

            {/* Explicit axis labels: direction is stated in words, not left
             * to be inferred from color or bar position alone. */}
            <div className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              <span>← Less spending</span>
              <span>More spending →</span>
            </div>

            <ul className="flex flex-col gap-3">
              {movers.map((mover, i) => {
                const pct = (Math.abs(mover.change) / largest) * 100;
                const rose = mover.change > 0;
                return (
                  <li key={mover.category} className="flex flex-col gap-1">
                    <div className="flex items-baseline justify-between gap-3 text-sm">
                      <span className="flex min-w-0 items-center gap-2">
                        <span
                          aria-hidden
                          className="h-2 w-2 shrink-0 rounded-full"
                          style={{ background: colorForCategory(mover.category) }}
                        />
                        <span className="truncate">{mover.category}</span>
                      </span>
                      <span
                        className={cn(
                          "shrink-0 tabular-nums",
                          rose ? "text-warning" : "text-success",
                        )}
                      >
                        {rose ? "+" : "−"}
                        {formatCurrency(Math.abs(mover.change))}
                        {mover.change_pct !== null && (
                          <span className="ml-1.5 text-xs text-muted-foreground">
                            ({rose ? "+" : ""}
                            {mover.change_pct}%)
                          </span>
                        )}
                      </span>
                    </div>
                    {/* Zero-centered diverging bar: increases grow right of
                     * center, decreases grow left -- direction is readable
                     * from POSITION, not color alone. The 1px tick at
                     * center is the visible zero reference every bar is
                     * measured from. */}
                    <div
                      className="relative h-2.5 overflow-hidden rounded-sm bg-muted"
                      role="img"
                      aria-label={`${mover.category} ${rose ? "increased" : "decreased"} by ${formatCurrency(Math.abs(mover.change))}`}
                    >
                      <span
                        aria-hidden
                        className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-border"
                      />
                      <motion.span
                        className={cn(
                          "absolute top-0 h-full rounded-sm",
                          rose ? "bg-warning left-1/2" : "bg-success right-1/2",
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
          </>
        )}
      </CardContent>
    </Card>
  );
}
