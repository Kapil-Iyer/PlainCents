import { Card, CardContent } from "@/components/ui/card";
import { formatCurrency } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { PortfolioSummary } from "@/lib/portfolioMath";

interface PortfolioSummaryMetricsProps {
  summary: PortfolioSummary;
}

/**
 * Summary metrics for Portfolio Analytics. `knownCostBasis`/`knownPnl` are
 * computed ONLY from holdings with a recorded average cost -- coverage is
 * always shown alongside them so the figures are never mistaken for the
 * whole portfolio's totals (unknown cost basis is excluded, never
 * counted as $0).
 */
export function PortfolioSummaryMetrics({ summary }: PortfolioSummaryMetricsProps) {
  const { totalMarketValue, knownCostBasis, knownPnl, holdingsCount, holdingsWithCostBasis } = summary;
  const fullCoverage = holdingsCount > 0 && holdingsWithCostBasis === holdingsCount;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <Card variant="elevated">
        <CardContent className="pt-6">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Total market value
          </p>
          <p className="mt-1 text-2xl font-bold tabular-nums">{formatCurrency(totalMarketValue)}</p>
          <p className="mt-1 text-xs text-muted-foreground">Across all priced holdings</p>
        </CardContent>
      </Card>

      <Card variant="elevated">
        <CardContent className="pt-6">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Known cost basis
          </p>
          <p className="mt-1 text-2xl font-bold tabular-nums">{formatCurrency(knownCostBasis)}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {fullCoverage
              ? "All holdings have a recorded cost"
              : `Only holdings with a recorded average cost`}
          </p>
        </CardContent>
      </Card>

      <Card variant="elevated">
        <CardContent className="pt-6">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Unrealized P&amp;L
          </p>
          <p
            className={cn(
              "mt-1 text-2xl font-bold tabular-nums",
              holdingsWithCostBasis === 0
                ? "text-muted-foreground"
                : knownPnl >= 0
                  ? "text-success"
                  : "text-destructive",
            )}
          >
            {holdingsWithCostBasis === 0 ? "—" : formatCurrency(knownPnl)}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {fullCoverage
              ? "Covers every holding"
              : "Covers only holdings with a known cost basis"}
          </p>
        </CardContent>
      </Card>

      <Card variant="elevated">
        <CardContent className="pt-6">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Cost-basis coverage
          </p>
          <p className="mt-1 text-2xl font-bold tabular-nums">
            {holdingsWithCostBasis} of {holdingsCount}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">Holdings with a recorded average cost</p>
        </CardContent>
      </Card>
    </div>
  );
}
