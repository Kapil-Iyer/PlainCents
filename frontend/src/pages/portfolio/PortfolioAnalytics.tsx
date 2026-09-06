import { EmptyState } from "@/components/shared/EmptyState";
import { allocationByHolding, pnlByHolding, summarizePortfolio } from "@/lib/portfolioMath";
import type { HoldingResponse } from "@/types/holding";

import { PortfolioAllocationChart } from "@/pages/portfolio/PortfolioAllocationChart";
import { PortfolioPnlChart } from "@/pages/portfolio/PortfolioPnlChart";
import { PortfolioSummaryMetrics } from "@/pages/portfolio/PortfolioSummaryMetrics";

interface PortfolioAnalyticsProps {
  holdings: HoldingResponse[];
}

/**
 * Portfolio Analytics -- a separate section below the holdings table.
 * Deliberately NOT part of the spending Dashboard: spending and portfolio
 * are separate domains, and nothing computed here ever feeds transaction
 * analytics/forecasts (see PortfolioHowItWorks' "Separate from spending").
 *
 * All three pieces below (summary metrics, allocation, P&L) derive from
 * the SAME @/lib/portfolioMath functions applied to the SAME `holdings`
 * prop -- one source of calculation semantics, computed frontend-side
 * from the existing GET /api/holdings response rather than a redundant
 * backend summary endpoint (the calculations are simple and holdings
 * counts are small; see the completion pass's own audit notes).
 *
 * No portfolio-value-over-time chart here, and deliberately so: PlainCents
 * has no truthful historical portfolio snapshots (a price refresh only
 * ever knows "now"), so a performance-over-time chart would have to
 * fabricate history it doesn't have.
 */
export function PortfolioAnalytics({ holdings }: PortfolioAnalyticsProps) {
  if (holdings.length === 0) {
    return null;
  }

  const summary = summarizePortfolio(holdings);
  const allocation = allocationByHolding(holdings);
  const pnl = pnlByHolding(holdings);

  return (
    <div data-tour="portfolio-analytics" className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">Portfolio analytics</h2>
        <p className="text-sm text-muted-foreground">
          Computed from your current holdings and their latest known prices.
        </p>
      </div>

      <PortfolioSummaryMetrics summary={summary} />

      {summary.totalMarketValue === 0 ? (
        <EmptyState
          title="Nothing priced yet"
          description="Click Refresh Prices above to see allocation and gain/loss."
          className="border-none py-6"
        />
      ) : (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
          <PortfolioAllocationChart slices={allocation} />
          <PortfolioPnlChart slices={pnl} totalHoldingsCount={holdings.length} />
        </div>
      )}
    </div>
  );
}
