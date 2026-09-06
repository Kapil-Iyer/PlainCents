import type { HoldingResponse } from "@/types/holding";

/**
 * Shared portfolio calculation semantics (Portfolio + Power BI completion
 * pass). One source of truth for every place that needs these numbers --
 * PortfolioAnalytics' summary metrics, its allocation chart, and its P&L
 * chart all call these same functions, so they can never silently disagree
 * with each other or with the holdings table above them.
 *
 * LOCKED SEMANTICS (do not change without also updating the backend, which
 * computes current_value/pnl identically in PortfolioService._to_response):
 *   market_value = shares * current_price
 *   cost_basis   = shares * avg_cost         (only when avg_cost is known)
 *   pnl          = market_value - cost_basis (only when both are known)
 *
 * Unknown cost basis is never coerced to 0 anywhere in this file -- a
 * holding with no avg_cost is EXCLUDED from cost-basis/P&L aggregates, not
 * counted as a $0 cost basis (which would fabricate a huge fake gain).
 */

export interface PurchaseLot {
  shares: number;
  price: number;
}

/** Weighted average price per share across purchase lots:
 * sum(shares_i * price_i) / sum(shares_i). Returns null when there is
 * nothing valid to average (no lots, or every lot has non-positive
 * shares) -- never 0, which would read as a real, deliberate cost basis
 * of zero. Lots with non-finite or non-positive shares are ignored rather
 * than rejecting the whole calculation, so one bad row doesn't block the
 * others. */
export function weightedAverageCost(lots: PurchaseLot[]): number | null {
  let totalShares = 0;
  let totalCost = 0;
  for (const lot of lots) {
    if (!Number.isFinite(lot.shares) || !Number.isFinite(lot.price) || lot.shares <= 0) continue;
    totalShares += lot.shares;
    totalCost += lot.shares * lot.price;
  }
  if (totalShares <= 0) return null;
  return Math.round((totalCost / totalShares) * 100) / 100;
}

export interface PortfolioSummary {
  /** Sum of market value across every holding with a known current price. */
  totalMarketValue: number;
  /** Sum of shares * avg_cost, only across holdings with a known avg_cost. */
  knownCostBasis: number;
  /** Sum of (market_value - cost_basis), only across holdings with BOTH a
   * known price and a known avg_cost. */
  knownPnl: number;
  holdingsCount: number;
  /** How many holdings have a recorded avg_cost -- "3 of 4 holdings have
   * cost basis", never silently implied to be all of them. */
  holdingsWithCostBasis: number;
}

export function summarizePortfolio(holdings: HoldingResponse[]): PortfolioSummary {
  let totalMarketValue = 0;
  let knownCostBasis = 0;
  let knownPnl = 0;
  let holdingsWithCostBasis = 0;

  for (const h of holdings) {
    if (h.current_value !== null) {
      totalMarketValue += h.current_value;
    }
    if (h.avg_cost !== null) {
      holdingsWithCostBasis += 1;
      knownCostBasis += h.shares * h.avg_cost;
      if (h.pnl !== null) {
        knownPnl += h.pnl;
      }
    }
  }

  return {
    totalMarketValue,
    knownCostBasis,
    knownPnl,
    holdingsCount: holdings.length,
    holdingsWithCostBasis,
  };
}

export interface AllocationSlice {
  ticker: string;
  value: number;
  pct: number;
}

/** Allocation by CURRENT MARKET VALUE only -- ticker-level, never an
 * invented sector/asset-type grouping. Holdings with no known price are
 * excluded (there is nothing honest to allocate), not shown as a $0
 * slice. */
export function allocationByHolding(holdings: HoldingResponse[]): AllocationSlice[] {
  const priced = holdings.filter((h) => h.current_value !== null) as (HoldingResponse & {
    current_value: number;
  })[];
  const total = priced.reduce((sum, h) => sum + h.current_value, 0);
  return priced
    .map((h) => ({
      ticker: h.ticker,
      value: h.current_value,
      pct: total > 0 ? (h.current_value / total) * 100 : 0,
    }))
    .sort((a, b) => b.value - a.value);
}

export interface PnlSlice {
  ticker: string;
  pnl: number;
}

/** Signed P&L per holding, known-cost-basis holdings only -- an unknown
 * cost basis is excluded, never rendered as a $0 bar (which would read as
 * "broke even", a fabricated claim). */
export function pnlByHolding(holdings: HoldingResponse[]): PnlSlice[] {
  return holdings
    .filter((h): h is HoldingResponse & { pnl: number } => h.pnl !== null)
    .map((h) => ({ ticker: h.ticker, pnl: h.pnl }))
    .sort((a, b) => Math.abs(b.pnl) - Math.abs(a.pnl));
}
