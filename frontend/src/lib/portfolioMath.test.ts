import { describe, expect, it } from "vitest";

import {
  allocationByHolding,
  pnlByHolding,
  summarizePortfolio,
  weightedAverageCost,
} from "@/lib/portfolioMath";
import type { HoldingResponse } from "@/types/holding";

function holding(overrides: Partial<HoldingResponse> = {}): HoldingResponse {
  return {
    id: 1,
    ticker: "AAPL",
    shares: 10,
    avg_cost: 100,
    current_price: 150,
    current_value: 1500,
    pnl: 500,
    price_last_updated: "2026-01-01T00:00:00",
    price_is_demo_snapshot: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("weightedAverageCost", () => {
  it("computes the generic weighted average across purchase lots", () => {
    // 5@100 + 3@120 + 2@140 = 500+360+280=1140 / 10 = 114... wait recompute
    const result = weightedAverageCost([
      { shares: 5, price: 100 },
      { shares: 3, price: 120 },
      { shares: 2, price: 140 },
    ]);
    // (5*100 + 3*120 + 2*140) / 10 = (500+360+280)/10 = 1140/10 = 114
    expect(result).toBe(114);
  });

  it("returns null for no lots", () => {
    expect(weightedAverageCost([])).toBeNull();
  });

  it("ignores lots with non-positive or non-finite shares", () => {
    const result = weightedAverageCost([
      { shares: 10, price: 100 },
      { shares: 0, price: 999 },
      { shares: -5, price: 1 },
      { shares: NaN, price: 1 },
    ]);
    expect(result).toBe(100);
  });

  it("returns null (never 0) when every lot is invalid", () => {
    expect(weightedAverageCost([{ shares: 0, price: 100 }])).toBeNull();
  });

  it("rounds to 2 decimal places", () => {
    const result = weightedAverageCost([
      { shares: 3, price: 10 },
      { shares: 7, price: 20 },
    ]);
    // (30+140)/10 = 17
    expect(result).toBe(17);
  });
});

describe("summarizePortfolio", () => {
  it("sums market value across all priced holdings", () => {
    const summary = summarizePortfolio([
      holding({ ticker: "AAPL", current_value: 1500 }),
      holding({ ticker: "MSFT", current_value: 3000 }),
    ]);
    expect(summary.totalMarketValue).toBe(4500);
  });

  it("excludes unknown cost basis from knownCostBasis/knownPnl, never treats it as zero", () => {
    const summary = summarizePortfolio([
      holding({ ticker: "AAPL", shares: 10, avg_cost: 100, pnl: 500 }),
      holding({ ticker: "MSFT", shares: 5, avg_cost: null, current_value: 1500, pnl: null }),
    ]);
    expect(summary.holdingsCount).toBe(2);
    expect(summary.holdingsWithCostBasis).toBe(1);
    expect(summary.knownCostBasis).toBe(1000); // only AAPL's 10*100
    expect(summary.knownPnl).toBe(500); // only AAPL's pnl
  });

  it("handles an empty portfolio", () => {
    const summary = summarizePortfolio([]);
    expect(summary).toEqual({
      totalMarketValue: 0,
      knownCostBasis: 0,
      knownPnl: 0,
      holdingsCount: 0,
      holdingsWithCostBasis: 0,
    });
  });

  it("excludes an unpriced holding from totalMarketValue", () => {
    const summary = summarizePortfolio([holding({ current_value: null, pnl: null })]);
    expect(summary.totalMarketValue).toBe(0);
  });
});

describe("allocationByHolding", () => {
  it("computes percentage share of total market value, sorted descending", () => {
    const slices = allocationByHolding([
      holding({ ticker: "AAPL", current_value: 1000 }),
      holding({ ticker: "MSFT", current_value: 3000 }),
    ]);
    expect(slices[0]).toEqual({ ticker: "MSFT", value: 3000, pct: 75 });
    expect(slices[1]).toEqual({ ticker: "AAPL", value: 1000, pct: 25 });
  });

  it("excludes holdings with no known price entirely, never as a zero slice", () => {
    const slices = allocationByHolding([
      holding({ ticker: "AAPL", current_value: 1000 }),
      holding({ ticker: "TSLA", current_value: null, current_price: null, pnl: null }),
    ]);
    expect(slices).toHaveLength(1);
    expect(slices[0].ticker).toBe("AAPL");
  });

  it("returns an empty array for an empty portfolio", () => {
    expect(allocationByHolding([])).toEqual([]);
  });
});

describe("pnlByHolding", () => {
  it("returns signed P&L per holding with known cost basis, sorted by magnitude", () => {
    const slices = pnlByHolding([
      holding({ ticker: "AAPL", pnl: 100 }),
      holding({ ticker: "MSFT", pnl: -500 }),
    ]);
    expect(slices[0]).toEqual({ ticker: "MSFT", pnl: -500 });
    expect(slices[1]).toEqual({ ticker: "AAPL", pnl: 100 });
  });

  it("excludes holdings with unknown cost basis, never as a zero bar", () => {
    const slices = pnlByHolding([
      holding({ ticker: "AAPL", pnl: 100 }),
      holding({ ticker: "MSFT", avg_cost: null, pnl: null }),
    ]);
    expect(slices).toHaveLength(1);
    expect(slices[0].ticker).toBe("AAPL");
  });

  it("preserves negative P&L (losses) distinctly from positive (gains)", () => {
    const slices = pnlByHolding([holding({ pnl: -250 })]);
    expect(slices[0].pnl).toBe(-250);
  });
});
