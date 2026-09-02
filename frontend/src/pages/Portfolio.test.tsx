import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";
import type { HoldingResponse } from "@/types/holding";

import { PortfolioPage } from "@/pages/Portfolio";

vi.mock("@/api/holdings", () => ({
  listHoldings: vi.fn(),
  createHolding: vi.fn(),
  updateHolding: vi.fn(),
  deleteHolding: vi.fn(),
  refreshPrices: vi.fn(),
}));

const neverRefreshed: HoldingResponse = {
  id: 1,
  ticker: "AAPL",
  shares: 10,
  avg_cost: 100,
  current_price: null,
  current_value: null,
  pnl: null,
  price_last_updated: null,
  created_at: "2026-01-15T00:00:00Z",
  updated_at: "2026-01-15T00:00:00Z",
};

const cached: HoldingResponse = {
  ...neverRefreshed,
  id: 2,
  ticker: "MSFT",
  current_price: 300,
  current_value: 3000,
  pnl: 2000,
  price_last_updated: "2026-01-15T00:00:00",
};

describe("PortfolioPage", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("renders an empty state when there are no holdings", async () => {
    const { listHoldings } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([]);

    renderWithProviders(<PortfolioPage />);

    expect(await screen.findByText("No holdings yet")).toBeInTheDocument();
  });

  it("never calls refreshPrices just from loading the page", async () => {
    const { listHoldings, refreshPrices } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([neverRefreshed]);

    renderWithProviders(<PortfolioPage />);

    await screen.findByText("AAPL");
    expect(refreshPrices).not.toHaveBeenCalled();
  });

  it("distinguishes never-refreshed from cached prices", async () => {
    const { listHoldings } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([neverRefreshed, cached]);

    renderWithProviders(<PortfolioPage />);

    await screen.findByText("AAPL");
    expect(screen.getByText("Not yet refreshed")).toBeInTheDocument();
    expect(screen.getByText("$300.00")).toBeInTheDocument();
  });

  it("shows loading state and transient feedback on a partial refresh failure", async () => {
    const user = userEvent.setup();
    const { listHoldings, refreshPrices } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([neverRefreshed]);
    vi.mocked(refreshPrices).mockResolvedValue({
      refreshed: [],
      failed: [{ ticker: "AAPL", error: "price_fetch_failed" }],
    });

    renderWithProviders(<PortfolioPage />);

    await user.click(await screen.findByRole("button", { name: "Refresh prices" }));

    expect(refreshPrices).toHaveBeenCalled();
    await waitFor(() => expect(screen.getByText("Couldn't refresh AAPL")).toBeInTheDocument());
  });
});
