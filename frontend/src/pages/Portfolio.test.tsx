import { screen, waitFor, within } from "@testing-library/react";
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
  price_is_demo_snapshot: false,
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

const unknownCostBasis: HoldingResponse = {
  ...neverRefreshed,
  id: 4,
  ticker: "TSLA",
  avg_cost: null,
  current_price: 250,
  current_value: 2500,
  pnl: null,
  price_last_updated: "2026-01-15T00:00:00",
};

const demoSnapshot: HoldingResponse = {
  ...neverRefreshed,
  id: 3,
  ticker: "VTI",
  current_price: 268.75,
  current_value: 2687.5,
  pnl: 1175.0,
  price_last_updated: "2024-01-01T00:00:00",
  price_is_demo_snapshot: true,
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

    const table = await screen.findByRole("table");
    expect(within(table).getByText("AAPL")).toBeInTheDocument();
    expect(within(table).getByText("Not yet refreshed")).toBeInTheDocument();
    expect(within(table).getByText("$300.00")).toBeInTheDocument();
  });

  it("labels a demo-seeded price as a snapshot, never a real fetch date", async () => {
    const { listHoldings } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([demoSnapshot]);

    renderWithProviders(<PortfolioPage />);

    const table = await screen.findByRole("table");
    expect(within(table).getByText("VTI")).toBeInTheDocument();
    expect(within(table).getByText(/Demo snapshot/)).toBeInTheDocument();
    expect(within(table).queryByText(/^as of/)).not.toBeInTheDocument();
  });

  it("labels a genuinely cached price as a real fetch, not a demo snapshot", async () => {
    const { listHoldings } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([cached]);

    renderWithProviders(<PortfolioPage />);

    const table = await screen.findByRole("table");
    expect(within(table).getByText("MSFT")).toBeInTheDocument();
    expect(within(table).getByText(/^as of/)).toBeInTheDocument();
    expect(within(table).queryByText(/Demo snapshot/)).not.toBeInTheDocument();
  });

  it("shows honest dashes for a holding with no recorded cost basis (market value still works)", async () => {
    const { listHoldings } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([unknownCostBasis]);

    renderWithProviders(<PortfolioPage />);

    const table = await screen.findByRole("table");
    expect(within(table).getByRole("button", { name: "Add cost basis" })).toBeInTheDocument();
    expect(within(table).getByText("$2,500.00")).toBeInTheDocument(); // market value still shown
    expect(within(table).getAllByText("—").length).toBeGreaterThan(0); // P&L honestly unavailable
    // Portfolio Analytics reflects the same honesty: no fabricated P&L bar.
    expect(await screen.findByText("Cost basis unavailable")).toBeInTheDocument();
  });

  it("clicking Add cost basis opens the edit dialog for that holding", async () => {
    const user = userEvent.setup();
    const { listHoldings } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([unknownCostBasis]);

    renderWithProviders(<PortfolioPage />);
    await user.click(await screen.findByRole("button", { name: "Add cost basis" }));

    expect(await screen.findByText("Edit holding")).toBeInTheDocument();
  });

  it("creates a holding without an average cost", async () => {
    const user = userEvent.setup();
    const { listHoldings, createHolding } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([]);
    vi.mocked(createHolding).mockResolvedValue({ ...unknownCostBasis, ticker: "MSFT" });

    renderWithProviders(<PortfolioPage />);
    const openButtons = await screen.findAllByRole("button", { name: "Add holding" });
    await user.click(openButtons[0]);
    const dialog = await screen.findByRole("dialog");
    await user.type(within(dialog).getByLabelText("Ticker"), "MSFT");
    await user.type(within(dialog).getByLabelText("Shares"), "10");
    await user.click(within(dialog).getByRole("button", { name: "Add holding" }));

    await waitFor(() =>
      expect(createHolding).toHaveBeenCalledWith({ ticker: "MSFT", shares: 10, avg_cost: null }),
    );
  });

  it("the purchase-lot calculator computes a weighted average and applies it to the form", async () => {
    const user = userEvent.setup();
    const { listHoldings } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([]);

    renderWithProviders(<PortfolioPage />);
    const openButtons = await screen.findAllByRole("button", { name: "Add holding" });
    await user.click(openButtons[0]);
    await user.click(screen.getByRole("button", { name: "Calculate from purchases" }));

    const shareInputs = screen.getAllByPlaceholderText("5");
    const priceInputs = screen.getAllByPlaceholderText("100.00");
    await user.type(shareInputs[0], "5");
    await user.type(priceInputs[0], "100");
    await user.type(shareInputs[1], "3");
    await user.type(priceInputs[1], "120");
    await user.click(screen.getByRole("button", { name: "Add purchase" }));
    const shareInputs2 = screen.getAllByPlaceholderText("5");
    const priceInputs2 = screen.getAllByPlaceholderText("100.00");
    await user.type(shareInputs2[2], "2");
    await user.type(priceInputs2[2], "140");

    // (5*100 + 3*120 + 2*140) / 10 = 114
    expect(await screen.findByText("$114.00")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Use this average cost" }));

    expect(screen.getByLabelText("Average cost per share (optional)")).toHaveValue(114);
  });

  it("editing a holding to clear its average cost warns and sends null", async () => {
    const user = userEvent.setup();
    const { listHoldings, updateHolding } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([{ ...unknownCostBasis, ticker: "MSFT", avg_cost: 100 }]);
    vi.mocked(updateHolding).mockResolvedValue(unknownCostBasis);

    renderWithProviders(<PortfolioPage />);
    await user.click(await screen.findByRole("button", { name: "Edit MSFT" }));
    const dialog = await screen.findByRole("dialog");
    const avgCostInput = within(dialog).getByLabelText("Average cost per share (optional)");
    await user.clear(avgCostInput);

    expect(within(dialog).getByText(/Clearing this will remove/)).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "Save changes" }));

    await waitFor(() =>
      expect(updateHolding).toHaveBeenCalledWith(unknownCostBasis.id, {
        shares: unknownCostBasis.shares,
        avg_cost: null,
      }),
    );
  });

  it("renders Portfolio Analytics with correct summary metrics from real holdings", async () => {
    const { listHoldings } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([
      { ...cached, ticker: "AAPL", avg_cost: 100, shares: 10, current_price: 150, current_value: 1500, pnl: 500 },
      { ...unknownCostBasis, ticker: "TSLA", shares: 5, current_price: 200, current_value: 1000, pnl: null },
    ]);

    renderWithProviders(<PortfolioPage />);
    await screen.findByText("Portfolio analytics");

    expect(screen.getByText("$2,500.00")).toBeInTheDocument(); // total market value: 1500+1000
    expect(screen.queryByText("2 of 2")).not.toBeInTheDocument(); // only 1 of 2 has cost basis
    expect(screen.getByText("1 of 2")).toBeInTheDocument();
  });

  it("shows an honest dash, never a fabricated $0.00, when cost basis is known but nothing is priced yet", async () => {
    const { listHoldings } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([
      { ...neverRefreshed, ticker: "MSFT", avg_cost: 280, current_price: null, current_value: null, pnl: null },
    ]);

    renderWithProviders(<PortfolioPage />);
    await screen.findByText("Portfolio analytics");

    expect(screen.getByText("$2,800.00")).toBeInTheDocument(); // known cost basis: 10*280
    expect(screen.getByText("No priced holdings with a known cost basis yet")).toBeInTheDocument();
  });

  it("Portfolio Analytics never renders for an empty portfolio", async () => {
    const { listHoldings } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([]);

    renderWithProviders(<PortfolioPage />);

    await screen.findByText("No holdings yet");
    expect(screen.queryByText("Portfolio analytics")).not.toBeInTheDocument();
  });

  it("How your portfolio works renders even with no holdings yet", async () => {
    const { listHoldings } = await import("@/api/holdings");
    vi.mocked(listHoldings).mockResolvedValue([]);

    renderWithProviders(<PortfolioPage />);

    expect(await screen.findByText("How your portfolio works")).toBeInTheDocument();
    expect(
      screen.getByText(/never affect your transaction totals/),
    ).toBeInTheDocument();
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
