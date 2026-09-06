import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";
import type { HoldingResponse } from "@/types/holding";

import { HoldingsTable } from "@/pages/portfolio/HoldingsTable";

vi.mock("@/api/holdings", () => ({
  deleteHolding: vi.fn().mockResolvedValue({ id: 1, deleted: true }),
  createHolding: vi.fn(),
  updateHolding: vi.fn(),
}));

const mockHoldings: HoldingResponse[] = [
  {
    id: 1,
    ticker: "AAPL",
    shares: 10,
    avg_cost: 100,
    current_price: 150,
    current_value: 1500,
    pnl: 500,
    price_last_updated: "2026-01-15T00:00:00",
    price_is_demo_snapshot: false,
    created_at: "2026-01-15T00:00:00Z",
    updated_at: "2026-01-15T00:00:00Z",
  },
  {
    id: 2,
    ticker: "TSLA",
    shares: 5,
    avg_cost: 200,
    current_price: null,
    current_value: null,
    pnl: null,
    price_last_updated: null,
    price_is_demo_snapshot: false,
    created_at: "2026-01-16T00:00:00Z",
    updated_at: "2026-01-16T00:00:00Z",
  },
];

describe("HoldingsTable", () => {
  it("renders holdings with price/P&L, and a never-refreshed row without a fabricated value", () => {
    renderWithProviders(<HoldingsTable holdings={mockHoldings} />);

    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("TSLA")).toBeInTheDocument();
    expect(screen.getByText("$500.00")).toBeInTheDocument(); // AAPL's pnl
    expect(screen.getByText("Not yet refreshed")).toBeInTheDocument(); // TSLA
  });

  it("requires confirmation before deleting a holding", async () => {
    const user = userEvent.setup();
    const { deleteHolding } = await import("@/api/holdings");
    renderWithProviders(<HoldingsTable holdings={mockHoldings} />);

    await user.click(screen.getByRole("button", { name: "Delete AAPL" }));

    expect(await screen.findByText("Delete this holding?")).toBeInTheDocument();
    expect(deleteHolding).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Delete" }));

    expect(deleteHolding).toHaveBeenCalledWith(1);
  });

  it("opens the edit dialog without a ticker field editable", async () => {
    const user = userEvent.setup();
    renderWithProviders(<HoldingsTable holdings={mockHoldings} />);

    await user.click(screen.getByRole("button", { name: "Edit AAPL" }));

    const dialog = await screen.findByRole("dialog");
    expect(dialog).toHaveTextContent("Edit holding");
    expect(screen.getByLabelText("Ticker")).toBeDisabled();
  });
});
