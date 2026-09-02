import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";
import type { DashboardSummaryResponse } from "@/types/dashboard";
import type { TransactionResponse } from "@/types/transaction";

import { DashboardPage } from "@/pages/Dashboard";

vi.mock("@/api/dashboard", () => ({
  getDashboardSummary: vi.fn(),
}));

vi.mock("@/api/demo", () => ({
  loadDemo: vi.fn(),
  clearDemo: vi.fn(),
}));

const baseTransaction: TransactionResponse = {
  id: 1,
  date: "2026-06-15",
  merchant: "Loblaws",
  raw_description: null,
  amount: 84.32,
  bank_source: "TD",
  predicted_category: "Food & Dining",
  confirmed_category: null,
  effective_category: "Food & Dining",
  is_manual_override: false,
  created_at: "2026-06-15T00:00:00Z",
  updated_at: "2026-06-15T00:00:00Z",
};

function emptySummary(): DashboardSummaryResponse {
  return {
    period: { current: "2026-06", previous: "2026-05" },
    total_spend_current: 0,
    total_spend_previous: 0,
    change_pct: 0,
    category_breakdown: [],
    spending_trend: [
      { month: "2026-01", total_spend: 0 },
      { month: "2026-02", total_spend: 0 },
    ],
    recent_transactions: [],
    forecast_summary: null,
    portfolio_summary: null,
    data_mode: "EMPTY",
  };
}

function realSummary(): DashboardSummaryResponse {
  return {
    period: { current: "2026-06", previous: "2026-05" },
    total_spend_current: 150.5,
    total_spend_previous: 100,
    change_pct: 50.5,
    category_breakdown: [
      { category: "Food & Dining", total_spend: 150.5, pct_of_total: 100 },
    ],
    spending_trend: [
      { month: "2026-05", total_spend: 100 },
      { month: "2026-06", total_spend: 150.5 },
    ],
    recent_transactions: [baseTransaction],
    forecast_summary: null,
    portfolio_summary: null,
    data_mode: "REAL",
  };
}

describe("DashboardPage", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("renders the onboarding empty state when data_mode is EMPTY", async () => {
    const { getDashboardSummary } = await import("@/api/dashboard");
    vi.mocked(getDashboardSummary).mockResolvedValue(emptySummary());

    renderWithProviders(<DashboardPage />);

    expect(await screen.findByText("Welcome to PlainCents")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Import real data/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Load demo data/ })).toBeInTheDocument();
  });

  it("renders an error state when the API call fails", async () => {
    const { getDashboardSummary } = await import("@/api/dashboard");
    vi.mocked(getDashboardSummary).mockRejectedValue(new Error("network error"));

    renderWithProviders(<DashboardPage />);

    expect(await screen.findByText("Couldn't load the dashboard")).toBeInTheDocument();
  });

  it("renders summary metrics, category breakdown, and recent transactions for real data", async () => {
    const { getDashboardSummary } = await import("@/api/dashboard");
    vi.mocked(getDashboardSummary).mockResolvedValue(realSummary());

    renderWithProviders(<DashboardPage />);

    expect(await screen.findByText("$150.50")).toBeInTheDocument();
    expect(screen.getByText("$100.00")).toBeInTheDocument();
    expect(screen.getByText("50.5%")).toBeInTheDocument();
    expect(screen.getByText("Loblaws")).toBeInTheDocument();
  });
});
