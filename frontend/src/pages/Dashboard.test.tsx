import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";
import type { DashboardSummaryResponse } from "@/types/dashboard";
import type { TransactionResponse } from "@/types/transaction";

import { DashboardPage } from "@/pages/Dashboard";

vi.mock("@/api/dashboard", () => ({
  getDashboardSummary: vi.fn(),
  getAvailableMonths: vi.fn().mockResolvedValue({ months: [] }),
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
    is_current_incomplete: true,
    current_month_has_data: false,
    total_spend_current: 0,
    total_spend_previous: 0,
    total_spend_previous_to_date: 0,
    comparable_day: 15,
    change_pct: 0,
    category_breakdown: [],
    spending_trend: [
      { month: "2026-01", total_spend: 0, has_data: true },
      { month: "2026-02", total_spend: 0, has_data: true },
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
    is_current_incomplete: true,
    current_month_has_data: true,
    total_spend_current: 150.5,
    total_spend_previous: 100,
    total_spend_previous_to_date: 100,
    comparable_day: 15,
    change_pct: 50.5,
    category_breakdown: [
      { category: "Food & Dining", total_spend: 150.5, pct_of_total: 100 },
    ],
    spending_trend: [
      { month: "2026-05", total_spend: 100, has_data: true },
      { month: "2026-06", total_spend: 150.5, has_data: true },
    ],
    recent_transactions: [baseTransaction],
    forecast_summary: null,
    portfolio_summary: null,
    data_mode: "REAL",
  };
}

describe("DashboardPage", () => {
  beforeEach(async () => {
    vi.resetAllMocks();
    const { getAvailableMonths } = await import("@/api/dashboard");
    // No dedicated assertions target the analysis-month selector here — a
    // single-month result hides it (see AnalysisMonthSelector), keeping
    // these tests focused on the summary data itself.
    vi.mocked(getAvailableMonths).mockResolvedValue({ months: [] });
  });

  it("renders the onboarding empty state when data_mode is EMPTY", async () => {
    const { getDashboardSummary } = await import("@/api/dashboard");
    vi.mocked(getDashboardSummary).mockResolvedValue(emptySummary());

    renderWithProviders(<DashboardPage />);

    expect(await screen.findByText("Welcome to PlainCents")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Import real data/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load demo data" })).toBeInTheDocument();
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

  it("shows honest 'not imported yet' copy instead of $0 when the current month has no data", async () => {
    // CURRENT-MONTH-DEFAULT FIX: current_month_has_data=false must never
    // render as "$0 spent" / a pace comparison -- this is the exact
    // behavior "no transactions imported this month" != "you spent $0"
    // guards against, at the component level.
    const { getDashboardSummary } = await import("@/api/dashboard");
    vi.mocked(getDashboardSummary).mockResolvedValue({
      ...realSummary(),
      total_spend_current: 0,
      current_month_has_data: false,
    });

    renderWithProviders(<DashboardPage />);

    expect(await screen.findByText("Not imported yet")).toBeInTheDocument();
    expect(screen.getByText(/No transactions imported for/)).toBeInTheDocument();
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
    expect(screen.queryByText(/^-?100(\.0)?%$/)).not.toBeInTheDocument();
  });
});
