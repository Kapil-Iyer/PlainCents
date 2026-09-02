import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";
import type { ForecastRunResponse, ForecastStatusResponse } from "@/types/forecast";

import { ForecastPage } from "@/pages/Forecast";

vi.mock("@/api/forecasts", () => ({
  getForecastStatus: vi.fn(),
  getLatestForecast: vi.fn(),
  runForecast: vi.fn(),
}));

function coldStartStatus(): ForecastStatusResponse {
  return {
    status: "cold_start",
    months_available: 5,
    months_required: 12,
    latest_run_id: null,
    is_stale: null,
  };
}

function noForecastYetStatus(): ForecastStatusResponse {
  return {
    status: "no_forecast_yet",
    months_available: 12,
    months_required: 12,
    latest_run_id: null,
    is_stale: null,
  };
}

function readyStatus(overrides: Partial<ForecastStatusResponse> = {}): ForecastStatusResponse {
  return {
    status: "ready",
    months_available: 12,
    months_required: 12,
    latest_run_id: 1,
    is_stale: false,
    ...overrides,
  };
}

function sampleRun(overrides: Partial<ForecastRunResponse> = {}): ForecastRunResponse {
  return {
    run_id: 1,
    generated_at: "2026-06-01T12:00:00Z",
    is_stale: false,
    stale_reason: null,
    months_available: 12,
    predictions: [
      {
        category: "Food & Dining",
        forecast_month: "2026-07",
        month_offset: 1,
        predicted_amount: 120.5,
        is_available: true,
        unavailable_reason: null,
      },
      {
        category: "Food & Dining",
        forecast_month: "2026-08",
        month_offset: 2,
        predicted_amount: 118.2,
        is_available: true,
        unavailable_reason: null,
      },
      {
        category: "Food & Dining",
        forecast_month: "2026-09",
        month_offset: 3,
        predicted_amount: 121.0,
        is_available: true,
        unavailable_reason: null,
      },
      {
        category: "Healthcare",
        forecast_month: "2026-07",
        month_offset: 1,
        predicted_amount: null,
        is_available: false,
        unavailable_reason: "insufficient_history",
      },
      {
        category: "Healthcare",
        forecast_month: "2026-08",
        month_offset: 2,
        predicted_amount: null,
        is_available: false,
        unavailable_reason: "insufficient_history",
      },
      {
        category: "Healthcare",
        forecast_month: "2026-09",
        month_offset: 3,
        predicted_amount: null,
        is_available: false,
        unavailable_reason: "insufficient_history",
      },
    ],
    ...overrides,
  };
}

describe("ForecastPage", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("renders the cold-start state when months_available < 12", async () => {
    const { getForecastStatus } = await import("@/api/forecasts");
    vi.mocked(getForecastStatus).mockResolvedValue(coldStartStatus());

    renderWithProviders(<ForecastPage />);

    expect(await screen.findByText("Not enough history yet")).toBeInTheDocument();
    expect(screen.getByText(/You have 5 so far/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /generate forecast/i })).not.toBeInTheDocument();
  });

  it("renders the stale warning when the latest run is stale", async () => {
    const { getForecastStatus, getLatestForecast } = await import("@/api/forecasts");
    vi.mocked(getForecastStatus).mockResolvedValue(readyStatus({ is_stale: true }));
    vi.mocked(getLatestForecast).mockResolvedValue(sampleRun({ is_stale: true }));

    renderWithProviders(<ForecastPage />);

    expect(await screen.findByText(/may be out of date/i)).toBeInTheDocument();
  });

  it("does not show the stale warning for a fresh, non-stale forecast", async () => {
    const { getForecastStatus, getLatestForecast } = await import("@/api/forecasts");
    vi.mocked(getForecastStatus).mockResolvedValue(readyStatus({ is_stale: false }));
    vi.mocked(getLatestForecast).mockResolvedValue(sampleRun({ is_stale: false }));

    renderWithProviders(<ForecastPage />);

    await screen.findByText("Per-category forecast");
    expect(screen.queryByText(/may be out of date/i)).not.toBeInTheDocument();
  });

  it("shows a loading state on the button while a forecast is being generated", async () => {
    const user = userEvent.setup();
    const { getForecastStatus, getLatestForecast, runForecast } = await import("@/api/forecasts");
    vi.mocked(getForecastStatus).mockResolvedValue(noForecastYetStatus());
    vi.mocked(getLatestForecast).mockResolvedValue({ status: "no_forecast_yet" });
    let resolveRun: (value: ForecastRunResponse) => void = () => {};
    vi.mocked(runForecast).mockReturnValue(
      new Promise((resolve) => {
        resolveRun = resolve;
      }),
    );

    renderWithProviders(<ForecastPage />);

    const button = await screen.findByRole("button", { name: /generate forecast/i });
    await user.click(button);

    expect(await screen.findByRole("button", { name: /generating/i })).toBeDisabled();

    resolveRun(sampleRun());
  });

  it("renders forecast results, including unavailable categories, after a successful run", async () => {
    const { getForecastStatus, getLatestForecast } = await import("@/api/forecasts");
    vi.mocked(getForecastStatus).mockResolvedValue(readyStatus());
    vi.mocked(getLatestForecast).mockResolvedValue(sampleRun());

    renderWithProviders(<ForecastPage />);

    expect(await screen.findByText("Per-category forecast")).toBeInTheDocument();
    expect(screen.getAllByText("Food & Dining").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Not enough history").length).toBeGreaterThan(0);
  });

  it("renders the empty state when eligible but no forecast has been generated yet", async () => {
    const { getForecastStatus, getLatestForecast } = await import("@/api/forecasts");
    vi.mocked(getForecastStatus).mockResolvedValue(noForecastYetStatus());
    vi.mocked(getLatestForecast).mockResolvedValue({ status: "no_forecast_yet" });

    renderWithProviders(<ForecastPage />);

    expect(await screen.findByText("No forecast yet")).toBeInTheDocument();
  });

  it("renders an error state when the status call fails", async () => {
    const { getForecastStatus } = await import("@/api/forecasts");
    vi.mocked(getForecastStatus).mockRejectedValue(new Error("network error"));

    renderWithProviders(<ForecastPage />);

    expect(await screen.findByText("Couldn't load the forecast")).toBeInTheDocument();
  });
});
