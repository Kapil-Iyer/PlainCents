import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";

import { OnboardingEmptyState } from "@/components/OnboardingEmptyState";

vi.mock("@/api/demo", () => ({
  loadDemo: vi.fn(),
  clearDemo: vi.fn(),
}));

describe("OnboardingEmptyState", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("offers Import real data, Load demo data, and Load demo data & start tour, distinctly", () => {
    renderWithProviders(<OnboardingEmptyState />);

    expect(screen.getByRole("link", { name: /Import real data/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load demo data" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Load demo data & start tour/ })).toBeInTheDocument();
  });

  it("hides Load demo data & start tour when showWalkthrough is false", () => {
    renderWithProviders(<OnboardingEmptyState showWalkthrough={false} />);

    expect(
      screen.queryByRole("button", { name: /Load demo data & start tour/ }),
    ).not.toBeInTheDocument();
    // The plain, tour-less path must still be there.
    expect(screen.getByRole("button", { name: "Load demo data" })).toBeInTheDocument();
  });

  it("Load demo data & start tour loads Demo data first, then starts the tour -- never against an empty app", async () => {
    const user = userEvent.setup();
    const { loadDemo } = await import("@/api/demo");
    vi.mocked(loadDemo).mockResolvedValue({ mode: "DEMO", summary: { transactions: 100 } });

    renderWithProviders(<OnboardingEmptyState />);
    await user.click(screen.getByRole("button", { name: /Load demo data & start tour/ }));

    // Demo data must be loaded (awaited) before the tour starts -- the tour
    // spotlights real charts that don't exist yet in EMPTY mode, so
    // starting it first would leave every chart step un-anchored.
    await waitFor(() => expect(loadDemo).toHaveBeenCalled());
  });

  it("plain Load demo data never starts the tour", async () => {
    const user = userEvent.setup();
    const { loadDemo } = await import("@/api/demo");
    vi.mocked(loadDemo).mockResolvedValue({ mode: "DEMO", summary: { transactions: 100 } });

    renderWithProviders(<OnboardingEmptyState />);
    await user.click(screen.getByRole("button", { name: "Load demo data" }));

    expect(loadDemo).toHaveBeenCalled();
    await waitFor(() => expect(screen.getByText("Demo data loaded")).toBeInTheDocument());
  });

  it("shows an error toast if loading demo data fails, without crashing", async () => {
    const user = userEvent.setup();
    const { loadDemo } = await import("@/api/demo");
    vi.mocked(loadDemo).mockRejectedValue(new Error("network error"));

    renderWithProviders(<OnboardingEmptyState />);
    await user.click(screen.getByRole("button", { name: "Load demo data" }));

    await waitFor(() => expect(screen.getByText("Couldn't load demo data")).toBeInTheDocument());
  });

  it("shows an error toast if Load demo data & start tour fails to load demo data, without crashing", async () => {
    const user = userEvent.setup();
    const { loadDemo } = await import("@/api/demo");
    vi.mocked(loadDemo).mockRejectedValue(new Error("network error"));

    renderWithProviders(<OnboardingEmptyState />);
    await user.click(screen.getByRole("button", { name: /Load demo data & start tour/ }));

    await waitFor(() => expect(screen.getByText("Couldn't load demo data")).toBeInTheDocument());
  });
});
