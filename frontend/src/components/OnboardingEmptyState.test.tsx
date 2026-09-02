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

  it("offers both Import real data and Load demo data, distinctly", () => {
    renderWithProviders(<OnboardingEmptyState />);

    expect(screen.getByRole("link", { name: /Import real data/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Load demo data/ })).toBeInTheDocument();
  });

  it("loading demo data calls the load API and shows a success toast", async () => {
    const user = userEvent.setup();
    const { loadDemo } = await import("@/api/demo");
    vi.mocked(loadDemo).mockResolvedValue({ mode: "DEMO", summary: { transactions: 100 } });

    renderWithProviders(<OnboardingEmptyState />);
    await user.click(screen.getByRole("button", { name: /Load demo data/ }));

    expect(loadDemo).toHaveBeenCalled();
    await waitFor(() => expect(screen.getByText("Demo data loaded")).toBeInTheDocument());
  });

  it("shows an error toast if loading demo data fails, without crashing", async () => {
    const user = userEvent.setup();
    const { loadDemo } = await import("@/api/demo");
    vi.mocked(loadDemo).mockRejectedValue(new Error("network error"));

    renderWithProviders(<OnboardingEmptyState />);
    await user.click(screen.getByRole("button", { name: /Load demo data/ }));

    await waitFor(() => expect(screen.getByText("Couldn't load demo data")).toBeInTheDocument());
  });
});
