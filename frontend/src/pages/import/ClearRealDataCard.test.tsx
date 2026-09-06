import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";
import { AppStateProvider } from "@/context/AppStateContext";

import { ClearRealDataCard } from "@/pages/import/ClearRealDataCard";

vi.mock("@/api/demo", () => ({
  loadDemo: vi.fn(),
  clearDemo: vi.fn(),
  clearRealData: vi.fn(),
}));
vi.mock("@/api/health", () => ({
  getDemoStatus: vi.fn(),
}));

async function renderWithMode(mode: "EMPTY" | "DEMO" | "REAL") {
  const { getDemoStatus } = await import("@/api/health");
  vi.mocked(getDemoStatus).mockResolvedValue({ mode, can_load_demo: mode === "EMPTY" });

  return renderWithProviders(
    <AppStateProvider>
      <ClearRealDataCard />
    </AppStateProvider>,
  );
}

describe("ClearRealDataCard", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("renders nothing when mode is EMPTY", async () => {
    await renderWithMode("EMPTY");
    await waitFor(() => expect(screen.queryByText("Danger zone")).not.toBeInTheDocument());
  });

  it("renders nothing when mode is DEMO", async () => {
    await renderWithMode("DEMO");
    await waitFor(() => expect(screen.queryByText("Danger zone")).not.toBeInTheDocument());
  });

  it("shows the danger zone and requires confirmation before clearing when mode is REAL", async () => {
    const user = userEvent.setup();
    const { clearRealData } = await import("@/api/demo");
    vi.mocked(clearRealData).mockResolvedValue({ mode: "EMPTY", cleared: true, summary: { transactions: 3 } });

    await renderWithMode("REAL");
    await screen.findByText("Danger zone");

    await user.click(screen.getByRole("button", { name: "Clear all real data" }));
    expect(clearRealData).not.toHaveBeenCalled(); // not yet -- confirmation dialog first

    await screen.findByText("Clear all real data?");
    await user.click(screen.getByRole("button", { name: "Clear all real data" })); // confirm in dialog

    await waitFor(() => expect(clearRealData).toHaveBeenCalled());
  });

  it("shows an error toast and does not crash if clearing fails", async () => {
    const user = userEvent.setup();
    const { clearRealData } = await import("@/api/demo");
    vi.mocked(clearRealData).mockRejectedValue(new Error("network error"));

    await renderWithMode("REAL");
    await screen.findByText("Danger zone");

    await user.click(screen.getByRole("button", { name: "Clear all real data" }));
    await screen.findByText("Clear all real data?");
    const dialogButtons = screen.getAllByRole("button", { name: "Clear all real data" });
    await user.click(dialogButtons[dialogButtons.length - 1]);

    await waitFor(() => expect(screen.getByText("Couldn't clear real data")).toBeInTheDocument());
  });
});
