import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";

import { GuidedTourProvider } from "@/context/GuidedTourContext";
import { TopNav } from "@/components/layout/TopNav";

const useAppState = vi.fn();
vi.mock("@/context/AppStateContext", () => ({
  useAppState: () => useAppState(),
}));

vi.mock("@/api/health", () => ({
  getHealth: vi.fn().mockResolvedValue({ categorization_model: "loaded" }),
}));

function renderTopNav() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <GuidedTourProvider>
          <TopNav />
        </GuidedTourProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("TopNav", () => {
  it("shows the mode badge, tagged for the guided tour to spotlight", () => {
    useAppState.mockReturnValue({ mode: "DEMO" });

    renderTopNav();

    const badge = screen.getByText("Demo");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveAttribute("data-tour", "topnav-mode-badge");
  });

  it("Replay tour is always available and starts the guided tour without crashing", async () => {
    useAppState.mockReturnValue({ mode: "REAL", loadDemo: vi.fn(), isLoadingDemo: false });
    const user = userEvent.setup();

    renderTopNav();
    await user.click(screen.getByRole("button", { name: /Replay tour/ }));

    // No TourOverlay is mounted here (that's TourOverlay.test.tsx's job) --
    // this only guards the entry point itself against a crash.
    await waitFor(() => expect(screen.getByRole("button", { name: /Replay tour/ })).toBeInTheDocument());
  });

  it("Replay tour loads Demo data first when there's nothing loaded yet, before starting", async () => {
    const loadDemo = vi.fn().mockResolvedValue({ mode: "DEMO", summary: { transactions: 1 } });
    useAppState.mockReturnValue({ mode: "EMPTY", loadDemo, isLoadingDemo: false });
    const user = userEvent.setup();

    renderTopNav();
    await user.click(screen.getByRole("button", { name: /Replay tour/ }));

    // The tour spotlights real charts that don't exist yet in EMPTY mode --
    // Replay tour must not start against an empty app.
    await waitFor(() => expect(loadDemo).toHaveBeenCalled());
  });

  it("Replay tour does not attempt to load Demo data when Demo or Real data already exists", async () => {
    const loadDemo = vi.fn();
    useAppState.mockReturnValue({ mode: "DEMO", loadDemo, isLoadingDemo: false });
    const user = userEvent.setup();

    renderTopNav();
    await user.click(screen.getByRole("button", { name: /Replay tour/ }));

    expect(loadDemo).not.toHaveBeenCalled();
  });
});
