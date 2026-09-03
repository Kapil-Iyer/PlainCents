import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ToastHost } from "@/components/shared/Toast";
import { DemoBanner } from "@/components/shared/DemoBanner";

const useAppState = vi.fn();
vi.mock("@/context/AppStateContext", () => ({
  useAppState: () => useAppState(),
}));

function renderBanner() {
  // Phase 11B: DemoBanner now hosts the "Clear demo data" control, which
  // uses useToast() — needs a ToastHost ancestor, same as every other page.
  return render(
    <ToastHost>
      <DemoBanner />
    </ToastHost>,
  );
}

describe("DemoBanner", () => {
  it("renders while mode is DEMO", () => {
    useAppState.mockReturnValue({ mode: "DEMO", clearDemo: vi.fn(), isClearingDemo: false });

    renderBanner();

    expect(screen.getByText(/Demo Data/)).toBeInTheDocument();
  });

  it("renders nothing when mode is EMPTY", () => {
    useAppState.mockReturnValue({ mode: "EMPTY", clearDemo: vi.fn(), isClearingDemo: false });

    renderBanner();

    // ToastHost always renders its own (empty) notification region now that
    // DemoBanner needs it as an ancestor, so the container itself is no
    // longer guaranteed empty — assert on the banner's own content instead.
    expect(screen.queryByText(/Demo Data/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Clear demo data" })).not.toBeInTheDocument();
  });

  it("renders nothing when mode is REAL", () => {
    useAppState.mockReturnValue({ mode: "REAL", clearDemo: vi.fn(), isClearingDemo: false });

    renderBanner();

    expect(screen.queryByText(/Demo Data/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Clear demo data" })).not.toBeInTheDocument();
  });

  it("clears demo data via the confirm dialog", async () => {
    const clearDemo = vi.fn().mockResolvedValue({});
    useAppState.mockReturnValue({ mode: "DEMO", clearDemo, isClearingDemo: false });
    const user = userEvent.setup();

    renderBanner();
    await user.click(screen.getByRole("button", { name: "Clear demo data" }));

    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: "Clear demo data" }));

    expect(clearDemo).toHaveBeenCalled();
  });
});
