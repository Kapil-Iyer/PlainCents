import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { ToastHost } from "@/components/shared/Toast";
import { DemoReentryBanner } from "@/components/shared/DemoReentryBanner";

const useAppState = vi.fn();
vi.mock("@/context/AppStateContext", () => ({
  useAppState: () => useAppState(),
}));

function renderBanner(path = "/forecast") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <ToastHost>
        <DemoReentryBanner />
      </ToastHost>
    </MemoryRouter>,
  );
}

describe("DemoReentryBanner", () => {
  it("renders a Load demo data action on a non-Dashboard route while EMPTY", () => {
    useAppState.mockReturnValue({ mode: "EMPTY", loadDemo: vi.fn(), isLoadingDemo: false });

    renderBanner("/forecast");

    expect(screen.getByRole("button", { name: "Load demo data" })).toBeInTheDocument();
  });

  it("renders nothing on the Dashboard route (which already has its own onboarding CTA)", () => {
    useAppState.mockReturnValue({ mode: "EMPTY", loadDemo: vi.fn(), isLoadingDemo: false });

    renderBanner("/dashboard");

    expect(screen.queryByRole("button", { name: "Load demo data" })).not.toBeInTheDocument();
  });

  it("renders nothing while mode is DEMO", () => {
    useAppState.mockReturnValue({ mode: "DEMO", loadDemo: vi.fn(), isLoadingDemo: false });

    renderBanner("/forecast");

    expect(screen.queryByRole("button", { name: "Load demo data" })).not.toBeInTheDocument();
  });

  it("renders nothing while mode is REAL", () => {
    useAppState.mockReturnValue({ mode: "REAL", loadDemo: vi.fn(), isLoadingDemo: false });

    renderBanner("/forecast");

    expect(screen.queryByRole("button", { name: "Load demo data" })).not.toBeInTheDocument();
  });

  it("loads demo data via the action button, from a non-Dashboard route", async () => {
    const loadDemo = vi.fn().mockResolvedValue({});
    useAppState.mockReturnValue({ mode: "EMPTY", loadDemo, isLoadingDemo: false });
    const user = userEvent.setup();

    renderBanner("/transactions");
    await user.click(screen.getByRole("button", { name: "Load demo data" }));

    expect(loadDemo).toHaveBeenCalled();
  });
});
