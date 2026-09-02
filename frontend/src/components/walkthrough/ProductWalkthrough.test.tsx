import { act, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/utils";

import { ProductWalkthrough } from "@/components/walkthrough/ProductWalkthrough";
import { WALKTHROUGH_STAGES } from "@/components/walkthrough/walkthroughStages";

function mockReducedMotion(matches: boolean) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  }));
}

describe("ProductWalkthrough", () => {
  beforeEach(() => {
    mockReducedMotion(false);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders all five stage labels for direct-stage selection", () => {
    renderWithProviders(<ProductWalkthrough />);

    for (const stage of WALKTHROUGH_STAGES) {
      expect(screen.getByRole("tab", { name: `Go to ${stage.label}` })).toBeInTheDocument();
    }
  });

  it("starts on stage 1 (Import) and advances forward on Next", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ProductWalkthrough />);

    expect(screen.getByText("01 — Import")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Next stage" }));
    expect(screen.getByText("02 — Transactions")).toBeInTheDocument();
  });

  it("supports direct-stage selection via the tab dots", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ProductWalkthrough />);

    await user.click(screen.getByRole("tab", { name: "Go to Portfolio" }));
    expect(screen.getByText("05 — Portfolio")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Go to Portfolio" })).toHaveAttribute("aria-selected", "true");
  });

  it("wraps Previous from the first stage to the last", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ProductWalkthrough />);

    await user.click(screen.getByRole("button", { name: "Previous stage" }));
    expect(screen.getByText("05 — Portfolio")).toBeInTheDocument();
  });

  it("autoplays forward on a timer", () => {
    vi.useFakeTimers();
    renderWithProviders(<ProductWalkthrough />);

    expect(screen.getByText("01 — Import")).toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(4000);
    });
    expect(screen.getByText("02 — Transactions")).toBeInTheDocument();
  });

  it("pauses autoplay after a manual interaction", () => {
    vi.useFakeTimers();
    renderWithProviders(<ProductWalkthrough />);

    act(() => {
      screen.getByRole("tab", { name: "Go to Forecast" }).click();
    });
    expect(screen.getByText("04 — Forecast")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(10000);
    });
    // Manual selection stops autoplay — the stage should not have advanced.
    expect(screen.getByText("04 — Forecast")).toBeInTheDocument();
  });

  it("does not offer a play/pause control and does not autoplay under prefers-reduced-motion", () => {
    mockReducedMotion(true);
    vi.useFakeTimers();
    renderWithProviders(<ProductWalkthrough />);

    expect(screen.queryByRole("button", { name: /autoplay/i })).not.toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(10000);
    });
    expect(screen.getByText("01 — Import")).toBeInTheDocument();
  });

  it("preserves manual navigation controls alongside stage content (CTA-adjacent, not a replacement)", () => {
    renderWithProviders(<ProductWalkthrough />);

    expect(screen.getByRole("button", { name: "Previous stage" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next stage" })).toBeInTheDocument();
  });
});
