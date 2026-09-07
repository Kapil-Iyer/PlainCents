import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, useLocation } from "react-router-dom";

import { GuidedTourProvider, useGuidedTour } from "@/context/GuidedTourContext";
import { render } from "@testing-library/react";

import { TourOverlay } from "@/components/tour/TourOverlay";
import { TOUR_STEPS } from "@/components/tour/tourSteps";

/**
 * Isolated regression test for the tour spotlight bug fixed by keying
 * useSpotlightRect's locate effect on the step index, not just the target
 * selector string. The REAL tourSteps.ts (as of the Dashboard/Forecast
 * granularity pass) no longer has two directly-consecutive steps sharing a
 * target -- every visual now gets its own distinct target, which is good
 * product design but means the bug's exact original trigger (two
 * back-to-back steps reusing "page-header" for a different real element)
 * can no longer be exercised against the real step list.
 *
 * The underlying mechanism is still worth guarding directly, independent of
 * whatever tourSteps.ts happens to contain later -- so this file mocks a
 * minimal two-step tour where BOTH steps intentionally target the same
 * data-tour value on different routes/elements, isolated from
 * TourOverlay.test.tsx's real-step coverage.
 */
vi.mock("@/components/tour/tourSteps", () => ({
  TOUR_STEPS: [
    { id: "a", route: "/page-a", target: "shared", title: "Step A", body: "Body A" },
    { id: "b", route: "/page-b", target: "shared", title: "Step B", body: "Body B" },
  ],
}));

function RouteAwareHarness() {
  const tour = useGuidedTour();
  const location = useLocation();
  return (
    <>
      <button onClick={tour.start}>start-tour</button>
      {TOUR_STEPS.filter((s) => s.route === location.pathname).map((s) => (
        <div key={s.id} data-tour={s.target}>
          {s.target} on {location.pathname}
        </div>
      ))}
      <TourOverlay />
    </>
  );
}

function renderHarness() {
  return render(
    <MemoryRouter initialEntries={["/page-a"]}>
      <GuidedTourProvider>
        <RouteAwareHarness />
      </GuidedTourProvider>
    </MemoryRouter>,
  );
}

describe("TourOverlay -- same data-tour value reused by consecutive steps", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
  });

  it("re-locates the spotlight target when the next step reuses the same selector for a different element", async () => {
    const user = userEvent.setup();
    const addSpy = vi.spyOn(window, "addEventListener");
    renderHarness();

    await user.click(screen.getByText("start-tour"));
    await screen.findByText("Step A");
    const resizeListenersAtA = addSpy.mock.calls.filter((c) => c[0] === "resize").length;
    expect(resizeListenersAtA).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByText("Step B");
    const resizeListenersAtB = addSpy.mock.calls.filter((c) => c[0] === "resize").length;

    // A fresh `addEventListener("resize", ...)` call on arriving at Step B
    // proves the locate effect actually re-ran -- without keying on the
    // step index, this count stays flat, since the dependency array looks
    // unchanged to React ("shared" === "shared").
    expect(resizeListenersAtB).toBeGreaterThan(resizeListenersAtA);

    addSpy.mockRestore();
  });
});
