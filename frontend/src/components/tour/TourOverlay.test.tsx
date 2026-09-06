import { act } from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { MemoryRouter, useLocation } from "react-router-dom";

import { GuidedTourProvider, useGuidedTour } from "@/context/GuidedTourContext";
import { render } from "@testing-library/react";

import { TOUR_STEPS } from "@/components/tour/tourSteps";
import { TourOverlay } from "@/components/tour/TourOverlay";

const STORAGE_KEY = "plaincents:tourCompleted";

function LocationDisplay() {
  const location = useLocation();
  return <span data-testid="location">{location.pathname}</span>;
}

/** Every step's target element is present at once here (unlike the real
 * app, where only the current route's own elements exist) -- this isolates
 * the tour's own control-flow (stepping, navigation calls, Back/Next/Skip/
 * Done, localStorage persistence) from real page mounting, which the
 * Dashboard/Portfolio/etc. test files already exercise indirectly by
 * rendering with a GuidedTourProvider (see test/utils.tsx). */
function Harness() {
  const tour = useGuidedTour();
  return (
    <>
      <LocationDisplay />
      <button onClick={tour.start}>start-tour</button>
      {Array.from(new Set(TOUR_STEPS.map((s) => s.target))).map((target) => (
        <div key={target} data-tour={target}>
          {target}
        </div>
      ))}
      <TourOverlay />
    </>
  );
}

function renderHarness() {
  return render(
    <MemoryRouter initialEntries={["/dashboard"]}>
      <GuidedTourProvider>
        <Harness />
      </GuidedTourProvider>
    </MemoryRouter>,
  );
}

describe("TourOverlay", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it("renders nothing until the tour is started", () => {
    renderHarness();

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("shows the first step's title and body, and navigates to its route", async () => {
    const user = userEvent.setup();
    renderHarness();

    await user.click(screen.getByText("start-tour"));

    expect(await screen.findByText(TOUR_STEPS[0].title)).toBeInTheDocument();
    expect(screen.getByText(TOUR_STEPS[0].body)).toBeInTheDocument();
    expect(screen.getByText("Step 1 of " + TOUR_STEPS.length)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId("location").textContent).toBe(TOUR_STEPS[0].route));
  });

  it("Next advances the step and navigates to that step's route", async () => {
    const user = userEvent.setup();
    renderHarness();

    await user.click(screen.getByText("start-tour"));
    await screen.findByText(TOUR_STEPS[0].title);
    await user.click(screen.getByRole("button", { name: "Next" }));

    expect(await screen.findByText(TOUR_STEPS[1].title)).toBeInTheDocument();
    expect(screen.getByText("Step 2 of " + TOUR_STEPS.length)).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId("location").textContent).toBe(TOUR_STEPS[1].route));
  });

  it("Back is disabled on the first step and returns to the previous step otherwise", async () => {
    const user = userEvent.setup();
    renderHarness();

    await user.click(screen.getByText("start-tour"));
    await screen.findByText(TOUR_STEPS[0].title);
    expect(screen.getByRole("button", { name: "Back" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByText(TOUR_STEPS[1].title);
    await user.click(screen.getByRole("button", { name: "Back" }));

    expect(await screen.findByText(TOUR_STEPS[0].title)).toBeInTheDocument();
  });

  it("the last step's advance button reads Done and closes the tour without completing early", async () => {
    const user = userEvent.setup();
    renderHarness();

    await user.click(screen.getByText("start-tour"));
    for (let i = 0; i < TOUR_STEPS.length - 1; i++) {
      await screen.findByText(TOUR_STEPS[i].title);
      await user.click(screen.getByRole("button", { name: "Next" }));
    }

    expect(await screen.findByText(TOUR_STEPS.at(-1)!.title)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Done" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Done" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe("true");
  });

  it("Skip closes the tour without marking it completed", async () => {
    const user = userEvent.setup();
    renderHarness();

    await user.click(screen.getByText("start-tour"));
    await screen.findByText(TOUR_STEPS[0].title);
    await user.click(screen.getByText("Skip"));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("Escape closes the tour, same as Skip", async () => {
    const user = userEvent.setup();
    renderHarness();

    await user.click(screen.getByText("start-tour"));
    await screen.findByText(TOUR_STEPS[0].title);
    await act(async () => {
      await user.keyboard("{Escape}");
    });

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("covers portfolio holdings/analytics/how-it-works and Power BI as distinct steps", () => {
    const ids = TOUR_STEPS.map((s) => s.id);
    expect(ids).toEqual(
      expect.arrayContaining([
        "portfolio-holdings",
        "portfolio-analytics",
        "portfolio-how-it-works",
        "powerbi",
      ]),
    );
  });

  it("the Power BI step is honest about being a snapshot, not a live connection", () => {
    const step = TOUR_STEPS.find((s) => s.id === "powerbi")!;
    expect(step.body).toMatch(/snapshot/i);
    expect(step.body).toMatch(/not a live connection/i);
  });

  it("consecutive portfolio steps share one route without extra navigation", async () => {
    const user = userEvent.setup();
    renderHarness();

    await user.click(screen.getByText("start-tour"));
    const holdingsIndex = TOUR_STEPS.findIndex((s) => s.id === "portfolio-holdings");
    for (let i = 0; i < holdingsIndex; i++) {
      await screen.findByText(TOUR_STEPS[i].title);
      await user.click(screen.getByRole("button", { name: "Next" }));
    }

    await screen.findByText(TOUR_STEPS[holdingsIndex].title);
    await waitFor(() =>
      expect(screen.getByTestId("location").textContent).toBe(TOUR_STEPS[holdingsIndex].route),
    );
    await user.click(screen.getByRole("button", { name: "Next" }));

    expect(await screen.findByText(TOUR_STEPS[holdingsIndex + 1].title)).toBeInTheDocument();
    expect(screen.getByTestId("location").textContent).toBe(TOUR_STEPS[holdingsIndex + 1].route);
  });

  it("falls back to a centered card when the target element cannot be found", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/dashboard"]}>
        <GuidedTourProvider>
          <TourOverlayOnlyHarness />
        </GuidedTourProvider>
      </MemoryRouter>,
    );

    await user.click(screen.getByText("start-tour"));

    // No matching [data-tour] element exists anywhere -- the card must
    // still render (centered, per useSpotlightRect's documented fallback)
    // rather than the tour silently doing nothing.
    expect(await screen.findByText(TOUR_STEPS[0].title)).toBeInTheDocument();
  });
});

function TourOverlayOnlyHarness() {
  const tour = useGuidedTour();
  return (
    <>
      <button onClick={tour.start}>start-tour</button>
      <TourOverlay />
    </>
  );
}
