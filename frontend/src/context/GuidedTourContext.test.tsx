import { act } from "react";
import { render, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { GuidedTourProvider, useGuidedTour } from "@/context/GuidedTourContext";
import { TOUR_STEPS } from "@/components/tour/tourSteps";

const STORAGE_KEY = "plaincents:tourCompleted";

function wrapper({ children }: { children: React.ReactNode }) {
  return <GuidedTourProvider>{children}</GuidedTourProvider>;
}

describe("GuidedTourContext", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it("throws when used outside a GuidedTourProvider", () => {
    expect(() => renderHook(() => useGuidedTour())).toThrow(
      "useGuidedTour must be used within a GuidedTourProvider",
    );
  });

  it("starts inactive, at step 0, with hasCompletedTour reflecting localStorage", () => {
    window.localStorage.setItem(STORAGE_KEY, "true");
    const { result } = renderHook(() => useGuidedTour(), { wrapper });

    expect(result.current.isActive).toBe(false);
    expect(result.current.stepIndex).toBe(0);
    expect(result.current.totalSteps).toBe(TOUR_STEPS.length);
    expect(result.current.hasCompletedTour).toBe(true);
  });

  it("defaults hasCompletedTour to false when nothing is stored", () => {
    const { result } = renderHook(() => useGuidedTour(), { wrapper });

    expect(result.current.hasCompletedTour).toBe(false);
  });

  it("start() activates the tour at step 0", () => {
    const { result } = renderHook(() => useGuidedTour(), { wrapper });

    act(() => result.current.start());

    expect(result.current.isActive).toBe(true);
    expect(result.current.stepIndex).toBe(0);
  });

  it("next() advances stepIndex without completing before the last step", () => {
    const { result } = renderHook(() => useGuidedTour(), { wrapper });

    act(() => result.current.start());
    act(() => result.current.next());

    expect(result.current.stepIndex).toBe(1);
    expect(result.current.isActive).toBe(true);
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("next() on the last step ends the tour and persists completion", () => {
    const { result } = renderHook(() => useGuidedTour(), { wrapper });

    act(() => result.current.start());
    for (let i = 0; i < TOUR_STEPS.length - 1; i++) {
      act(() => result.current.next());
    }
    expect(result.current.stepIndex).toBe(TOUR_STEPS.length - 1);

    act(() => result.current.next());

    expect(result.current.isActive).toBe(false);
    expect(result.current.hasCompletedTour).toBe(true);
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe("true");
  });

  it("back() never goes below step 0", () => {
    const { result } = renderHook(() => useGuidedTour(), { wrapper });

    act(() => result.current.start());
    act(() => result.current.back());

    expect(result.current.stepIndex).toBe(0);
  });

  it("skip() ends the tour without marking it completed", () => {
    const { result } = renderHook(() => useGuidedTour(), { wrapper });

    act(() => result.current.start());
    act(() => result.current.next());
    act(() => result.current.skip());

    expect(result.current.isActive).toBe(false);
    expect(result.current.hasCompletedTour).toBe(false);
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("start() after a completed tour replays from step 0 (Replay Tour)", () => {
    const { result } = renderHook(() => useGuidedTour(), { wrapper });

    act(() => result.current.start());
    for (let i = 0; i < TOUR_STEPS.length; i++) {
      act(() => result.current.next());
    }
    expect(result.current.isActive).toBe(false);
    expect(result.current.hasCompletedTour).toBe(true);

    act(() => result.current.start());

    expect(result.current.isActive).toBe(true);
    expect(result.current.stepIndex).toBe(0);
  });

  it("degrades to hasCompletedTour=false when localStorage throws (private browsing etc.)", () => {
    const original = window.localStorage.getItem;
    window.localStorage.getItem = () => {
      throw new DOMException("blocked");
    };

    const { result } = renderHook(() => useGuidedTour(), { wrapper });
    expect(result.current.hasCompletedTour).toBe(false);

    window.localStorage.getItem = original;
  });
});

// Sanity check that `render` (not just renderHook) tolerates the provider
// with no consumer doing anything unusual -- guards against a provider-level
// crash that renderHook's minimal host might not surface.
describe("GuidedTourProvider", () => {
  it("renders its children without crashing", () => {
    const { getByText } = render(
      <GuidedTourProvider>
        <span>child</span>
      </GuidedTourProvider>,
    );
    expect(getByText("child")).toBeInTheDocument();
  });
});
