import * as React from "react";

import { TOUR_STEPS } from "@/components/tour/tourSteps";

const STORAGE_KEY = "plaincents:tourCompleted";

/** No localStorage precedent existed anywhere else in this codebase before
 * this feature — every read/write here is wrapped in try/catch so a private
 * window, blocked site data, or a disabled storage API degrades to "treat
 * the tour as not yet completed" rather than throwing. */
function readCompletedFlag(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

function writeCompletedFlag(): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, "true");
  } catch {
    // Best-effort only — a failed write just means "Replay Tour" always
    // has something to do next session instead of nothing; never a crash.
  }
}

interface GuidedTourContextValue {
  /** Whether the spotlight overlay is currently showing. */
  isActive: boolean;
  stepIndex: number;
  totalSteps: number;
  /** Persisted across sessions via localStorage -- purely informational
   * (e.g. for a first-run nudge); starting the tour never depends on it. */
  hasCompletedTour: boolean;
  start: () => void;
  next: () => void;
  back: () => void;
  /** Ends the tour early without marking it complete -- "Skip" is not
   * "Done": a skipped tour should still nudge the user again later. */
  skip: () => void;
}

const GuidedTourContext = React.createContext<GuidedTourContextValue | null>(null);

export function GuidedTourProvider({ children }: { children: React.ReactNode }) {
  const [isActive, setIsActive] = React.useState(false);
  const [stepIndex, setStepIndex] = React.useState(0);
  const [hasCompletedTour, setHasCompletedTour] = React.useState(readCompletedFlag);

  const start = React.useCallback(() => {
    setStepIndex(0);
    setIsActive(true);
  }, []);

  const next = React.useCallback(() => {
    setStepIndex((i) => {
      if (i + 1 >= TOUR_STEPS.length) {
        setIsActive(false);
        writeCompletedFlag();
        setHasCompletedTour(true);
        return i;
      }
      return i + 1;
    });
  }, []);

  const back = React.useCallback(() => {
    setStepIndex((i) => Math.max(0, i - 1));
  }, []);

  const skip = React.useCallback(() => {
    setIsActive(false);
  }, []);

  const value = React.useMemo<GuidedTourContextValue>(
    () => ({
      isActive,
      stepIndex,
      totalSteps: TOUR_STEPS.length,
      hasCompletedTour,
      start,
      next,
      back,
      skip,
    }),
    [isActive, stepIndex, hasCompletedTour, start, next, back, skip],
  );

  return <GuidedTourContext.Provider value={value}>{children}</GuidedTourContext.Provider>;
}

export function useGuidedTour(): GuidedTourContextValue {
  const ctx = React.useContext(GuidedTourContext);
  if (!ctx) {
    throw new Error("useGuidedTour must be used within a GuidedTourProvider");
  }
  return ctx;
}
