import * as React from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useLocation, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { useGuidedTour } from "@/context/GuidedTourContext";
import { TOUR_STEPS } from "@/components/tour/tourSteps";

const CARD_WIDTH = 320;
const CARD_MARGIN = 12;
const SPOTLIGHT_PADDING = 8;
// How long to keep retrying to find the target element after navigating to
// its route before giving up and showing the card un-anchored (centered) --
// covers a lazy-loaded route's Suspense fallback swapping in the real page.
const LOCATE_TIMEOUT_MS = 3000;

interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
}

function rectFromElement(el: Element): Rect {
  const r = el.getBoundingClientRect();
  return { top: r.top, left: r.left, width: r.width, height: r.height };
}

/** Locates the current step's target element in the live DOM, retrying
 * (route transitions and lazy-loaded pages mean the element may not exist
 * on the first paint) until found or LOCATE_TIMEOUT_MS elapses, then keeps
 * its position in sync with scroll/resize while it stays mounted. */
function useSpotlightRect(selector: string | null): Rect | null {
  const [rect, setRect] = React.useState<Rect | null>(null);

  React.useEffect(() => {
    if (!selector) return;
    let cancelled = false;
    let rafId: number;
    const startedAt = performance.now();

    const locate = () => {
      if (cancelled) return;
      const el = document.querySelector(`[data-tour="${selector}"]`);
      if (el) {
        setRect(rectFromElement(el));
        return;
      }
      if (performance.now() - startedAt < LOCATE_TIMEOUT_MS) {
        rafId = requestAnimationFrame(locate);
      } else {
        setRect(null);
      }
    };
    setRect(null);
    locate();

    const reposition = () => {
      const el = document.querySelector(`[data-tour="${selector}"]`);
      if (el) setRect(rectFromElement(el));
    };
    window.addEventListener("scroll", reposition, true);
    window.addEventListener("resize", reposition);

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
      window.removeEventListener("scroll", reposition, true);
      window.removeEventListener("resize", reposition);
    };
  }, [selector]);

  return rect;
}

/**
 * Dazia-style spotlight guided tour over the LIVE app (PATCH C): a dark
 * overlay with a cutout around a real element (via the box-shadow trick --
 * no canvas, no SVG mask), an explanation card, and Back/Next/Skip/Done
 * controls that also drive real route navigation between steps. Replaces
 * the retired presentation-only ProductWalkthrough slideshow, which never
 * touched the real app at all.
 *
 * Mounted once in AppShell so it can spotlight elements on any route; keeps
 * its own visibility state entirely from GuidedTourContext.
 */
export function TourOverlay() {
  const { isActive, stepIndex, totalSteps, next, back, skip } = useGuidedTour();
  const location = useLocation();
  const navigate = useNavigate();
  const reduceMotion = useReducedMotion();

  const step = TOUR_STEPS[stepIndex];

  // Route transition: navigate to this step's page if we're not already
  // there. A no-op (no extra history entry) when already on the route.
  React.useEffect(() => {
    if (!isActive || !step) return;
    if (location.pathname !== step.route) {
      navigate(step.route);
    }
    // Only re-run when the step itself changes (or the tour (re)starts) --
    // not on every location change, which would fight a legitimate user
    // navigation away from the tour's own route.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive, step]);

  const rect = useSpotlightRect(isActive ? (step?.target ?? null) : null);

  React.useEffect(() => {
    if (!isActive) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") skip();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isActive, skip]);

  if (!isActive || !step) return null;

  const isLast = stepIndex === totalSteps - 1;

  const cardPos = (() => {
    if (!rect) {
      // No target found (or still loading) -- center the card so the tour
      // can still be read and dismissed rather than showing nothing.
      return {
        top: window.innerHeight / 2 - 110,
        left: window.innerWidth / 2 - CARD_WIDTH / 2,
      };
    }
    const spaceBelow = window.innerHeight - rect.top - rect.height;
    const top = spaceBelow > 240 ? rect.top + rect.height + CARD_MARGIN : rect.top - 240 - CARD_MARGIN;
    const left = Math.min(
      Math.max(rect.left, CARD_MARGIN),
      window.innerWidth - CARD_WIDTH - CARD_MARGIN,
    );
    return { top: Math.max(CARD_MARGIN, top), left };
  })();

  return (
    <AnimatePresence>
      <motion.div
        key="tour-backdrop"
        className="fixed inset-0 z-[100]"
        initial={reduceMotion ? false : { opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        role="dialog"
        aria-modal="true"
        aria-label={`Guided tour: ${step.title}`}
      >
        {/* The spotlight itself: a transparent rectangle whose oversized
         * box-shadow paints the dark backdrop everywhere EXCEPT this rect --
         * spatial cutout, not a color overlay, so the real element under it
         * stays fully visible and readable. */}
        {rect ? (
          <div
            aria-hidden
            className="absolute rounded-lg ring-2 ring-primary transition-all duration-300"
            style={{
              top: rect.top - SPOTLIGHT_PADDING,
              left: rect.left - SPOTLIGHT_PADDING,
              width: rect.width + SPOTLIGHT_PADDING * 2,
              height: rect.height + SPOTLIGHT_PADDING * 2,
              boxShadow: "0 0 0 9999px rgba(0, 0, 0, 0.88)",
            }}
          />
        ) : (
          <div aria-hidden className="absolute inset-0 bg-black/80" />
        )}

        <motion.div
          key={step.id}
          initial={reduceMotion ? false : { opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25 }}
          className="absolute flex flex-col gap-3 rounded-lg border border-border bg-card p-4 shadow-lg"
          style={{ top: cardPos.top, left: cardPos.left, width: CARD_WIDTH }}
        >
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Step {stepIndex + 1} of {totalSteps}
            </span>
            <button
              type="button"
              onClick={skip}
              className="text-xs font-medium text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              Skip
            </button>
          </div>

          <div>
            <h2 className="text-sm font-semibold">{step.title}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{step.body}</p>
          </div>

          <div className="flex items-center gap-1.5" aria-hidden>
            {TOUR_STEPS.map((s, i) => (
              <span
                key={s.id}
                className={`h-1.5 flex-1 rounded-full transition-colors ${
                  i <= stepIndex ? "bg-primary" : "bg-muted"
                }`}
              />
            ))}
          </div>

          <div className="flex items-center justify-between gap-2">
            <Button type="button" variant="ghost" size="sm" onClick={back} disabled={stepIndex === 0}>
              Back
            </Button>
            <Button type="button" size="sm" onClick={next}>
              {isLast ? "Done" : "Next"}
            </Button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
