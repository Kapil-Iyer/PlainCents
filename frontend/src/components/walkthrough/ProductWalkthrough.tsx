import * as React from "react";
import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

import { WALKTHROUGH_STAGES } from "@/components/walkthrough/walkthroughStages";

const AUTOPLAY_MS = 4000;

/**
 * Recruiter/product walkthrough (Build Plan Phase 10, authorized addition).
 *
 * Presentation-only: this component owns no application data and calls no
 * API — see walkthroughStages.tsx for the static content it renders. It is
 * explicitly NOT "Explore Demo" (which loads the real, interactive Phase 9
 * demo dataset via useLoadDemo()); this only shows what the product does.
 *
 * Restrained autoplay through the 5 stages, pausing permanently on any user
 * interaction (manual nav, hover, or focus) so it never fights the reader.
 * Respects prefers-reduced-motion by not autoplaying at all.
 */
export function ProductWalkthrough() {
  const [index, setIndex] = React.useState(0);
  const [playing, setPlaying] = React.useState(true);
  const prefersReducedMotion = usePrefersReducedMotion();

  const autoplayActive = playing && !prefersReducedMotion;

  React.useEffect(() => {
    if (!autoplayActive) return;
    const id = window.setInterval(() => {
      setIndex((i) => (i + 1) % WALKTHROUGH_STAGES.length);
    }, AUTOPLAY_MS);
    return () => window.clearInterval(id);
  }, [autoplayActive]);

  const goTo = (next: number) => {
    setIndex(((next % WALKTHROUGH_STAGES.length) + WALKTHROUGH_STAGES.length) % WALKTHROUGH_STAGES.length);
    setPlaying(false);
  };

  const stage = WALKTHROUGH_STAGES[index];

  return (
    <div
      className="mx-auto w-full max-w-xl overflow-hidden rounded-xl border border-border bg-card shadow-sm"
      role="region"
      aria-roledescription="carousel"
      aria-label="PlainCents product walkthrough"
      onMouseEnter={() => setPlaying(false)}
    >
      {/* Browser-frame chrome */}
      <div className="flex items-center gap-1.5 border-b border-border bg-muted/50 px-3 py-2">
        <span className="h-2.5 w-2.5 rounded-full bg-destructive/40" />
        <span className="h-2.5 w-2.5 rounded-full bg-warning/40" />
        <span className="h-2.5 w-2.5 rounded-full bg-success/40" />
        <span className="ml-3 truncate rounded-sm bg-background px-2 py-0.5 text-[11px] text-muted-foreground">
          plaincents.app/{stage.id}
        </span>
      </div>

      <div className="border-b border-border px-4 pt-4">
        <p className="text-xs font-medium uppercase tracking-wide text-primary">{stage.eyebrow}</p>
        <h3 className="text-base font-semibold">{stage.title}</h3>
        <p className="pb-4 text-sm text-muted-foreground">{stage.description}</p>
      </div>

      <div
        key={stage.id}
        aria-live="polite"
        className={cn("min-h-[220px] bg-background", !prefersReducedMotion && "animate-in fade-in duration-300")}
      >
        {stage.render()}
      </div>

      <div className="flex items-center justify-between gap-2 border-t border-border px-3 py-2.5">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          aria-label="Previous stage"
          onClick={() => goTo(index - 1)}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>

        <div className="flex items-center gap-1.5" role="tablist" aria-label="Walkthrough stages">
          {WALKTHROUGH_STAGES.map((s, i) => (
            <button
              key={s.id}
              type="button"
              role="tab"
              aria-selected={i === index}
              aria-label={`Go to ${s.label}`}
              onClick={() => goTo(i)}
              className={cn(
                "h-2 w-2 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                i === index ? "bg-primary" : "bg-muted-foreground/30 hover:bg-muted-foreground/50",
              )}
            />
          ))}
        </div>

        <div className="flex items-center gap-1">
          {!prefersReducedMotion && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label={playing ? "Pause autoplay" : "Resume autoplay"}
              onClick={() => setPlaying((p) => !p)}
            >
              {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
          )}
          <Button type="button" variant="ghost" size="icon" aria-label="Next stage" onClick={() => goTo(index + 1)}>
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState(
    () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );

  React.useEffect(() => {
    const mql = window.matchMedia("(prefers-reduced-motion: reduce)");
    const handler = () => setReduced(mql.matches);
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, []);

  return reduced;
}
