import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  LayoutGrid,
  LineChart,
  ShieldCheck,
  UploadCloud,
  UserCheck,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { PIPELINE_STEPS } from "@/data/methodology";

const ICONS = [UploadCloud, ShieldCheck, BrainCircuit, UserCheck, LineChart, LayoutGrid];

/** One illustrative example transaction, shown in a different representation
 * at each stage so the diagram demonstrates *what changes*, not just names
 * the stage. Purely illustrative — PIPELINE_STEPS itself is documented as
 * "purely presentational, not ML-evidence-bearing", and this never
 * substitutes for the evidence-bearing Categorization/Forecasting sections. */
const EXAMPLE = {
  raw: "2026-03-14,COFFEE HOUSE 482,-6.25",
  merchant: "COFFEE HOUSE 482",
  amount: -6.25,
  date: "2026-03-14",
  category: "Food & Dining",
} as const;

function StageToken({ stageId }: { stageId: string }) {
  switch (stageId) {
    case "csv":
      return (
        <code className="block rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">
          {EXAMPLE.raw}
        </code>
      );
    case "normalize":
      return (
        <code className="block whitespace-pre-wrap rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">
          {`{ merchant: "${EXAMPLE.merchant}", amount: ${EXAMPLE.amount}, date: "${EXAMPLE.date}" }`}
        </code>
      );
    case "categorize":
      return (
        <div className="flex flex-wrap items-center justify-center gap-2">
          <span className="text-xs text-muted-foreground">
            {EXAMPLE.merchant
              .toLowerCase()
              .split(" ")
              .map((word, i) => (
                <span
                  key={i}
                  className={cn("mr-1 rounded px-1", i < 2 && "bg-primary/15 text-primary")}
                >
                  {word}
                </span>
              ))}
          </span>
          <Badge variant="predicted">{EXAMPLE.category}</Badge>
        </div>
      );
    case "confirm":
      return (
        <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-muted-foreground">
          <span>Kept or corrected by you —</span>
          <Badge variant="confirmed">{EXAMPLE.category}</Badge>
        </div>
      );
    case "forecast":
      return (
        <div className="flex items-end justify-center gap-1.5" aria-hidden>
          {[18, 24, 20, 26].map((h, i) => (
            <div key={i} className="w-3 rounded-t bg-primary/60" style={{ height: h }} />
          ))}
          <div
            className="w-3 rounded-t border border-dashed border-primary/60 bg-primary/10"
            style={{ height: 22 }}
          />
        </div>
      );
    case "insights":
      return (
        <div className="flex flex-wrap justify-center gap-1.5">
          {["Dashboard", "Forecast", "Portfolio"].map((label) => (
            <span
              key={label}
              className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground"
            >
              {label}
            </span>
          ))}
        </div>
      );
    default:
      return null;
  }
}

/** Overview section: one illustrative transaction, followed stage by stage
 * through `Bank CSV → Normalize → Categorize → Confirm → Forecast →
 * Insights`. Controlled progression (station click / prev-next) — no
 * scroll-hijacking, normal page scroll and hash navigation are untouched. */
export function PipelineDiagram() {
  const prefersReducedMotion = useReducedMotion();
  const [active, setActive] = useState(0);
  const step = PIPELINE_STEPS[active];

  return (
    <Card variant="elevated">
      <CardHeader>
        <CardTitle>PlainCents in one pipeline</CardTitle>
        <CardDescription>
          One example transaction, followed stage by stage. Illustrative — not a real record.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div
          role="group"
          aria-label="Pipeline stage"
          className="flex flex-col gap-2 md:flex-row md:gap-1.5"
        >
          {PIPELINE_STEPS.map((s, i) => {
            const Icon = ICONS[i] ?? LayoutGrid;
            const isActive = i === active;
            return (
              <button
                key={s.id}
                type="button"
                aria-pressed={isActive}
                onClick={() => setActive(i)}
                className={cn(
                  "flex flex-1 flex-col items-center gap-1.5 rounded-lg border px-2 py-3 text-center transition-colors",
                  isActive
                    ? "border-primary/60 bg-primary/5"
                    : "border-border bg-card hover:border-border-strong/60",
                )}
              >
                <span
                  className={cn(
                    "flex h-9 w-9 items-center justify-center rounded-full",
                    isActive ? "bg-primary text-primary-foreground" : "bg-primary/15 text-primary",
                  )}
                >
                  <Icon className="h-4 w-4" />
                </span>
                <p className={cn("text-xs font-semibold", isActive && "text-primary")}>{s.label}</p>
                {isActive && (
                  <motion.div
                    layoutId="pipeline-active-underline"
                    className="h-0.5 w-8 rounded-full bg-primary"
                    transition={prefersReducedMotion ? { duration: 0 } : { type: "spring", stiffness: 500, damping: 40 }}
                  />
                )}
              </button>
            );
          })}
        </div>

        <div className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card px-4 py-5">
          <button
            type="button"
            aria-label="Previous stage"
            onClick={() => setActive((i) => Math.max(0, i - 1))}
            disabled={active === 0}
            className="shrink-0 rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent disabled:opacity-30"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>

          <div className="relative min-h-[64px] flex-1 text-center">
            {/* Entry never waits on the previous stage's exit animation to
             * finish — each stage's content is independently keyed and
             * absolutely positioned so a slow/stalled exit (e.g. a
             * throttled background tab) can never block the next stage's
             * content from appearing immediately. */}
            <AnimatePresence>
              <motion.div
                key={step.id}
                initial={prefersReducedMotion ? undefined : { opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={prefersReducedMotion ? undefined : { opacity: 0, y: -6 }}
                transition={{ duration: 0.2 }}
                className="absolute inset-0 flex flex-col items-center justify-center gap-2"
              >
                <StageToken stageId={step.id} />
                <p className="text-xs text-muted-foreground">{step.description}</p>
              </motion.div>
            </AnimatePresence>
          </div>

          <button
            type="button"
            aria-label="Next stage"
            onClick={() => setActive((i) => Math.min(PIPELINE_STEPS.length - 1, i + 1))}
            disabled={active === PIPELINE_STEPS.length - 1}
            className="shrink-0 rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent disabled:opacity-30"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </CardContent>
    </Card>
  );
}
