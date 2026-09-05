import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface Step {
  id: string;
  title: string;
  page: string;
  body: string;
  /** A miniature of the screen this step describes. Schematic on purpose:
   * a real screenshot would go stale the first time the UI moved. */
  visual: () => React.ReactElement;
}

const STEPS: Step[] = [
  {
    id: "start",
    title: "Start empty, or load the demo",
    page: "Dashboard",
    body: "A fresh install has nothing in it. You can either import your own statement straight away, or load a year of demo data to look around first. Demo data is kept completely separate from real data and clears in one click.",
    visual: () => (
      <Screen>
        <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
          <div className="h-10 w-10 rounded-full border-2 border-dashed border-border-strong" />
          <Bar w="w-28" />
          <Bar w="w-40" muted />
          <div className="mt-1 flex gap-2">
            <Pill label="Import a statement" active />
            <Pill label="Load demo data" />
          </div>
        </div>
      </Screen>
    ),
  },
  {
    id: "upload",
    title: "Upload a bank CSV",
    page: "Import",
    body: "Export a statement from RBC, Scotiabank, TD or CIBC and drop the file in. You can name the bank, or leave it on Auto-detect — PlainCents matches the file's columns against each bank's known export shape and refuses the file outright rather than guessing wrong.",
    visual: () => (
      <Screen>
        <div className="flex h-full flex-col gap-3">
          <Bar w="w-24" />
          <div className="flex flex-1 flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed border-border-strong">
            <Bar w="w-32" muted />
            <Bar w="w-20" muted />
          </div>
          <div className="flex justify-end">
            <Pill label="Upload & preview" active />
          </div>
        </div>
      </Screen>
    ),
  },
  {
    id: "preview",
    title: "Check the preview before anything is saved",
    page: "Import",
    body: "You see exactly what will be imported: how many purchases, how many rows are already in your account, and which were skipped because they were credits rather than spending. Every row shows the category it will be filed under — not a guess that changes later.",
    visual: () => (
      <Screen>
        <div className="flex h-full flex-col gap-2">
          <div className="grid grid-cols-4 gap-1.5">
            <Tile label="42" caption="to import" accent />
            <Tile label="3" caption="already in" />
            <Tile label="1" caption="unreadable" />
            <Tile label="2" caption="credits" />
          </div>
          <Rows n={4} />
        </div>
      </Screen>
    ),
  },
  {
    id: "confirm",
    title: "Confirm the import",
    page: "Import",
    body: "Only now is anything written. Rows already in your account are skipped, so re-importing an overlapping statement is safe. The categories that get stored are the ones the preview showed you.",
    visual: () => (
      <Screen>
        <div className="flex h-full flex-col items-center justify-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-success/15 text-success">
            ✓
          </div>
          <Bar w="w-32" />
          <Bar w="w-44" muted />
          <Pill label="View transactions" active />
        </div>
      </Screen>
    ),
  },
  {
    id: "review",
    title: "Review what it decided",
    page: "Transactions",
    body: "Every transaction is listed with its category. A category PlainCents chose itself is shown differently from one you set, so you can always tell the machine's opinion from your own.",
    visual: () => (
      <Screen>
        <div className="flex h-full flex-col gap-2">
          <div className="flex gap-1.5">
            <Pill label="Transactions" active />
            <Pill label="Insights" />
          </div>
          <Rows n={5} />
        </div>
      </Screen>
    ),
  },
  {
    id: "correct",
    title: "Correct anything it got wrong",
    page: "Transactions",
    body: "Change a category and it's saved immediately as yours. The original prediction is kept underneath, so the record of what the system thought never disappears.",
    visual: () => (
      <Screen>
        <div className="flex h-full flex-col gap-2">
          <Rows n={2} />
          <div className="rounded-md border border-primary/40 bg-primary/5 p-2">
            <div className="flex items-center gap-2">
              <Bar w="w-16" />
              <span className="text-[9px] text-muted-foreground line-through">Food &amp; Dining</span>
              <span className="text-[9px] text-primary">→ Healthcare</span>
            </div>
          </div>
          <Rows n={2} />
        </div>
      </Screen>
    ),
  },
  {
    id: "remember",
    title: "Watch it reuse your correction",
    page: "Import",
    body: "The next time that merchant appears — even with a different card number or store code in the description — your category is applied automatically, on that bank. You fix a merchant once, not every month.",
    visual: () => (
      <Screen>
        <div className="flex h-full flex-col justify-center gap-2.5">
          <MemoryRow suffix="4821" />
          <MemoryRow suffix="9137" remembered />
          <MemoryRow suffix="0284" remembered />
        </div>
      </Screen>
    ),
  },
  {
    id: "dashboard",
    title: "Read the dashboard",
    page: "Dashboard",
    body: "This month against last, whether you're ahead of your usual pace, and which categories account for the difference. Your corrections are already folded in — the charts and the transaction list never disagree.",
    visual: () => (
      <Screen>
        <div className="flex h-full flex-col gap-2">
          <div className="grid grid-cols-3 gap-1.5">
            <Tile label="$1,284" caption="this month" accent />
            <Tile label="$1,102" caption="last month" />
            <Tile label="+16%" caption="change" />
          </div>
          <div className="flex flex-1 items-end gap-1.5">
            {[40, 62, 48, 75, 58, 88].map((h, i) => (
              <div
                key={i}
                className="flex-1 rounded-sm bg-primary/35"
                style={{ height: `${h}%` }}
              />
            ))}
          </div>
        </div>
      </Screen>
    ),
  },
  {
    id: "insights",
    title: "Dig into categories and merchants",
    page: "Transactions → Insights",
    body: "How each category has moved over 6, 12 or 24 months, and which specific merchants take the biggest share. Merchants are grouped by identity, so one shop is one row rather than a dozen near-identical ones.",
    visual: () => (
      <Screen>
        <div className="flex h-full flex-col gap-2">
          <div className="flex gap-1.5">
            <Pill label="Transactions" />
            <Pill label="Insights" active />
          </div>
          <div className="flex flex-1 flex-col justify-center gap-2">
            {[92, 68, 51, 34].map((w, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full bg-primary/60" style={{ width: `${w}%` }} />
                </div>
                <Bar w="w-8" muted />
              </div>
            ))}
          </div>
        </div>
      </Screen>
    ),
  },
  {
    id: "forecast",
    title: "Get a forecast once there's history",
    page: "Forecast",
    body: "After three completed months, PlainCents will project the next three by category. Before that it tells you how many months are still needed rather than showing a number it can't stand behind.",
    visual: () => (
      <Screen>
        <div className="flex h-full items-end gap-2">
          {[52, 66, 44].map((h, i) => (
            <div key={i} className="flex flex-1 flex-col items-center gap-1">
              <div className="w-full rounded-sm bg-muted-foreground/35" style={{ height: `${h}px` }} />
              <span className="text-[8px] text-muted-foreground">M{i + 1}</span>
            </div>
          ))}
          <span className="pb-4 text-muted-foreground">→</span>
          {[54, 54, 54].map((h, i) => (
            <div key={i} className="flex flex-1 flex-col items-center gap-1">
              <div
                className="w-full rounded-sm border border-dashed border-primary bg-primary/20"
                style={{ height: `${h}px` }}
              />
              <span className="text-[8px] text-primary">+{i + 1}</span>
            </div>
          ))}
        </div>
      </Screen>
    ),
  },
];

/**
 * The app walkthrough: a stepper through the actual workflow, start to
 * finish.
 *
 * Schematic miniatures rather than screenshots — deliberately. A screenshot
 * is out of date the first time a button moves, and a walkthrough that shows
 * a UI the user cannot find is worse than no walkthrough. These abstract
 * shapes convey layout and sequence, which is what this section is for; the
 * video section below is where the real interface belongs.
 */
export function AppWalkthroughSection() {
  const [index, setIndex] = useState(0);
  const reduceMotion = useReducedMotion();
  const step = STEPS[index];

  const go = (next: number) => setIndex((next + STEPS.length) % STEPS.length);

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      go(index + 1);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      go(index - 1);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">Using PlainCents, step by step</h2>
        <p className="text-sm text-muted-foreground">
          The whole workflow, from an empty install to a forecast.
        </p>
      </div>

      <div
        role="tablist"
        aria-label="Walkthrough steps"
        onKeyDown={handleKeyDown}
        className="flex flex-wrap gap-1.5"
      >
        {STEPS.map((s, i) => (
          <button
            key={s.id}
            type="button"
            role="tab"
            aria-selected={i === index}
            aria-controls="walkthrough-panel"
            tabIndex={i === index ? 0 : -1}
            onClick={() => setIndex(i)}
            className={cn(
              "rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              i === index
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:text-foreground",
            )}
          >
            {i + 1}
          </button>
        ))}
      </div>

      <Card>
        <CardContent className="pt-6">
          <div
            id="walkthrough-panel"
            role="tabpanel"
            className="grid grid-cols-1 items-center gap-6 lg:grid-cols-2"
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={step.id}
                initial={reduceMotion ? false : { opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={reduceMotion ? { opacity: 1 } : { opacity: 0, x: -12 }}
                transition={{ duration: 0.22, ease: "easeOut" }}
                className="flex flex-col gap-3"
              >
                <div className="flex items-center gap-2">
                  <Badge variant="secondary">
                    Step {index + 1} of {STEPS.length}
                  </Badge>
                  <Badge variant="outline">{step.page}</Badge>
                </div>
                <h3 className="text-lg font-semibold">{step.title}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">{step.body}</p>
              </motion.div>
            </AnimatePresence>

            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={`${step.id}-visual`}
                initial={reduceMotion ? false : { opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={reduceMotion ? { opacity: 1 } : { opacity: 0, scale: 0.98 }}
                transition={{ duration: 0.22, ease: "easeOut" }}
              >
                {step.visual()}
              </motion.div>
            </AnimatePresence>
          </div>

          <div className="mt-6 flex items-center justify-between gap-3 border-t border-border pt-4">
            <Button variant="outline" size="sm" onClick={() => go(index - 1)}>
              <ChevronLeft className="h-4 w-4" />
              Previous
            </Button>
            <span className="text-xs tabular-nums text-muted-foreground">
              {index + 1} / {STEPS.length}
            </span>
            <Button variant="outline" size="sm" onClick={() => go(index + 1)}>
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

/* -- schematic primitives -------------------------------------------------- */

function Screen({ children }: { children: React.ReactNode }) {
  return (
    <div
      aria-hidden
      className="aspect-[4/3] w-full rounded-lg border border-border bg-elevated p-3 shadow-sm"
    >
      {children}
    </div>
  );
}

function Bar({ w, muted }: { w: string; muted?: boolean }) {
  return <span className={cn("block h-2 rounded-full", w, muted ? "bg-muted" : "bg-muted-foreground/40")} />;
}

function Pill({ label, active }: { label: string; active?: boolean }) {
  return (
    <span
      className={cn(
        "rounded-full px-2 py-0.5 text-[9px]",
        active ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
      )}
    >
      {label}
    </span>
  );
}

function Tile({ label, caption, accent }: { label: string; caption: string; accent?: boolean }) {
  return (
    <div className={cn("rounded-md px-1.5 py-1", accent ? "bg-primary/10" : "bg-muted/60")}>
      <p className={cn("text-[10px] font-semibold tabular-nums", accent && "text-primary")}>{label}</p>
      <p className="text-[8px] text-muted-foreground">{caption}</p>
    </div>
  );
}

function Rows({ n }: { n: number }) {
  return (
    <div className="flex flex-col gap-1.5">
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} className="flex items-center gap-2 rounded border border-border/60 px-2 py-1.5">
          <Bar w="w-10" muted />
          <Bar w="w-16" />
          <span className="ml-auto">
            <Bar w="w-8" muted />
          </span>
        </div>
      ))}
    </div>
  );
}

function MemoryRow({ suffix, remembered }: { suffix: string; remembered?: boolean }) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded border px-2 py-1.5 text-[9px]",
        remembered ? "border-success/40 bg-success/5" : "border-border/60",
      )}
    >
      <span className="text-muted-foreground">…{suffix} CAREWELL PHARMACY</span>
      <span className={cn("ml-auto", remembered ? "text-success" : "text-muted-foreground")}>
        {remembered ? "Healthcare ✓ yours" : "Healthcare"}
      </span>
    </div>
  );
}
