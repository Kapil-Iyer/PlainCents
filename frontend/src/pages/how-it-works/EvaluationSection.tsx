import { useState } from "react";
import { motion, useReducedMotion } from "framer-motion";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  CATEGORIZATION_SPLIT_SIZES,
  FORECASTING_FINAL_RESULT,
  FORECASTING_FINAL_TRAIN_WINDOW,
  MERCHANT_ISOLATION_EXPLANATION,
  SEALED_FINAL_TEST_DISCIPLINE,
  SPLIT_ROLES,
  TEMPORAL_VALIDATION_EXPLANATION,
} from "@/data/methodology";

import { Disclosure } from "@/pages/how-it-works/Disclosure";

type EvalView = "categorization" | "forecasting";

const ZONES = [
  { id: "train", label: "TRAIN", dot: "bg-primary/50", ...CATEGORIZATION_SPLIT_SIZES.train },
  { id: "validation", label: "VALIDATION", dot: "bg-warning/60", ...CATEGORIZATION_SPLIT_SIZES.validation },
  { id: "final_test", label: "FINAL_TEST", dot: "bg-success/60", ...CATEGORIZATION_SPLIT_SIZES.final_test },
] as const;

function roleDescription(id: string) {
  return SPLIT_ROLES.find((r) => r.id === id)?.description ?? "";
}

function PartitionDiagram() {
  const prefersReducedMotion = useReducedMotion();
  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-muted-foreground">
        Each dot is one merchant group — an illustrative placeholder, not a real merchant name or
        dataset observation. Merchant-grouped splitting means every dot belongs to exactly one zone
        below, never more than one.
      </p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {ZONES.map((zone, zi) => (
          <div key={zone.id} className="rounded-lg border border-border p-3">
            <p className="text-sm font-semibold text-primary">{zone.label}</p>
            <p className="text-xs text-muted-foreground">
              {zone.rows} rows · {zone.merchantGroups} merchant groups
            </p>
            <div className="mt-2 flex flex-wrap gap-1" aria-hidden>
              {Array.from({ length: zone.merchantGroups }).map((_, i) => (
                <motion.span
                  key={i}
                  initial={prefersReducedMotion ? false : { opacity: 0, scale: 0.4 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.25, delay: prefersReducedMotion ? 0 : zi * 0.1 + i * 0.012 }}
                  className={cn("h-2 w-2 rounded-full", zone.dot)}
                />
              ))}
            </div>
            <p className="mt-2 text-xs text-muted-foreground">{roleDescription(zone.id)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function TemporalStrip() {
  const prefersReducedMotion = useReducedMotion();
  const { start, end, months } = FORECASTING_FINAL_TRAIN_WINDOW;
  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs text-muted-foreground">
        Schematic — illustrates the expanding-window protocol, not literal per-origin dates. Time
        flows left to right.
      </p>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <div className="relative h-8 flex-1 overflow-hidden rounded-md bg-muted">
          <motion.div
            className="h-full bg-primary/50"
            initial={prefersReducedMotion ? false : { width: "0%" }}
            animate={{ width: "88%" }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          />
        </div>
        <div className="flex gap-1">
          {FORECASTING_FINAL_RESULT.reservedMonths.map((m) => (
            <span
              key={m}
              className="flex h-8 w-10 items-center justify-center rounded-md border border-dashed border-success/60 bg-success/10 text-[10px] font-medium text-success"
            >
              {m.slice(5)}
            </span>
          ))}
        </div>
      </div>
      <div className="flex flex-col gap-1 text-xs text-muted-foreground sm:flex-row sm:justify-between">
        <span>
          TRAIN window: {start} → {end} ({months} months, expanding)
        </span>
        <span>Reserved (FINAL): {FORECASTING_FINAL_RESULT.reservedMonths.join(", ")}</span>
      </div>
    </div>
  );
}

export function EvaluationSection() {
  const [view, setView] = useState<EvalView>("categorization");

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <CardTitle>Evaluation Methodology</CardTitle>
          <CardDescription>How every number on this page was actually produced.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div
            role="group"
            aria-label="Evaluation view"
            className="inline-flex w-fit rounded-lg border border-border bg-card p-1"
          >
            <button
              type="button"
              aria-pressed={view === "categorization"}
              onClick={() => setView("categorization")}
              className={cn(
                "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                view === "categorization"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent",
              )}
            >
              Categorization split
            </button>
            <button
              type="button"
              aria-pressed={view === "forecasting"}
              onClick={() => setView("forecasting")}
              className={cn(
                "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                view === "forecasting"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent",
              )}
            >
              Forecasting timeline
            </button>
          </div>

          {view === "categorization" ? <PartitionDiagram /> : <TemporalStrip />}
        </CardContent>
      </Card>

      {view === "categorization" ? (
        <Disclosure summary="Read the full merchant-isolation & sealed-final-test methodology">
          <p>{MERCHANT_ISOLATION_EXPLANATION}</p>
          <p>{SEALED_FINAL_TEST_DISCIPLINE}</p>
        </Disclosure>
      ) : (
        <Disclosure summary="Read the full temporal-validation methodology">
          <p>{TEMPORAL_VALIDATION_EXPLANATION}</p>
        </Disclosure>
      )}
    </div>
  );
}
