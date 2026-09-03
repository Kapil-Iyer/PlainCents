import { useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { CheckCircle2 } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  FORECASTING_CANDIDATES,
  FORECASTING_FINAL_RESULT,
  FORECASTING_PIPELINE_EXPLANATION,
  FORECASTING_SELECTION_RATIONALE,
  FORECASTING_STRATEGY_NA_NOTE,
  type ForecastingCandidate,
} from "@/data/methodology";

import { Disclosure } from "@/pages/how-it-works/Disclosure";
import { EvidenceBadge, LimitationNote } from "@/pages/how-it-works/EvidenceBadge";

function strategyLabel(strategy: "last-known-history" | "recursive" | null) {
  if (strategy === null) return "N/A";
  return strategy === "last-known-history" ? "Last-known" : "Recursive";
}

/** Shared y-domain across every tile so the *shape* of degradation is
 * directly comparable — a candidate that rises toward +3 visibly rises,
 * one that stays flat visibly stays flat. Domain is headroom around the
 * actual min/max evaluated WAPE values (0.1765-0.2887), not a fabricated
 * reference range. */
const WAPE_MIN = 0.15;
const WAPE_MAX = 0.3;
const SPARK_W = 100;
const SPARK_H = 36;

function wapeToY(w: number) {
  const t = (w - WAPE_MIN) / (WAPE_MAX - WAPE_MIN);
  return SPARK_H - Math.max(0, Math.min(1, t)) * SPARK_H;
}

function Sparkline({ h1, h2, h3, dashed }: { h1: number; h2: number; h3: number; dashed: boolean }) {
  const prefersReducedMotion = useReducedMotion();
  const points: [number, number][] = [
    [0, wapeToY(h1)],
    [SPARK_W / 2, wapeToY(h2)],
    [SPARK_W, wapeToY(h3)],
  ];
  const path = `M ${points[0][0]} ${points[0][1]} L ${points[1][0]} ${points[1][1]} L ${points[2][0]} ${points[2][1]}`;

  return (
    <svg
      viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
      className="h-9 w-full"
      preserveAspectRatio="none"
      aria-hidden
    >
      <motion.path
        d={path}
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeDasharray={dashed ? "4 3" : undefined}
        initial={prefersReducedMotion ? false : { pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      />
      {points.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={2.5} fill="currentColor" />
      ))}
    </svg>
  );
}

function ForecastTile({ candidate }: { candidate: ForecastingCandidate }) {
  const [expanded, setExpanded] = useState(false);
  const dashed = candidate.strategy === "recursive";

  return (
    <div
      className={cn(
        "flex flex-col gap-2 rounded-lg border p-3",
        candidate.selected ? "border-primary/60 bg-primary/5" : "border-border",
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className={cn("text-sm font-semibold", candidate.selected && "text-primary")}>
          {candidate.label}
        </span>
        {candidate.selected ? (
          <span className="flex items-center gap-1 text-xs font-medium text-primary">
            <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />
            Selected
          </span>
        ) : (
          <span className="text-xs text-muted-foreground">
            Strategy: <span className="font-medium text-foreground">{strategyLabel(candidate.strategy)}</span>
          </span>
        )}
      </div>
      {candidate.selected && (
        <span className="text-xs text-muted-foreground">
          Strategy: <span className="font-medium text-foreground">{strategyLabel(candidate.strategy)}</span>
        </span>
      )}

      <div className={candidate.selected ? "text-primary" : "text-muted-foreground"}>
        <Sparkline
          h1={candidate.validationWape.h1}
          h2={candidate.validationWape.h2}
          h3={candidate.validationWape.h3}
          dashed={dashed}
        />
      </div>
      <div className="flex justify-between text-[11px] tabular-nums text-muted-foreground">
        <span>+1 {candidate.validationWape.h1.toFixed(4)}</span>
        <span>+2 {candidate.validationWape.h2.toFixed(4)}</span>
        <span>+3 {candidate.validationWape.h3.toFixed(4)}</span>
      </div>

      {candidate.rejectionReason && (
        <>
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
            className="self-start text-xs font-medium text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
          >
            {expanded ? "Hide reason" : "Why rejected?"}
          </button>
          {expanded && <p className="text-xs text-muted-foreground">{candidate.rejectionReason}</p>}
        </>
      )}
    </div>
  );
}

export function ForecastingSection() {
  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <CardTitle>Spending Forecasting</CardTitle>
          <CardDescription>{FORECASTING_PIPELINE_EXPLANATION}</CardDescription>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Candidates benchmarked — pooled VALIDATION WAPE by horizon</CardTitle>
          <CardDescription>
            14 expanding-window origins. Lower WAPE is better. Ridge and Random Forest each ran under
            two multi-step strategies — last-known-history (solid line) and recursive (dashed line) —
            kept as separate tiles, exactly as the evidence represents them. Naive and Seasonal Naive
            have no strategy axis — shown as N/A.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <p className="text-xs text-muted-foreground">{FORECASTING_STRATEGY_NA_NOTE}</p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {FORECASTING_CANDIDATES.map((c) => (
              <ForecastTile key={c.id + (c.strategy ?? "")} candidate={c} />
            ))}
          </div>
        </CardContent>
      </Card>

      <Disclosure summary="Why Naive was selected" defaultOpen>
        <ul className="flex flex-col gap-2">
          {FORECASTING_SELECTION_RATIONALE.map((r, i) => (
            <li key={i} className="flex gap-2">
              <span className="text-primary" aria-hidden>
                •
              </span>
              {r}
            </li>
          ))}
        </ul>
      </Disclosure>

      <Card>
        <CardHeader className="flex-row items-start justify-between gap-3">
          <div>
            <CardTitle>Reserved-period final result</CardTitle>
            <CardDescription>{FORECASTING_FINAL_RESULT.resultLabel}</CardDescription>
          </div>
          <EvidenceBadge tier={FORECASTING_FINAL_RESULT.evidenceTier} />
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Combined WAPE" value={`${FORECASTING_FINAL_RESULT.combinedWapePct.toFixed(2)}%`} />
            <Stat label="+1 month" value={`${FORECASTING_FINAL_RESULT.byHorizonWapePct.h1.toFixed(2)}%`} />
            <Stat label="+2 months" value={`${FORECASTING_FINAL_RESULT.byHorizonWapePct.h2.toFixed(2)}%`} />
            <Stat label="+3 months" value={`${FORECASTING_FINAL_RESULT.byHorizonWapePct.h3.toFixed(2)}%`} />
          </div>
          <p className="text-xs text-muted-foreground">
            Reserved months: {FORECASTING_FINAL_RESULT.reservedMonths.join(", ")} ·{" "}
            {FORECASTING_FINAL_RESULT.nPredictions} predictions
          </p>
          <LimitationNote>{FORECASTING_FINAL_RESULT.limitation}</LimitationNote>
        </CardContent>
      </Card>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-muted/50 px-3 py-2">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold tabular-nums">{value}</p>
    </div>
  );
}
