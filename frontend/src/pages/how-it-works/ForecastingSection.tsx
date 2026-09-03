import { motion, useReducedMotion } from "framer-motion";
import { CheckCircle2, XCircle } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  FORECASTING_CANDIDATES,
  FORECASTING_FINAL_RESULT,
  FORECASTING_PIPELINE_EXPLANATION,
  FORECASTING_SELECTION_RATIONALE,
  FORECASTING_STRATEGY_NA_NOTE,
} from "@/data/methodology";

import { EvidenceBadge, LimitationNote } from "@/pages/how-it-works/EvidenceBadge";

function strategyLabel(strategy: "last-known-history" | "recursive" | null) {
  if (strategy === null) return "N/A";
  return strategy === "last-known-history" ? "Last-known" : "Recursive";
}

export function ForecastingSection() {
  const prefersReducedMotion = useReducedMotion();

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
            14 expanding-window origins. Ridge and Random Forest each ran under two multi-step
            strategies — last-known-history and recursive — kept as separate rows, exactly as the
            evidence represents them.
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-muted-foreground">
                <th className="py-2 pr-4 font-medium">Model</th>
                <th className="py-2 pr-4 font-medium">Strategy</th>
                <th className="py-2 pr-4 text-right font-medium">+1</th>
                <th className="py-2 pr-4 text-right font-medium">+2</th>
                <th className="py-2 pr-4 text-right font-medium">+3</th>
                <th className="py-2 pr-4 text-right font-medium">Result</th>
              </tr>
            </thead>
            <tbody>
              {FORECASTING_CANDIDATES.map((c, i) => (
                <motion.tr
                  key={c.id + c.strategy}
                  initial={prefersReducedMotion ? undefined : { opacity: 0 }}
                  whileInView={prefersReducedMotion ? undefined : { opacity: 1 }}
                  viewport={{ once: true, margin: "-40px" }}
                  transition={{ duration: 0.25, delay: i * 0.05 }}
                  className={cn("border-b border-border last:border-0", c.selected && "bg-primary/5")}
                >
                  <td className={cn("py-2 pr-4 font-medium", c.selected && "text-primary")}>{c.label}</td>
                  <td className="py-2 pr-4 text-muted-foreground">{strategyLabel(c.strategy)}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">{c.validationWape.h1.toFixed(4)}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">{c.validationWape.h2.toFixed(4)}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">{c.validationWape.h3.toFixed(4)}</td>
                  <td className="py-2 pr-4 text-right">
                    {c.selected ? (
                      <CheckCircle2 className="ml-auto h-4 w-4 text-primary" aria-label="Selected" />
                    ) : (
                      <XCircle className="ml-auto h-4 w-4 text-muted-foreground" aria-label="Not selected" />
                    )}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-muted-foreground">{FORECASTING_STRATEGY_NA_NOTE}</p>
        </CardContent>
      </Card>

      <Card variant="elevated">
        <CardHeader>
          <CardTitle>Why Naive was selected</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="flex flex-col gap-2 text-sm text-muted-foreground">
            {FORECASTING_SELECTION_RATIONALE.map((r, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-primary" aria-hidden>•</span>
                {r}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

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
