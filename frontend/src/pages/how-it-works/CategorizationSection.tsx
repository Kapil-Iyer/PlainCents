import { motion, useReducedMotion } from "framer-motion";
import { CheckCircle2, XCircle } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  CATEGORIZATION_CANDIDATES,
  CATEGORIZATION_FINAL_RESULT,
  CATEGORIZATION_PIPELINE_EXPLANATION,
  CATEGORIZATION_SELECTION_RATIONALE,
} from "@/data/methodology";

import { EvidenceBadge, LimitationNote } from "@/pages/how-it-works/EvidenceBadge";

export function CategorizationSection() {
  const prefersReducedMotion = useReducedMotion();

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <CardTitle>Transaction Categorization</CardTitle>
          <CardDescription>{CATEGORIZATION_PIPELINE_EXPLANATION}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
            <code className="rounded bg-muted px-2 py-1 text-xs">merchant description</code>
            <span aria-hidden>→</span>
            <code className="rounded bg-muted px-2 py-1 text-xs">TF-IDF</code>
            <span aria-hidden>→</span>
            <code className="rounded bg-muted px-2 py-1 text-xs">Logistic Regression</code>
            <span aria-hidden>→</span>
            <code className="rounded bg-muted px-2 py-1 text-xs">predicted category</code>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Candidates benchmarked on VALIDATION</CardTitle>
          <CardDescription>
            All three candidates, same frozen TRAIN/VALIDATION split. Exact accuracy and macro F1 —
            never rounded beyond what the source report itself presents.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {CATEGORIZATION_CANDIDATES.map((c, i) => (
            <motion.div
              key={c.id}
              initial={prefersReducedMotion ? undefined : { opacity: 0, y: 10 }}
              whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.3, delay: i * 0.08 }}
              className={cn(
                "flex flex-col gap-2 rounded-lg border p-4 sm:flex-row sm:items-center sm:justify-between",
                c.selected ? "border-primary/60 bg-primary/5 shadow-sm shadow-primary/10" : "border-border",
              )}
            >
              <div className="flex items-center gap-2">
                {c.selected ? (
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" />
                ) : (
                  <XCircle className="h-4 w-4 shrink-0 text-muted-foreground" />
                )}
                <div>
                  <p className={cn("text-sm font-semibold", c.selected && "text-primary")}>
                    {c.label}
                    {c.selected && <span className="ml-2 text-xs font-medium text-primary">Selected</span>}
                  </p>
                  {c.rejectionReason && (
                    <p className="mt-0.5 max-w-xl text-xs text-muted-foreground">{c.rejectionReason}</p>
                  )}
                </div>
              </div>
              <div className="flex shrink-0 gap-4 text-sm tabular-nums sm:text-right">
                <div>
                  <p className="text-xs text-muted-foreground">Accuracy</p>
                  <p className="font-semibold">{c.validationAccuracyPct.toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Macro F1</p>
                  <p className="font-semibold">{c.validationMacroF1.toFixed(4)}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </CardContent>
      </Card>

      <Card variant="elevated">
        <CardHeader>
          <CardTitle>Why TF-IDF + Logistic Regression was selected</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="flex flex-col gap-2 text-sm text-muted-foreground">
            {CATEGORIZATION_SELECTION_RATIONALE.map((r, i) => (
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
            <CardTitle>Held-out final result</CardTitle>
            <CardDescription>{CATEGORIZATION_FINAL_RESULT.resultLabel}</CardDescription>
          </div>
          <EvidenceBadge tier={CATEGORIZATION_FINAL_RESULT.evidenceTier} />
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {/* Held-out accuracy is shown alongside macro F1 and sample size,
           * not alone as a hero number — Macro F1 and n are equally
           * prominent, per the "not the hero statistic" display rule. */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Accuracy" value={`${CATEGORIZATION_FINAL_RESULT.accuracyPct.toFixed(1)}%`} />
            <Stat label="Macro F1" value={CATEGORIZATION_FINAL_RESULT.macroF1.toFixed(4)} />
            <Stat label="Rows" value={String(CATEGORIZATION_FINAL_RESULT.nRows)} />
            <Stat label="Merchant groups" value={String(CATEGORIZATION_FINAL_RESULT.nMerchantGroups)} />
          </div>
          <LimitationNote>{CATEGORIZATION_FINAL_RESULT.limitation}</LimitationNote>
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
