import { useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { ArrowRight, CheckCircle2, XCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  CATEGORIZATION_CANDIDATES,
  CATEGORIZATION_FINAL_RESULT,
  CATEGORIZATION_PIPELINE_EXPLANATION,
  CATEGORIZATION_SELECTION_RATIONALE,
  type CategorizationCandidate,
} from "@/data/methodology";

import { Disclosure } from "@/pages/how-it-works/Disclosure";
import { EvidenceBadge, LimitationNote } from "@/pages/how-it-works/EvidenceBadge";

/** Macro F1's true axis is 0-1. The domain below is only a rendering
 * choice (headroom above the highest evaluated candidate, 0.2552) — it does
 * not imply an unevaluated benchmark or reference line. Every bar also
 * prints its exact value, so the domain choice can't misrepresent a number. */
const F1_AXIS_MAX = 0.3;

function CandidateBar({ candidate, index }: { candidate: CategorizationCandidate; index: number }) {
  const prefersReducedMotion = useReducedMotion();
  const [expanded, setExpanded] = useState(false);
  const widthPct = Math.min(100, (candidate.validationMacroF1 / F1_AXIS_MAX) * 100);

  return (
    <div
      className={cn(
        "flex flex-col gap-2 rounded-lg border p-3",
        candidate.selected ? "border-primary/60 bg-primary/5" : "border-border",
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          {candidate.selected ? (
            <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" aria-hidden />
          ) : (
            <XCircle className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
          )}
          <span className={cn("text-sm font-semibold", candidate.selected && "text-primary")}>
            {candidate.label}
          </span>
          {candidate.selected && <span className="text-xs font-medium text-primary">Selected</span>}
        </div>
        <div className="flex gap-3 text-xs tabular-nums text-muted-foreground">
          <span>
            Macro F1 <strong className="font-semibold text-foreground">{candidate.validationMacroF1.toFixed(4)}</strong>
          </span>
          <span>
            Accuracy <strong className="font-semibold text-foreground">{candidate.validationAccuracyPct.toFixed(1)}%</strong>
          </span>
        </div>
      </div>

      <div
        className="h-3 w-full overflow-hidden rounded-full bg-muted"
        role="img"
        aria-label={`${candidate.label}: validation macro F1 ${candidate.validationMacroF1}, accuracy ${candidate.validationAccuracyPct}%`}
      >
        <motion.div
          className={cn("h-full rounded-full", candidate.selected ? "bg-primary" : "bg-muted-foreground/50")}
          initial={prefersReducedMotion ? false : { width: 0 }}
          animate={{ width: `${widthPct}%` }}
          transition={{ duration: 0.6, delay: prefersReducedMotion ? 0 : index * 0.1, ease: "easeOut" }}
        />
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

/** Illustrative merchant-text -> feature -> prediction strip. Explicitly
 * NOT measured feature weights — the frozen evidence doesn't expose
 * per-token TF-IDF weights, so this is presented as an explanatory example
 * only, clearly labeled as such. */
function TextFlow() {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <code className="rounded bg-muted px-2 py-1 text-muted-foreground">coffee house 482</code>
        <ArrowRight className="h-3 w-3 shrink-0 text-muted-foreground" aria-hidden />
        <span className="flex gap-1">
          <span className="rounded bg-primary/15 px-1.5 py-0.5 text-primary">coffee</span>
          <span className="rounded bg-primary/15 px-1.5 py-0.5 text-primary">house</span>
          <span className="rounded bg-muted px-1.5 py-0.5 text-muted-foreground">482</span>
        </span>
        <ArrowRight className="h-3 w-3 shrink-0 text-muted-foreground" aria-hidden />
        <code className="rounded bg-muted px-2 py-1 text-muted-foreground">TF-IDF vector</code>
        <ArrowRight className="h-3 w-3 shrink-0 text-muted-foreground" aria-hidden />
        <Badge variant="predicted">Food &amp; Dining</Badge>
      </div>
      <p className="text-xs text-muted-foreground">
        Illustrative example — not measured feature weights or a dataset row.
      </p>
    </div>
  );
}

export function CategorizationSection() {
  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <CardTitle>Transaction Categorization</CardTitle>
          <CardDescription>{CATEGORIZATION_PIPELINE_EXPLANATION}</CardDescription>
        </CardHeader>
        <CardContent>
          <TextFlow />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Candidates benchmarked on VALIDATION</CardTitle>
          <CardDescription>
            All three candidates, same frozen TRAIN/VALIDATION split. Bar length is macro F1 (the
            primary selection metric) — exact values are printed, never only implied by the bar.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {CATEGORIZATION_CANDIDATES.map((c, i) => (
            <CandidateBar key={c.id} candidate={c} index={i} />
          ))}
        </CardContent>
      </Card>

      <Disclosure summary="Why TF-IDF + Logistic Regression was selected" defaultOpen>
        <ul className="flex flex-col gap-2">
          {CATEGORIZATION_SELECTION_RATIONALE.map((r, i) => (
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
            <CardTitle>Held-out final result</CardTitle>
            <CardDescription>{CATEGORIZATION_FINAL_RESULT.resultLabel}</CardDescription>
          </div>
          <EvidenceBadge tier={CATEGORIZATION_FINAL_RESULT.evidenceTier} />
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
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
