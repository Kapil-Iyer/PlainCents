import { useMemo, useState, type ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { ArrowRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { CATEGORIES, type Category } from "@/constants/categories";
import { HUMAN_IN_LOOP_FACTS, HUMAN_IN_LOOP_STEPS } from "@/data/methodology";

/**
 * Demo-only illustrative state. This entire section is LOCAL REACT STATE —
 * no API request, no query/mutation, no persistence, no model call. It
 * demonstrates the predicted/confirmed/effective-category mechanism without
 * touching any real transaction, exactly as ML_E_CLAIM_MATRIX.json's
 * "PlainCents preserves user corrections separately from model
 * predictions" (SUPPORTED) / "...automatically retrains..." (NOT_SUPPORTED)
 * claims describe.
 */
const PREDICTED_CATEGORY: Category = "Transport";
const OTHER_CATEGORIES = CATEGORIES.filter((c) => c !== PREDICTED_CATEGORY);

/** Illustrative starting downstream totals — not real spending data. */
const BASE_TOTALS: Record<Category, number> = {
  "Food & Dining": 4,
  Transport: 3,
  "Rent & Utilities": 2,
  Entertainment: 1,
  Healthcare: 1,
  Shopping: 2,
  Subscriptions: 1,
  Other: 1,
};

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}

function DownstreamBars({ totals, highlight }: { totals: Record<Category, number>; highlight: Category[] }) {
  const prefersReducedMotion = useReducedMotion();
  const max = Math.max(...Object.values(totals), 1);
  return (
    <div className="flex flex-col gap-1.5">
      {CATEGORIES.map((cat) => (
        <div key={cat} className="flex items-center gap-2 text-xs">
          <span
            className={cn(
              "w-32 shrink-0 truncate text-muted-foreground",
              highlight.includes(cat) && "font-semibold text-foreground",
            )}
          >
            {cat}
          </span>
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
            <motion.div
              className={cn("h-full rounded-full", highlight.includes(cat) ? "bg-primary" : "bg-muted-foreground/40")}
              animate={{ width: `${(totals[cat] / max) * 100}%` }}
              transition={prefersReducedMotion ? { duration: 0 } : { duration: 0.4, ease: "easeOut" }}
            />
          </div>
          <span
            data-testid={`downstream-count-${cat}`}
            className="w-4 shrink-0 text-right tabular-nums text-muted-foreground"
          >
            {totals[cat]}
          </span>
        </div>
      ))}
    </div>
  );
}

export function HumanInLoopSection() {
  const [confirmed, setConfirmed] = useState<Category | null>(null);
  const effective = confirmed ?? PREDICTED_CATEGORY;

  const totals = useMemo(() => {
    const t = { ...BASE_TOTALS };
    if (confirmed && confirmed !== PREDICTED_CATEGORY) {
      t[PREDICTED_CATEGORY] -= 1;
      t[confirmed] += 1;
    }
    return t;
  }, [confirmed]);

  const correctionStep = HUMAN_IN_LOOP_STEPS.find((s) => s.id === "correction")!;
  const effectiveStep = HUMAN_IN_LOOP_STEPS.find((s) => s.id === "effective")!;

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <CardTitle>Human-in-the-Loop</CardTitle>
          <CardDescription>
            AI proposes a category. You decide. Try it — pick a different category below.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <div className="flex flex-wrap items-center gap-3">
            <Field label="predicted_category">
              <Badge variant="predicted">{PREDICTED_CATEGORY}</Badge>
            </Field>
            <ArrowRight className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
            <Field label="confirmed_category">
              {confirmed ? (
                <Badge variant="confirmed">{confirmed}</Badge>
              ) : (
                <span className="text-xs italic text-muted-foreground">not corrected</span>
              )}
            </Field>
            <ArrowRight className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
            <Field label="effective_category">
              <Badge variant={confirmed ? "confirmed" : "predicted"}>{effective}</Badge>
            </Field>
          </div>

          <div className="flex flex-col gap-1.5 sm:max-w-xs">
            <label htmlFor="hitl-category-picker" className="text-xs text-muted-foreground">
              Correct the prediction
            </label>
            <select
              id="hitl-category-picker"
              aria-label="Correct the predicted category"
              value={confirmed ?? ""}
              onChange={(e) => setConfirmed((e.target.value || null) as Category | null)}
              className="rounded-md border border-border bg-card px-2 py-1.5 text-sm text-foreground"
            >
              <option value="">Keep predicted ({PREDICTED_CATEGORY})</option>
              {OTHER_CATEGORIES.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">{correctionStep.description}</p>
          </div>

          <div>
            <p className="mb-2 text-xs font-medium text-muted-foreground">
              Downstream (e.g. category totals) — reacts to the effective category
            </p>
            <DownstreamBars
              totals={totals}
              highlight={[PREDICTED_CATEGORY, confirmed].filter((c): c is Category => Boolean(c))}
            />
          </div>

          <code className="block rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">
            {effectiveStep.label}
          </code>
        </CardContent>
      </Card>

      <Card variant="elevated">
        <CardHeader>
          <CardTitle>What this means in practice</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="flex flex-col gap-2 text-sm text-muted-foreground">
            {HUMAN_IN_LOOP_FACTS.map((fact, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-primary" aria-hidden>
                  •
                </span>
                {fact}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
