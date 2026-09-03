import { motion, useReducedMotion } from "framer-motion";
import {
  BrainCircuit,
  LayoutGrid,
  LineChart,
  ShieldCheck,
  UploadCloud,
  UserCheck,
} from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PIPELINE_STEPS } from "@/data/methodology";

const ICONS = [UploadCloud, ShieldCheck, BrainCircuit, UserCheck, LineChart, LayoutGrid];

/** Overview section: `Bank CSV → Normalize → Categorize → Confirm →
 * Forecast → Insights`, the full product pipeline in one diagram. */
export function PipelineDiagram() {
  const prefersReducedMotion = useReducedMotion();

  return (
    <Card variant="elevated">
      <CardHeader>
        <CardTitle>PlainCents in one pipeline</CardTitle>
        <CardDescription>From raw transactions to insights you can act on.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-2 md:flex-row md:items-stretch md:gap-0">
          {PIPELINE_STEPS.map((step, i) => {
            const Icon = ICONS[i] ?? LayoutGrid;
            return (
              <div key={step.id} className="flex flex-1 items-center md:items-stretch">
                <motion.div
                  initial={prefersReducedMotion ? undefined : { opacity: 0, y: 12 }}
                  whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-40px" }}
                  transition={{ duration: 0.35, delay: i * 0.07 }}
                  className="flex flex-1 flex-col items-center gap-2 rounded-lg border border-border bg-card px-3 py-4 text-center"
                >
                  <span className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/15 text-primary">
                    <Icon className="h-5 w-5" />
                  </span>
                  <p className="text-sm font-semibold">{step.label}</p>
                  <p className="text-xs text-muted-foreground">{step.description}</p>
                </motion.div>
                {i < PIPELINE_STEPS.length - 1 && (
                  <motion.div
                    initial={prefersReducedMotion ? undefined : { scaleX: 0, opacity: 0 }}
                    whileInView={prefersReducedMotion ? undefined : { scaleX: 1, opacity: 1 }}
                    viewport={{ once: true, margin: "-40px" }}
                    transition={{ duration: 0.3, delay: i * 0.07 + 0.15 }}
                    className="hidden w-6 shrink-0 origin-left self-center border-t-2 border-dashed border-primary/40 md:block"
                    aria-hidden
                  />
                )}
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
