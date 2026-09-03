import { motion, useReducedMotion } from "framer-motion";
import { ArrowRight } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { HUMAN_IN_LOOP_FACTS, HUMAN_IN_LOOP_STEPS } from "@/data/methodology";

export function HumanInLoopSection() {
  const prefersReducedMotion = useReducedMotion();

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <CardTitle>Human-in-the-Loop</CardTitle>
          <CardDescription>
            Your corrections are authoritative. The model's original guess is never lost.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-3 md:flex-row md:items-stretch">
            {HUMAN_IN_LOOP_STEPS.map((step, i) => (
              <div key={step.id} className="flex flex-1 items-center gap-2 md:flex-col md:items-stretch">
                <motion.div
                  initial={prefersReducedMotion ? undefined : { opacity: 0, y: 10 }}
                  whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-40px" }}
                  transition={{ duration: 0.3, delay: i * 0.08 }}
                  className="flex-1 rounded-lg border border-border bg-card p-3"
                >
                  <code className="block break-words text-xs font-semibold text-primary">{step.label}</code>
                  <p className="mt-1 text-xs text-muted-foreground">{step.description}</p>
                </motion.div>
                {i < HUMAN_IN_LOOP_STEPS.length - 1 && (
                  <ArrowRight className="hidden h-4 w-4 shrink-0 text-muted-foreground md:block md:rotate-0" aria-hidden />
                )}
              </div>
            ))}
          </div>
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
                <span className="text-primary" aria-hidden>•</span>
                {fact}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
