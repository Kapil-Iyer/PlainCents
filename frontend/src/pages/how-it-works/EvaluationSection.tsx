import { motion, useReducedMotion } from "framer-motion";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  MERCHANT_ISOLATION_EXPLANATION,
  SEALED_FINAL_TEST_DISCIPLINE,
  SPLIT_ROLES,
  TEMPORAL_VALIDATION_EXPLANATION,
} from "@/data/methodology";

export function EvaluationSection() {
  const prefersReducedMotion = useReducedMotion();

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <CardTitle>Evaluation Methodology</CardTitle>
          <CardDescription>How every number on this page was actually produced.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {SPLIT_ROLES.map((role, i) => (
              <motion.div
                key={role.id}
                initial={prefersReducedMotion ? undefined : { opacity: 0, y: 10 }}
                whileInView={prefersReducedMotion ? undefined : { opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-40px" }}
                transition={{ duration: 0.3, delay: i * 0.08 }}
                className="rounded-lg border border-border p-4"
              >
                <p className="text-sm font-semibold text-primary">{role.label}</p>
                <p className="mt-1 text-xs text-muted-foreground">{role.description}</p>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Merchant-group isolation</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{MERCHANT_ISOLATION_EXPLANATION}</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Sealed final-test discipline</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{SEALED_FINAL_TEST_DISCIPLINE}</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Temporal expanding-window validation (forecasting)</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{TEMPORAL_VALIDATION_EXPLANATION}</p>
        </CardContent>
      </Card>
    </div>
  );
}
