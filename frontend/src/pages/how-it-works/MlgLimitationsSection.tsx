import { motion, useReducedMotion } from "framer-motion";
import { AlertCircle } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { MLG_LIMITATIONS } from "@/data/methodology/mlg";
import {
  CATEGORIZATION_EVIDENCE_QUALIFIER,
  NOT_SUPPORTED_CLAIMS,
} from "@/data/methodology/claims";

/**
 * The limitations, stated plainly and not tucked behind a disclosure.
 *
 * Every item here is something the evidence genuinely cannot rule out. This
 * section is placed last because it is the conclusion of everything above,
 * not because it is the fine print — a user who reads only this page's
 * headline numbers and stops would come away with a more confident picture
 * than the evidence supports, and this is where that gets corrected.
 */
export function MlgLimitationsSection() {
  const reduceMotion = useReducedMotion();

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">What PlainCents can&apos;t tell you</h2>
        <p className="text-sm text-muted-foreground">
          The honest limits of everything described above.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {MLG_LIMITATIONS.map((limitation, i) => (
          <motion.div
            key={limitation.title}
            initial={reduceMotion ? false : { opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: reduceMotion ? 0 : i * 0.04, ease: "easeOut" }}
          >
            <Card className="h-full">
              <CardContent className="flex gap-3 pt-6">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-warning" aria-hidden />
                <div className="flex flex-col gap-1">
                  <h3 className="text-sm font-semibold">{limitation.title}</h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {limitation.body}
                  </p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Claims PlainCents does not make. Rendered, not hidden behind a
       * disclosure: each of these is a sentence a reader could reasonably
       * infer from the numbers elsewhere on this page, which is exactly why
       * it needs saying out loud. */}
      <Card>
        <CardContent className="flex flex-col gap-3 pt-6">
          <h3 className="text-sm font-semibold">Things PlainCents does not claim</h3>
          <ul className="flex flex-col gap-2">
            {NOT_SUPPORTED_CLAIMS.map((claim) => (
              <li key={claim} className="flex gap-2 text-sm text-muted-foreground">
                <span aria-hidden className="mt-0.5 shrink-0 font-mono text-destructive">
                  ✕
                </span>
                <span className="leading-relaxed line-through decoration-destructive/40">
                  {claim}
                </span>
              </li>
            ))}
          </ul>
          <p className="border-t border-border pt-3 text-xs leading-relaxed text-muted-foreground">
            {CATEGORIZATION_EVIDENCE_QUALIFIER}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
