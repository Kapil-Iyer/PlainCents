import { motion, useReducedMotion } from "framer-motion";
import { CheckCircle2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Disclosure } from "@/pages/how-it-works/Disclosure";
import { cn } from "@/lib/utils";
import { MLG_CANDIDATES, MLG_DATASET, MLG_FINAL_TEST } from "@/data/methodology/mlg";

/**
 * The evaluation record.
 *
 * Structured so a normal user gets the honest one-paragraph version at the
 * top and can stop there, while anything research-shaped sits behind
 * disclosures. The headline number is deliberately the sealed test score,
 * not the validation score — validation was used to choose the model, so
 * quoting it as "how good is it" would be quoting the number the model was
 * selected on.
 */
export function MlgEvaluationSection() {
  const reduceMotion = useReducedMotion();
  const improvement = MLG_FINAL_TEST.macroF1WithPolicy / MLG_FINAL_TEST.previousMacroF1;

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">How well does it actually work?</h2>
        <p className="text-sm text-muted-foreground">
          Measured on merchants the model had never seen, on a corpus it was never trained on.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Metric
          value={MLG_FINAL_TEST.macroF1WithPolicy.toFixed(2)}
          label="Macro-F1 on held-out merchants"
          detail={`${MLG_FINAL_TEST.rows} rows across ${MLG_FINAL_TEST.merchantGroups} merchants, none of which appeared in training. Macro-F1 weights every category equally, so a model that only does well on the biggest one scores badly.`}
          reduceMotion={reduceMotion}
          delay={0}
          emphasis
        />
        <Metric
          value={`${improvement.toFixed(1)}×`}
          label="better than the previous version"
          detail={`The categorizer that shipped before this work scored ${MLG_FINAL_TEST.previousMacroF1.toFixed(2)} on its own sealed test, with four of eight categories at zero.`}
          reduceMotion={reduceMotion}
          delay={0.06}
        />
        <Metric
          value={`${MLG_FINAL_TEST.zeroFeatureRatePct}%`}
          label="of rows the model couldn't read at all"
          detail="Previously, descriptions that produced no readable features were silently answered with one fixed category — the cause of everything looking like Food & Dining. That no longer happens on this benchmark."
          reduceMotion={reduceMotion}
          delay={0.12}
        />
      </div>

      <Card>
        <CardContent className="flex flex-col gap-4 pt-6">
          <p className="text-sm leading-relaxed text-muted-foreground">
            Those numbers come from a corpus of{" "}
            <span className="font-medium text-foreground">fabricated</span> Canadian-bank-style
            descriptions — every merchant in it was invented for the benchmark. They describe how
            the model behaves on that corpus, and they are not a claim about how it performs on
            your statements. Your own exports carry no category labels, so no accuracy figure can
            be computed on them at all.
          </p>

          <Disclosure summary="How the evaluation was set up">
            <div className="flex flex-col gap-3 text-sm leading-relaxed text-muted-foreground">
              <p>
                The corpus is split by <span className="text-foreground">merchant</span>, not by
                row. Every transaction belonging to one merchant lands entirely in training,
                entirely in validation, or entirely in the final test — so the test measures
                whether the model can categorize a shop it has never encountered, rather than
                whether it can recall one it has.
              </p>
              <ul className="flex flex-col gap-1.5">
                <li>
                  <strong className="text-foreground">Training</strong> — {MLG_DATASET.trainRows}{" "}
                  rows, {MLG_DATASET.trainGroups} merchants. The only data the model ever sees.
                </li>
                <li>
                  <strong className="text-foreground">Validation</strong> —{" "}
                  {MLG_DATASET.validationRows} rows, {MLG_DATASET.validationGroups} merchants. Used
                  to choose between candidates, which is why its score is not quoted as the result.
                </li>
                <li>
                  <strong className="text-foreground">Final test</strong> —{" "}
                  {MLG_DATASET.finalTestRows} rows, {MLG_DATASET.finalTestGroups} merchants.
                  Evaluated once, after the model and its decision rules were frozen.
                </li>
              </ul>
              <p>
                The corpus itself was rebuilt during this work, from {MLG_DATASET.previousRows}{" "}
                rows / {MLG_DATASET.previousGroups} merchants to {MLG_DATASET.trainRows +
                  MLG_DATASET.validationRows +
                  MLG_DATASET.finalTestRows}{" "}
                / {MLG_DATASET.trainGroups + MLG_DATASET.validationGroups + MLG_DATASET.finalTestGroups}.
                The old one gave each descriptive word to exactly one merchant, so a held-out
                merchant shared nothing at all with training and generalization was impossible by
                construction — no amount of model tuning could have fixed that.
              </p>
            </div>
          </Disclosure>

          <Disclosure summary={`Every configuration tried (${MLG_CANDIDATES.length})`}>
            <div className="flex flex-col gap-3">
              {MLG_CANDIDATES.map((candidate) => (
                <div
                  key={candidate.id}
                  className={cn(
                    "rounded-md border px-3 py-2.5",
                    candidate.selected ? "border-success/40 bg-success/5" : "border-border",
                  )}
                >
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <span className="flex items-center gap-2 text-sm font-medium">
                      {candidate.selected && (
                        <CheckCircle2 className="h-4 w-4 text-success" aria-hidden />
                      )}
                      {candidate.label}
                    </span>
                    <span className="flex items-center gap-2 text-xs tabular-nums text-muted-foreground">
                      <Badge variant="outline">F1 {candidate.validationMacroF1.toFixed(3)}</Badge>
                      {candidate.zeroFeatureRatePct > 0 && (
                        <Badge variant="warning">
                          {candidate.zeroFeatureRatePct}% unreadable
                        </Badge>
                      )}
                    </span>
                  </div>
                  <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
                    <span className="font-medium text-foreground">Why try it: </span>
                    {candidate.hypothesis}
                  </p>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                    <span className="font-medium text-foreground">Result: </span>
                    {candidate.outcome}
                  </p>
                </div>
              ))}
            </div>
          </Disclosure>

          <Disclosure summary="Per-category results on the sealed test">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[420px] text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
                    <th scope="col" className="py-2 pr-3">Category</th>
                    <th scope="col" className="py-2 pr-3 text-right">Precision</th>
                    <th scope="col" className="py-2 pr-3 text-right">Recall</th>
                    <th scope="col" className="py-2 pr-3 text-right">F1</th>
                    <th scope="col" className="py-2 text-right">Rows</th>
                  </tr>
                </thead>
                <tbody>
                  {MLG_FINAL_TEST.perCategory.map((row) => (
                    <tr key={row.category} className="border-b border-border last:border-0">
                      <td className="py-1.5 pr-3">{row.category}</td>
                      <td className="py-1.5 pr-3 text-right tabular-nums">{row.precision.toFixed(2)}</td>
                      <td className="py-1.5 pr-3 text-right tabular-nums">{row.recall.toFixed(2)}</td>
                      <td className="py-1.5 pr-3 text-right tabular-nums">{row.f1.toFixed(2)}</td>
                      <td className="py-1.5 text-right tabular-nums text-muted-foreground">
                        {row.support}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
              The spread is the interesting part. Subscriptions and Entertainment have perfect
              precision but poor recall — when the model does commit to them it is right, but it
              often declines to. Other has the opposite shape, because every row the system
              declines to answer lands there, which is exactly the trade the abstention rule makes
              on purpose.
            </p>
          </Disclosure>
        </CardContent>
      </Card>
    </div>
  );
}

function Metric({
  value,
  label,
  detail,
  reduceMotion,
  delay,
  emphasis,
}: {
  value: string;
  label: string;
  detail: string;
  reduceMotion: boolean | null;
  delay: number;
  emphasis?: boolean;
}) {
  return (
    <motion.div
      initial={reduceMotion ? false : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: reduceMotion ? 0 : delay, ease: "easeOut" }}
    >
      <Card className={cn("h-full", emphasis && "border-primary/40")}>
        <CardContent className="flex flex-col gap-1.5 pt-6">
          <p
            className={cn(
              "text-3xl font-bold tabular-nums tracking-tight",
              emphasis && "text-primary",
            )}
          >
            {value}
          </p>
          <p className="text-sm font-medium">{label}</p>
          <p className="text-xs leading-relaxed text-muted-foreground">{detail}</p>
        </CardContent>
      </Card>
    </motion.div>
  );
}
