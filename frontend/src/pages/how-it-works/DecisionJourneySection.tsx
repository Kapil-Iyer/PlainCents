import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Bot, Cpu, ScanLine, ShieldQuestion, UserCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { MLG_ABSTENTION, MLG_AMBIGUITY_ROUTING } from "@/data/methodology/mlg";

type Outcome = "model" | "structural" | "lowconf" | "remembered";

interface Example {
  id: string;
  raw: string;
  normalized: string;
  outcome: Outcome;
  predicted: string;
  confirmed: string | null;
  note: string;
}

/**
 * Fabricated examples. None of these is a real transaction from anyone's
 * statement — they are the four shapes the pipeline treats differently,
 * written to be recognisable rather than real.
 */
const EXAMPLES: Example[] = [
  {
    id: "clear",
    raw: "VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY",
    normalized: "CAREWELL PHARMACY",
    outcome: "model",
    predicted: "Healthcare",
    confirmed: null,
    note: "The card-rail prefix and the card number are stripped, leaving a name the model can read. PHARMACY is a word it has seen attached to many different pharmacies, so it generalizes to one it has never seen.",
  },
  {
    id: "transfer",
    raw: "E-TRANSFER SENT",
    normalized: "(nothing left)",
    outcome: "structural",
    predicted: "Other",
    confirmed: null,
    note: "Strip the boilerplate and there is nothing underneath. This row could be rent, a gift or a repayment — the description simply doesn't say. It's routed to Other before the model is ever asked, because guessing here isn't classification, it's invention.",
  },
  {
    id: "unknown",
    raw: "OPOS ZENOVARA+7712",
    normalized: "ZENOVARA",
    outcome: "lowconf",
    predicted: "Other",
    confirmed: null,
    note: "A brand name with no descriptive word in it. The model produces an answer, but its top two categories are nearly tied — so instead of serving a coin flip dressed up as a decision, the system says Other and leaves it to you.",
  },
  {
    id: "remembered",
    raw: "VISA DEBIT PURCHASE - 9137 CAREWELL PHARMACY",
    normalized: "CAREWELL PHARMACY",
    outcome: "remembered",
    predicted: "Healthcare",
    confirmed: "Shopping",
    note: "You previously filed this pharmacy under Shopping. Different card number, same merchant — your category is applied automatically, and the model's own answer is kept alongside it rather than overwritten.",
  },
];

const OUTCOME_LABEL: Record<Outcome, string> = {
  model: "The model decided",
  structural: "Nothing to categorize",
  lowconf: "Not confident enough",
  remembered: "Your category, remembered",
};

/**
 * How one transaction becomes a category.
 *
 * The visual point of this section is the SYSTEM / HUMAN split: the two
 * columns at the end are different colours and different labels because they
 * are genuinely different things stored in different places. A user should
 * come away understanding that PlainCents never overwrites its own
 * prediction with your correction — it keeps both, which is what makes the
 * record auditable.
 */
export function DecisionJourneySection() {
  const [selected, setSelected] = useState(0);
  const reduceMotion = useReducedMotion();
  const example = EXAMPLES[selected];

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">How a transaction gets its category</h2>
        <p className="text-sm text-muted-foreground">
          Pick a row shape to follow it through the pipeline. All four are made-up examples.
        </p>
      </div>

      <div role="tablist" aria-label="Example transactions" className="flex flex-wrap gap-2">
        {EXAMPLES.map((ex, i) => (
          <button
            key={ex.id}
            type="button"
            role="tab"
            aria-selected={i === selected}
            onClick={() => setSelected(i)}
            className={cn(
              "rounded-md border px-3 py-1.5 text-xs font-medium transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              i === selected
                ? "border-primary bg-primary/10 text-foreground"
                : "border-border text-muted-foreground hover:text-foreground",
            )}
          >
            {OUTCOME_LABEL[ex.outcome]}
          </button>
        ))}
      </div>

      <Card>
        <CardContent className="flex flex-col gap-4 pt-6">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={example.id}
              initial={reduceMotion ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduceMotion ? { opacity: 1 } : { opacity: 0, y: -8 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="flex flex-col gap-4"
            >
              <Stage
                index={1}
                icon={ScanLine}
                title="What the bank sent"
                delay={0}
                reduceMotion={reduceMotion}
              >
                <code className="block break-all rounded bg-muted px-2 py-1.5 font-mono text-xs">
                  {example.raw}
                </code>
              </Stage>

              <Stage
                index={2}
                icon={Cpu}
                title="Stripped down to the merchant"
                delay={0.06}
                reduceMotion={reduceMotion}
              >
                <code className="block break-all rounded bg-muted px-2 py-1.5 font-mono text-xs">
                  {example.normalized}
                </code>
                <p className="text-xs text-muted-foreground">
                  Card-rail prefixes, card numbers, store codes and reference numbers are removed.
                  What&apos;s left is the merchant identity — and only ever a copy: your
                  transaction list still shows exactly what the bank sent.
                </p>
              </Stage>

              <Stage
                index={3}
                icon={example.outcome === "structural" ? ShieldQuestion : Bot}
                title={
                  example.outcome === "structural"
                    ? "Checked for anything worth classifying"
                    : "Handed to the classifier"
                }
                delay={0.12}
                reduceMotion={reduceMotion}
              >
                <p className="text-sm leading-relaxed text-muted-foreground">{example.note}</p>
              </Stage>

              <Stage
                index={4}
                icon={UserCheck}
                title="Two columns, kept separate"
                delay={0.18}
                reduceMotion={reduceMotion}
              >
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <ValueCard
                    label="What the system decided"
                    sublabel="predicted_category"
                    value={example.predicted}
                    tone="system"
                  />
                  <ValueCard
                    label="What you decided"
                    sublabel="confirmed_category"
                    value={example.confirmed ?? "— you haven't touched this row"}
                    tone={example.confirmed ? "human" : "empty"}
                  />
                </div>
                <div className="rounded-md border border-border-strong bg-elevated px-3 py-2">
                  <p className="text-xs text-muted-foreground">
                    Counted in every chart and forecast as
                  </p>
                  <p className="text-sm font-semibold">
                    {example.confirmed ?? example.predicted}
                  </p>
                </div>
              </Stage>
            </motion.div>
          </AnimatePresence>

          <div className="grid grid-cols-1 gap-3 border-t border-border pt-4 sm:grid-cols-2">
            <Fact
              value={`${MLG_AMBIGUITY_ROUTING.coveragePct}%`}
              label="of no-name rows caught by the structural check"
              detail={`with ${MLG_AMBIGUITY_ROUTING.falsePositiveRatePct}% false positives — down from ${MLG_AMBIGUITY_ROUTING.previousFalsePositiveRatePct}%, when the old rule was also swallowing legitimate transfers that did name a merchant`}
            />
            <Fact
              value={`${MLG_ABSTENTION.wrongRescued} vs ${MLG_ABSTENTION.correctCost}`}
              label="wrong answers avoided vs. right answers given up"
              detail={`on held-out data, by declining to answer on the ${MLG_ABSTENTION.abstainRatePct}% of rows where the top two categories were nearly tied`}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function Stage({
  index,
  icon: Icon,
  title,
  children,
  delay,
  reduceMotion,
}: {
  index: number;
  icon: typeof Bot;
  title: string;
  children: React.ReactNode;
  delay: number;
  reduceMotion: boolean | null;
}) {
  return (
    <motion.div
      initial={reduceMotion ? false : { opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.25, delay: reduceMotion ? 0 : delay, ease: "easeOut" }}
      className="flex gap-3"
    >
      <div className="flex flex-col items-center gap-1">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/15 text-xs font-semibold text-primary">
          {index}
        </span>
        {index < 4 && <span aria-hidden className="w-px flex-1 bg-border" />}
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-2 pb-1">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted-foreground" aria-hidden />
          <h3 className="text-sm font-semibold">{title}</h3>
        </div>
        {children}
      </div>
    </motion.div>
  );
}

function ValueCard({
  label,
  sublabel,
  value,
  tone,
}: {
  label: string;
  sublabel: string;
  value: string;
  tone: "system" | "human" | "empty";
}) {
  return (
    <div
      className={cn(
        "rounded-md border px-3 py-2",
        tone === "system" && "border-primary/30 bg-primary/5",
        tone === "human" && "border-success/40 bg-success/5",
        tone === "empty" && "border-border bg-muted/30",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-medium">{label}</p>
        <Badge variant="outline" className="font-mono text-[10px]">
          {sublabel}
        </Badge>
      </div>
      <p
        className={cn(
          "mt-1 text-sm font-semibold",
          tone === "empty" && "text-sm font-normal text-muted-foreground",
        )}
      >
        {value}
      </p>
    </div>
  );
}

function Fact({ value, label, detail }: { value: string; label: string; detail: string }) {
  return (
    <div className="rounded-md bg-muted/40 px-3 py-2.5">
      <p className="text-lg font-semibold tabular-nums">{value}</p>
      <p className="text-xs font-medium">{label}</p>
      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{detail}</p>
    </div>
  );
}
