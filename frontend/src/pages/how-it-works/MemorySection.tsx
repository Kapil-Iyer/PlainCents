import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowRight, Bot, RotateCcw, UserCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface Beat {
  id: string;
  when: string;
  headline: string;
  body: string;
  raw: string;
  predicted: string;
  confirmed: string | null;
  actor: "system" | "human";
}

/** Fabricated. One merchant, three encounters, three different card numbers. */
const BEATS: Beat[] = [
  {
    id: "first",
    when: "January — first time this merchant appears",
    headline: "PlainCents makes a call",
    body: "It has never seen this merchant before. It reads the description, decides Healthcare, and files it. Nobody has looked at this row yet, and the record says so.",
    raw: "VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY",
    predicted: "Healthcare",
    confirmed: null,
    actor: "system",
  },
  {
    id: "correction",
    when: "January — you disagree",
    headline: "You correct it",
    body: "You buy household goods there, not medication, so you file it under Shopping. Your choice is stored in its own column — the model's answer stays exactly where it was, untouched.",
    raw: "VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY",
    predicted: "Healthcare",
    confirmed: "Shopping",
    actor: "human",
  },
  {
    id: "reuse",
    when: "February — the next statement",
    headline: "Your correction is reused",
    body: "Different card number, different prefix, same merchant. PlainCents matches on the merchant identity underneath the noise and applies your category automatically. It still records what it would have said on its own.",
    raw: "CONTACTLESS INTERAC PURCHASE - 9137 CAREWELL PHARMACY",
    predicted: "Healthcare",
    confirmed: "Shopping",
    actor: "human",
  },
  {
    id: "change",
    when: "March — you change your mind",
    headline: "The most recent decision wins",
    body: "You reclassify it back to Healthcare. From here on that's what future imports use. There is no averaging and no voting — the last thing you said is what holds.",
    raw: "CAREWELL PHARMACY #0284",
    predicted: "Healthcare",
    confirmed: "Healthcare",
    actor: "human",
  },
];

/**
 * The correction lifecycle, as a scrubable timeline.
 *
 * The thing worth showing here is not "corrections are saved" — it's that
 * the system's opinion and yours are stored side by side and neither
 * destroys the other. That is what makes a category auditable months later,
 * and it is invisible in a UI that only ever shows one value per row.
 */
export function MemorySection() {
  const [beat, setBeat] = useState(0);
  const reduceMotion = useReducedMotion();
  const current = BEATS[beat];

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">How PlainCents remembers your corrections</h2>
        <p className="text-sm text-muted-foreground">
          One merchant, four months. Step through to see what changes and what doesn&apos;t.
        </p>
      </div>

      <Card>
        <CardContent className="flex flex-col gap-5 pt-6">
          {/* Timeline rail */}
          <ol className="flex flex-wrap gap-2">
            {BEATS.map((b, i) => (
              <li key={b.id}>
                <button
                  type="button"
                  onClick={() => setBeat(i)}
                  aria-current={i === beat ? "step" : undefined}
                  className={cn(
                    "flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-left text-xs transition-colors",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    i === beat
                      ? "border-primary bg-primary/10"
                      : "border-border text-muted-foreground hover:text-foreground",
                  )}
                >
                  <span
                    className={cn(
                      "flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-semibold",
                      b.actor === "human"
                        ? "bg-success/20 text-success"
                        : "bg-primary/20 text-primary",
                    )}
                  >
                    {b.actor === "human" ? <UserCheck className="h-3 w-3" /> : <Bot className="h-3 w-3" />}
                  </span>
                  {b.when.split(" — ")[0]}
                </button>
              </li>
            ))}
          </ol>

          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={current.id}
              initial={reduceMotion ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduceMotion ? { opacity: 1 } : { opacity: 0, y: -8 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="flex flex-col gap-4"
            >
              <div className="flex flex-col gap-1">
                <p className="text-xs uppercase tracking-wide text-muted-foreground">
                  {current.when}
                </p>
                <h3 className="text-base font-semibold">{current.headline}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">{current.body}</p>
              </div>

              <code className="block break-all rounded bg-muted px-2 py-1.5 font-mono text-xs">
                {current.raw}
              </code>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-[1fr_auto_1fr]">
                <Column
                  icon={Bot}
                  title="System"
                  sublabel="predicted_category"
                  value={current.predicted}
                  tone="system"
                  caption="Never overwritten by your correction."
                />
                <div className="flex items-center justify-center">
                  <ArrowRight className="h-4 w-4 rotate-90 text-muted-foreground sm:rotate-0" aria-hidden />
                </div>
                <Column
                  icon={UserCheck}
                  title="You"
                  sublabel="confirmed_category"
                  value={current.confirmed ?? "not set"}
                  tone={current.confirmed ? "human" : "empty"}
                  caption={
                    current.confirmed
                      ? "Only ever written by a real correction of yours."
                      : "Empty until you actually change something."
                  }
                />
              </div>

              <motion.div
                key={`${current.id}-effective`}
                initial={reduceMotion ? false : { scale: 0.98, opacity: 0.6 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.25, ease: "easeOut" }}
                className="rounded-md border border-border-strong bg-elevated px-3 py-2.5"
              >
                <p className="text-xs text-muted-foreground">
                  What every chart, filter and forecast actually uses
                </p>
                <p className="text-lg font-semibold">{current.confirmed ?? current.predicted}</p>
              </motion.div>
            </motion.div>
          </AnimatePresence>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4">
            <p className="max-w-lg text-xs leading-relaxed text-muted-foreground">
              A category PlainCents assigns itself — including the ones it assigns because it
              can&apos;t read the description — never counts as a correction, so it can never
              teach the system a preference you didn&apos;t express.
            </p>
            <Button variant="outline" size="sm" onClick={() => setBeat(0)}>
              <RotateCcw className="h-4 w-4" />
              Replay
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function Column({
  icon: Icon,
  title,
  sublabel,
  value,
  tone,
  caption,
}: {
  icon: typeof Bot;
  title: string;
  sublabel: string;
  value: string;
  tone: "system" | "human" | "empty";
  caption: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col gap-1.5 rounded-md border px-3 py-2.5",
        tone === "system" && "border-primary/30 bg-primary/5",
        tone === "human" && "border-success/40 bg-success/5",
        tone === "empty" && "border-border bg-muted/30",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-xs font-medium">
          <Icon className="h-3.5 w-3.5" aria-hidden />
          {title}
        </span>
        <Badge variant="outline" className="font-mono text-[10px]">
          {sublabel}
        </Badge>
      </div>
      <p className={cn("text-sm font-semibold", tone === "empty" && "font-normal text-muted-foreground")}>
        {value}
      </p>
      <p className="text-xs leading-relaxed text-muted-foreground">{caption}</p>
    </div>
  );
}
