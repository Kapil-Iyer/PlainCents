import { useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Calculator } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { cn, formatCurrency } from "@/lib/utils";

const PRESETS = [
  { id: "steady", label: "Steady", months: [300, 450, 600] },
  { id: "flat", label: "Flat", months: [420, 410, 430] },
  { id: "spike", label: "One big month", months: [280, 310, 900] },
] as const;

/**
 * The forecast, explained by doing the arithmetic in front of the user.
 *
 * The method is a three-month average. That is genuinely all it is, and the
 * most useful thing this section can do is make that concrete enough that
 * nobody mistakes it for something cleverer — including the "one big month"
 * preset, which shows the method's main weakness rather than hiding it.
 */
export function ForecastExplainerSection() {
  const [preset, setPreset] = useState(0);
  const reduceMotion = useReducedMotion();
  const months = PRESETS[preset].months;
  const forecast = Math.round((months.reduce((a, b) => a + b, 0) / months.length) * 100) / 100;
  const max = Math.max(...months, forecast);

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">How the forecast works</h2>
        <p className="text-sm text-muted-foreground">
          For each category: add up the last three months, divide by three. That&apos;s the whole
          method.
        </p>
      </div>

      <Card>
        <CardContent className="flex flex-col gap-5 pt-6">
          <div role="radiogroup" aria-label="Example history" className="flex flex-wrap gap-2">
            {PRESETS.map((p, i) => (
              <button
                key={p.id}
                type="button"
                role="radio"
                aria-checked={i === preset}
                onClick={() => setPreset(i)}
                className={cn(
                  "rounded-md border px-3 py-1.5 text-xs font-medium transition-colors",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  i === preset
                    ? "border-primary bg-primary/10 text-foreground"
                    : "border-border text-muted-foreground hover:text-foreground",
                )}
              >
                {p.label}
              </button>
            ))}
          </div>

          <div className="flex items-end gap-3 sm:gap-5">
            {months.map((value, i) => (
              <Column
                key={i}
                label={`Month ${i + 1}`}
                value={value}
                heightPct={(value / max) * 100}
                reduceMotion={reduceMotion}
                delay={i * 0.06}
              />
            ))}

            <div className="flex flex-col items-center gap-2 pb-9">
              <Calculator className="h-4 w-4 text-muted-foreground" aria-hidden />
              <span aria-hidden className="text-lg text-muted-foreground">
                →
              </span>
            </div>

            {[1, 2, 3].map((offset) => (
              <Column
                key={offset}
                label={`+${offset}`}
                value={forecast}
                heightPct={(forecast / max) * 100}
                forecast
                reduceMotion={reduceMotion}
                delay={0.2 + offset * 0.06}
              />
            ))}
          </div>

          <p className="rounded-md bg-muted/40 px-3 py-2 text-center font-mono text-xs sm:text-sm">
            ({months.map((m) => formatCurrency(m)).join(" + ")}) ÷ 3 ={" "}
            <span className="font-semibold text-primary">{formatCurrency(forecast)}</span>
          </p>

          <div className="grid grid-cols-1 gap-4 border-t border-border pt-4 sm:grid-cols-3">
            <Note title="Why three months">
              Three is the fewest months this method can run on — it&apos;s exactly one full
              window. Below that there is nothing to average, so PlainCents tells you how many
              months are still needed instead of showing a number.
            </Note>
            <Note title="Why all three months are the same">
              The forecast never feeds itself. Next month, the month after and the one after that
              are all the same average of the same three real months — so PlainCents shows the
              same figure rather than inventing a trend it hasn&apos;t measured.
            </Note>
            <Note title="What it can't do">
              It has no idea about a holiday, a move, or a one-off purchase. The &quot;one big
              month&quot; example above drags the forecast up by roughly a third — that is the
              method being honest about how simple it is.
            </Note>
          </div>

          <p className="text-xs leading-relaxed text-muted-foreground">
            Three months is a mathematical minimum, not a finding that three months forecasts as
            well as six or twelve. That comparison was never run at three months, and PlainCents
            does not claim it.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function Column({
  label,
  value,
  heightPct,
  forecast,
  reduceMotion,
  delay,
}: {
  label: string;
  value: number;
  heightPct: number;
  forecast?: boolean;
  reduceMotion: boolean | null;
  delay: number;
}) {
  return (
    <div className="flex flex-1 flex-col items-center gap-1.5">
      <span className="text-[10px] tabular-nums text-muted-foreground sm:text-xs">
        {formatCurrency(value)}
      </span>
      <div className="flex h-28 w-full items-end">
        <motion.div
          className={cn(
            "w-full rounded-t",
            forecast
              ? "border border-dashed border-primary bg-primary/20"
              : "bg-muted-foreground/40",
          )}
          initial={reduceMotion ? { height: `${heightPct}%` } : { height: 0 }}
          animate={{ height: `${heightPct}%` }}
          transition={{ duration: 0.4, delay: reduceMotion ? 0 : delay, ease: "easeOut" }}
        />
      </div>
      <span className={cn("text-[10px] sm:text-xs", forecast ? "text-primary" : "text-muted-foreground")}>
        {label}
      </span>
    </div>
  );
}

function Note({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <h3 className="text-sm font-semibold">{title}</h3>
      <p className="text-sm leading-relaxed text-muted-foreground">{children}</p>
    </div>
  );
}
