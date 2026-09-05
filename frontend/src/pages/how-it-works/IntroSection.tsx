import { motion, useReducedMotion } from "framer-motion";
import {
  Ban,
  CheckCircle2,
  FileSpreadsheet,
  LineChart,
  PieChart,
  UserCheck,
} from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";

const CAN_DO = [
  {
    icon: FileSpreadsheet,
    title: "Read your bank's own CSV",
    body: "Export a statement from RBC, Scotiabank, TD or CIBC and drop the file in. PlainCents works out which bank it came from and which rows are actual spending.",
  },
  {
    icon: PieChart,
    title: "Sort your spending automatically",
    body: "Every purchase is filed into one of eight categories from its description alone — no manual tagging to get started.",
  },
  {
    icon: UserCheck,
    title: "Learn from you when it's wrong",
    body: "Correct a category once and PlainCents remembers it for that merchant. The next statement you import already has your answer.",
  },
  {
    icon: LineChart,
    title: "Show you where it's going",
    body: "Category trends, your biggest merchants, what changed since last month, and a simple forecast once there's enough history.",
  },
] as const;

const WILL_NOT_DO = [
  "Connect to your bank. You export the file; PlainCents never sees a login.",
  "Track income. Credits and deposits are recognized and skipped — this is a spending tool.",
  "Move money, pay bills, or touch an account in any way.",
  "Give financial advice, or tell you what you should have spent.",
  "Send your data anywhere. It stays in a local database on the machine running the app.",
] as const;

/**
 * The product premise, before any technical detail.
 *
 * Someone landing here should be able to decide within about fifteen seconds
 * whether this tool is for them. The "won't do" column is not a disclaimer
 * section — for a finance app it is genuinely half the answer, and burying
 * it below the methodology would be the wrong order.
 */
export function IntroSection() {
  const reduceMotion = useReducedMotion();

  return (
    <div className="flex flex-col gap-6">
      <motion.div
        initial={reduceMotion ? false : { opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
        className="relative overflow-hidden rounded-xl border border-border bg-card p-6 sm:p-8"
      >
        {/* A quiet CSS gradient wash. A WebGL hero was considered and
         * rejected: it would cost a dependency and a GPU budget to add
         * atmosphere to a page whose job is to be trusted and read. */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-70"
          style={{
            background:
              "radial-gradient(1100px 380px at 8% -20%, hsl(var(--primary) / 0.16), transparent 60%), radial-gradient(800px 300px at 95% 0%, hsl(var(--success) / 0.10), transparent 60%)",
          }}
        />
        <div className="relative flex max-w-3xl flex-col gap-3">
          <p className="text-xs font-semibold uppercase tracking-widest text-primary">
            What is PlainCents?
          </p>
          <h2 className="text-balance text-2xl font-bold tracking-tight sm:text-3xl">
            A spending tracker that reads your bank statements and tells you where the money
            actually went.
          </h2>
          <p className="text-pretty text-sm leading-relaxed text-muted-foreground sm:text-base">
            You export a CSV from your bank. PlainCents figures out the format, keeps the
            purchases, sorts them into categories, and turns them into something you can read.
            When it gets a merchant wrong, you fix it once and it remembers.
          </p>
          <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
            It is built for one person looking at their own money — not a household budget system,
            not an accounting package, and not a bank. Everything below explains exactly how it
            works and, just as importantly, where it falls short.
          </p>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:col-span-2">
          {CAN_DO.map((item, i) => (
            <motion.div
              key={item.title}
              initial={reduceMotion ? false : { opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: reduceMotion ? 0 : 0.06 * i, ease: "easeOut" }}
            >
              <Card className="h-full">
                <CardContent className="flex flex-col gap-2 pt-6">
                  <item.icon className="h-5 w-5 text-primary" aria-hidden />
                  <h3 className="text-sm font-semibold">{item.title}</h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">{item.body}</p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={reduceMotion ? false : { opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: reduceMotion ? 0 : 0.24, ease: "easeOut" }}
        >
          <Card className="h-full border-border">
            <CardContent className="flex flex-col gap-3 pt-6">
              <div className="flex items-center gap-2">
                <Ban className="h-4 w-4 text-muted-foreground" aria-hidden />
                <h3 className="text-sm font-semibold">What it deliberately doesn&apos;t do</h3>
              </div>
              <ul className="flex flex-col gap-2.5">
                {WILL_NOT_DO.map((item) => (
                  <li key={item} className="flex gap-2 text-sm leading-relaxed text-muted-foreground">
                    <CheckCircle2
                      className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground"
                      aria-hidden
                    />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
