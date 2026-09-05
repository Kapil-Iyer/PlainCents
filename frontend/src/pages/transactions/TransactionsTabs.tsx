import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import { cn } from "@/lib/utils";

export type TransactionsTab = "list" | "insights";

const TABS: { id: TransactionsTab; label: string; hint: string }[] = [
  { id: "list", label: "Transactions", hint: "Every transaction, filterable and correctable" },
  { id: "insights", label: "Insights", hint: "Category trends and merchant analysis" },
];

/**
 * Tab bar for the Transactions page.
 *
 * The list and the analytics belong on the same page because they are two
 * views of the same rows — correcting a category in the list changes the
 * charts, and a merchant standing out in the charts is something you then go
 * look up in the list. Splitting them across routes would break that loop;
 * stacking them both on one scroll would bury the list.
 *
 * Implemented with the ARIA tabs pattern by hand rather than pulling in
 * another Radix package for two tabs: roving arrow-key focus, one tab stop,
 * and correct aria-controls/aria-labelledby wiring, in ~40 lines.
 */
export function TransactionsTabs({
  active,
  onChange,
}: {
  active: TransactionsTab;
  onChange: (tab: TransactionsTab) => void;
}) {
  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    event.preventDefault();
    const index = TABS.findIndex((t) => t.id === active);
    const next = event.key === "ArrowRight" ? index + 1 : index - 1;
    onChange(TABS[(next + TABS.length) % TABS.length].id);
  };

  return (
    <div
      role="tablist"
      aria-label="Transactions views"
      onKeyDown={handleKeyDown}
      className="flex gap-1 border-b border-border"
    >
      {TABS.map((tab) => {
        const selected = tab.id === active;
        return (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            type="button"
            role="tab"
            aria-selected={selected}
            aria-controls={`panel-${tab.id}`}
            tabIndex={selected ? 0 : -1}
            title={tab.hint}
            onClick={() => onChange(tab.id)}
            className={cn(
              "relative px-3 py-2 text-sm font-medium transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
              selected ? "text-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {tab.label}
            {selected && (
              <motion.span
                layoutId="transactions-tab-underline"
                className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-primary"
              />
            )}
          </button>
        );
      })}
    </div>
  );
}

/** A tab panel that fades its content in, and does nothing at all when the
 * viewer prefers reduced motion. */
export function TabPanel({
  id,
  active,
  children,
}: {
  id: TransactionsTab;
  active: boolean;
  children: React.ReactNode;
}) {
  const reduceMotion = useReducedMotion();
  return (
    <AnimatePresence mode="wait" initial={false}>
      {active && (
        <motion.div
          key={id}
          id={`panel-${id}`}
          role="tabpanel"
          aria-labelledby={`tab-${id}`}
          initial={reduceMotion ? false : { opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={reduceMotion ? { opacity: 1 } : { opacity: 0 }}
          transition={{ duration: 0.18, ease: "easeOut" }}
        >
          {children}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
