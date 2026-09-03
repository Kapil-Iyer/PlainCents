import { useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";

import { cn } from "@/lib/utils";

/**
 * Progressive-disclosure wrapper for secondary methodology detail (full
 * rationale paragraphs, full split-role prose). Built on native
 * <details>/<summary> — keyboard- and screen-reader-accessible with no
 * extra ARIA wiring, and the content is a normal (collapsed) DOM node, not
 * something conjured only by a hover state.
 *
 * Non-negotiable ML display rule: this must never be the only place an
 * evidence-tier qualification or a NOT_SUPPORTED claim is visible — those
 * stay in the always-rendered part of each section. Disclosure is only for
 * supporting detail (full rationale bullets, full methodology prose).
 */
export function Disclosure({
  summary,
  children,
  defaultOpen = false,
  className,
}: {
  summary: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
      className={cn("group rounded-lg border border-border bg-card", className)}
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-4 py-3 text-sm font-medium marker:content-none">
        {summary}
        <ChevronDown
          className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-180"
          aria-hidden
        />
      </summary>
      <div className="flex flex-col gap-2 px-4 pb-4 text-sm leading-relaxed text-muted-foreground">
        {children}
      </div>
    </details>
  );
}
