import { FlaskConical, Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";

interface EvidenceBadgeProps {
  tier: "Tier B" | "Synthetic";
  className?: string;
}

/**
 * ML display rules (non-negotiable): the evidence-tier qualifier must be
 * visible on the card itself, never hidden only in a tooltip/hover state or
 * a tiny footnote. This renders inline, at normal text size, everywhere a
 * metric that depends on it is shown.
 */
export function EvidenceBadge({ tier, className }: EvidenceBadgeProps) {
  const Icon = tier === "Tier B" ? FlaskConical : Sparkles;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border border-warning/40 bg-warning/10 px-2.5 py-1 text-xs font-semibold text-warning",
        className,
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {tier} evidence
    </span>
  );
}

export function LimitationNote({ children }: { children: React.ReactNode }) {
  return (
    <p className="rounded-md border border-border bg-muted/40 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
      {children}
    </p>
  );
}
