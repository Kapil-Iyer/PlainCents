import { CheckCircle2, RefreshCw, Sparkles, TrendingUp, UploadCloud, UserCheck, Wallet } from "lucide-react";

import { Badge } from "@/components/ui/badge";

/**
 * Static, hard-coded representative content for the recruiter/product
 * walkthrough (Build Plan Phase 10, authorized addition). This is
 * presentation-only: no API calls, no shared business logic with the real
 * pages, no mutation of app state. Values here are clearly demo-shaped
 * (round numbers, obviously-fictional merchants) so they never read as a
 * real user's financial data. Visual language (badges, colors, spacing)
 * intentionally reuses the same design tokens/components as the real app
 * so the walkthrough stays recognizably consistent with it without
 * duplicating any of its logic.
 */
export interface WalkthroughStage {
  id: string;
  label: string;
  eyebrow: string;
  title: string;
  description: string;
  render: () => React.ReactNode;
}

function Frame({ children }: { children: React.ReactNode }) {
  return <div className="flex flex-col gap-3 p-4">{children}</div>;
}

function MiniRow({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`flex items-center justify-between gap-3 rounded-md border border-border bg-card px-3 py-2 text-sm ${className}`}>
      {children}
    </div>
  );
}

export const WALKTHROUGH_STAGES: WalkthroughStage[] = [
  {
    id: "import",
    label: "Import",
    eyebrow: "01 — Import",
    title: "Bring in a bank CSV",
    description: "Upload a Canadian bank CSV export, preview it before anything is saved, and let the model suggest a category for each row.",
    render: () => (
      <Frame>
        <div className="flex items-center gap-2 rounded-md border-2 border-dashed border-border px-4 py-3 text-sm text-muted-foreground">
          <UploadCloud className="h-4 w-4 shrink-0" />
          statement_export.csv — ready to preview
        </div>
        <div className="overflow-hidden rounded-md border border-border">
          <div className="grid grid-cols-[1fr_auto_auto] gap-2 border-b border-border bg-muted/50 px-3 py-1.5 text-xs font-medium uppercase text-muted-foreground">
            <span>Merchant</span>
            <span>Amount</span>
            <span>Predicted</span>
          </div>
          {[
            ["Sunrise Coffee Co.", "$4.85", "Dining"],
            ["Metro Grocer", "$62.10", "Groceries"],
            ["Riverside Cinema", "$18.00", "Entertainment"],
          ].map(([m, a, c]) => (
            <div key={m} className="grid grid-cols-[1fr_auto_auto] items-center gap-2 border-b border-border px-3 py-1.5 text-sm last:border-0">
              <span className="truncate">{m}</span>
              <span className="tabular-nums text-muted-foreground">{a}</span>
              <Badge variant="predicted" className="justify-self-start">
                <Sparkles className="mr-1 h-3 w-3" />
                {c}
              </Badge>
            </div>
          ))}
        </div>
      </Frame>
    ),
  },
  {
    id: "transactions",
    label: "Transactions",
    eyebrow: "02 — Transactions",
    title: "Review and correct",
    description: "Every transaction shows what the model predicted. Correct one and it's clearly marked as confirmed by you everywhere else.",
    render: () => (
      <Frame>
        <MiniRow>
          <span className="truncate">Sunrise Coffee Co.</span>
          <Badge variant="predicted">
            <Sparkles className="mr-1 h-3 w-3" />
            Dining
          </Badge>
        </MiniRow>
        <MiniRow className="ring-1 ring-primary/40">
          <span className="truncate">Northgate Pharmacy</span>
          <Badge variant="confirmed">
            <UserCheck className="mr-1 h-3 w-3" />
            Health
          </Badge>
        </MiniRow>
        <p className="text-xs text-muted-foreground">
          You corrected this one — the model's original guess is still kept on record, but this
          confirmed category is what counts everywhere else.
        </p>
      </Frame>
    ),
  },
  {
    id: "dashboard",
    label: "Dashboard",
    eyebrow: "03 — Dashboard",
    title: "See spending at a glance",
    description: "A monthly summary, category breakdown, recent trend, and the most recent transactions — all from persisted data.",
    render: () => (
      <Frame>
        <div className="grid grid-cols-3 gap-2">
          {[
            ["Spent", "$2,410"],
            ["Top category", "Groceries"],
            ["vs. last month", "-6%"],
          ].map(([label, value]) => (
            <div key={label} className="rounded-md border border-border bg-card px-2.5 py-2">
              <p className="text-[10px] uppercase text-muted-foreground">{label}</p>
              <p className="text-sm font-semibold tabular-nums">{value}</p>
            </div>
          ))}
        </div>
        <div className="flex items-end gap-1.5 rounded-md border border-border bg-card px-3 py-3">
          {[40, 65, 50, 80, 60, 90].map((h, i) => (
            <div key={i} className="w-4 rounded-sm bg-primary/70" style={{ height: `${h}%` }} />
          ))}
        </div>
      </Frame>
    ),
  },
  {
    id: "forecast",
    label: "Forecast",
    eyebrow: "04 — Forecast",
    title: "Persisted, on-demand forecasts",
    description: "Category-level +1/+2/+3 month forecasts, generated explicitly and marked stale after your data changes — never silently retrained.",
    render: () => (
      <Frame>
        <MiniRow>
          <span className="flex items-center gap-1.5">
            <TrendingUp className="h-3.5 w-3.5 text-primary" />
            Groceries
          </span>
          <span className="tabular-nums text-muted-foreground">$420 · $435 · $410</span>
        </MiniRow>
        <MiniRow>
          <span className="flex items-center gap-1.5">
            <TrendingUp className="h-3.5 w-3.5 text-primary" />
            Dining
          </span>
          <span className="tabular-nums text-muted-foreground">$180 · $175 · $190</span>
        </MiniRow>
        <div className="flex items-center gap-2 rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
          <RefreshCw className="h-3.5 w-3.5 shrink-0" />
          Data changed since this forecast ran — refresh to update it.
        </div>
      </Frame>
    ),
  },
  {
    id: "portfolio",
    label: "Portfolio",
    eyebrow: "05 — Portfolio",
    title: "Holdings, refreshed on request",
    description: "Cached prices and current value/P&L, with a single explicit Refresh Prices action — opening the page never triggers a market lookup.",
    render: () => (
      <Frame>
        <MiniRow>
          <span className="flex items-center gap-1.5">
            <Wallet className="h-3.5 w-3.5 text-primary" />
            8 sh · TICKER-A
          </span>
          <span className="tabular-nums text-success">+$142.00</span>
        </MiniRow>
        <MiniRow>
          <span className="flex items-center gap-1.5">
            <Wallet className="h-3.5 w-3.5 text-primary" />
            20 sh · TICKER-B
          </span>
          <span className="tabular-nums text-destructive">-$38.50</span>
        </MiniRow>
        <div className="flex items-center gap-2 rounded-md border border-border bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
          <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
          Prices last refreshed 2 hours ago — click Refresh Prices for the latest.
        </div>
      </Frame>
    ),
  },
];
