import { useState } from "react";
import { Pencil, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/shared/ConfirmDialog";
import { useToast } from "@/components/shared/Toast";
import { useDeleteHolding } from "@/hooks/useHoldings";
import { cn, formatCurrency } from "@/lib/utils";
import type { HoldingResponse } from "@/types/holding";

import { HoldingFormDialog } from "@/pages/portfolio/HoldingFormDialog";
import { PriceStatus } from "@/pages/portfolio/PriceStatus";

interface HoldingsTableProps {
  holdings: HoldingResponse[];
}

export function HoldingsTable({ holdings }: HoldingsTableProps) {
  const [editing, setEditing] = useState<HoldingResponse | null>(null);
  const [deleting, setDeleting] = useState<HoldingResponse | null>(null);
  const deleteMutation = useDeleteHolding();
  const { toast } = useToast();

  const handleDelete = async () => {
    if (!deleting) return;
    await deleteMutation.mutateAsync(deleting.id);
    toast({ title: "Holding deleted" });
  };

  return (
    <>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="border-b border-border bg-muted/50 text-left text-xs font-medium uppercase tracking-wide text-muted-foreground">
              <th className="px-4 py-2.5">Ticker</th>
              <th className="px-4 py-2.5 text-right">Shares</th>
              <th className="px-4 py-2.5 text-right">Avg cost</th>
              <th className="px-4 py-2.5 text-right">Current price</th>
              <th className="px-4 py-2.5 text-right">Value</th>
              <th className="px-4 py-2.5 text-right">P&amp;L</th>
              <th className="px-4 py-2.5 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {holdings.map((holding) => (
              <tr key={holding.id} className="border-b border-border last:border-0 hover:bg-muted/30">
                <td className="px-4 py-2.5 font-medium">{holding.ticker}</td>
                <td className="whitespace-nowrap px-4 py-2.5 text-right tabular-nums">{holding.shares}</td>
                <td className="whitespace-nowrap px-4 py-2.5 text-right tabular-nums">
                  {holding.avg_cost === null ? (
                    <button
                      type="button"
                      onClick={() => setEditing(holding)}
                      className="text-muted-foreground underline decoration-dotted underline-offset-2 hover:text-foreground"
                    >
                      Add cost basis
                    </button>
                  ) : (
                    formatCurrency(holding.avg_cost)
                  )}
                </td>
                <td className="whitespace-nowrap px-4 py-2.5 text-right">
                  <PriceStatus holding={holding} />
                </td>
                <td className="whitespace-nowrap px-4 py-2.5 text-right tabular-nums">
                  {holding.current_value === null ? (
                    <span className="text-muted-foreground">—</span>
                  ) : (
                    formatCurrency(holding.current_value)
                  )}
                </td>
                <td
                  className={cn(
                    "whitespace-nowrap px-4 py-2.5 text-right font-medium tabular-nums",
                    holding.pnl === null
                      ? "text-muted-foreground"
                      : holding.pnl >= 0
                        ? "text-success"
                        : "text-destructive",
                  )}
                >
                  {holding.pnl === null ? "—" : formatCurrency(holding.pnl)}
                </td>
                <td className="px-4 py-2.5">
                  <div className="flex justify-end gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label={`Edit ${holding.ticker}`}
                      onClick={() => setEditing(holding)}
                    >
                      <Pencil className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label={`Delete ${holding.ticker}`}
                      onClick={() => setDeleting(holding)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {editing && (
        <HoldingFormDialog
          open={!!editing}
          onOpenChange={(open) => !open && setEditing(null)}
          holding={editing}
        />
      )}

      <ConfirmDialog
        open={!!deleting}
        onOpenChange={(open) => !open && setDeleting(null)}
        title="Delete this holding?"
        description={
          deleting ? `This will permanently delete "${deleting.ticker}". This can't be undone.` : undefined
        }
        confirmLabel="Delete"
        onConfirm={handleDelete}
      />
    </>
  );
}
