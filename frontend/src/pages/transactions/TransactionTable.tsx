import { useState } from "react";
import { Pencil, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/shared/ConfirmDialog";
import { useToast } from "@/components/shared/Toast";
import { useDeleteTransaction } from "@/hooks/useTransactions";
import { formatCurrency, formatDate } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { TransactionResponse } from "@/types/transaction";

import { CategoryBadge } from "@/pages/transactions/CategoryBadge";
import { TransactionFormDialog } from "@/pages/transactions/TransactionFormDialog";

interface TransactionTableProps {
  transactions: TransactionResponse[];
}

export function TransactionTable({ transactions }: TransactionTableProps) {
  const [editing, setEditing] = useState<TransactionResponse | null>(null);
  const [deleting, setDeleting] = useState<TransactionResponse | null>(null);
  const deleteMutation = useDeleteTransaction();
  const { toast } = useToast();

  const handleDelete = async () => {
    if (!deleting) return;
    await deleteMutation.mutateAsync(deleting.id);
    toast({ title: "Transaction deleted" });
  };

  return (
    <>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="border-b border-border bg-muted/50 text-left text-xs font-medium uppercase tracking-wide text-muted-foreground">
              <th className="px-4 py-2.5">Date</th>
              <th className="px-4 py-2.5">Merchant</th>
              <th className="px-4 py-2.5">Category</th>
              <th className="px-4 py-2.5 text-right">Amount</th>
              <th className="px-4 py-2.5 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {transactions.map((txn) => (
              <tr key={txn.id} className="border-b border-border last:border-0 hover:bg-muted/30">
                <td className="whitespace-nowrap px-4 py-2.5 text-muted-foreground">{formatDate(txn.date)}</td>
                <td className="px-4 py-2.5 font-medium">{txn.merchant}</td>
                <td className="px-4 py-2.5">
                  <CategoryBadge transaction={txn} />
                </td>
                <td
                  className={cn(
                    "whitespace-nowrap px-4 py-2.5 text-right font-medium tabular-nums",
                    // TD's convention (pipeline/ingest.py): a positive amount
                    // is a spend/withdrawal. A negative amount is a refund/
                    // credit back to the account, so it's the one worth
                    // highlighting distinctly.
                    txn.amount < 0 ? "text-success" : "text-foreground",
                  )}
                >
                  {formatCurrency(txn.amount)}
                </td>
                <td className="px-4 py-2.5">
                  <div className="flex justify-end gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label={`Edit ${txn.merchant}`}
                      onClick={() => setEditing(txn)}
                    >
                      <Pencil className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label={`Delete ${txn.merchant}`}
                      onClick={() => setDeleting(txn)}
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
        <TransactionFormDialog
          open={!!editing}
          onOpenChange={(open) => !open && setEditing(null)}
          transaction={editing}
        />
      )}

      <ConfirmDialog
        open={!!deleting}
        onOpenChange={(open) => !open && setDeleting(null)}
        title="Delete this transaction?"
        description={
          deleting
            ? `This will permanently delete "${deleting.merchant}" (${formatCurrency(deleting.amount)}). This can't be undone.`
            : undefined
        }
        confirmLabel="Delete"
        onConfirm={handleDelete}
      />
    </>
  );
}
