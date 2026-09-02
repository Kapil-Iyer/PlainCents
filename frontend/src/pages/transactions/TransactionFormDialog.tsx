import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CATEGORIES } from "@/constants/categories";
import { useCreateTransaction, useUpdateTransaction } from "@/hooks/useTransactions";
import { useToast } from "@/components/shared/Toast";
import { ApiError } from "@/types/common";
import type { TransactionResponse } from "@/types/transaction";

interface TransactionFormDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Present -> edit mode; absent -> create mode. */
  transaction?: TransactionResponse;
}

const NO_CATEGORY = "__none__";

export function TransactionFormDialog({ open, onOpenChange, transaction }: TransactionFormDialogProps) {
  const isEdit = !!transaction;
  const { toast } = useToast();
  const createMutation = useCreateTransaction();
  const updateMutation = useUpdateTransaction();

  const [date, setDate] = useState("");
  const [merchant, setMerchant] = useState("");
  const [amount, setAmount] = useState("");
  const [category, setCategory] = useState<string>(NO_CATEGORY);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setDate(transaction?.date ?? new Date().toISOString().slice(0, 10));
      setMerchant(transaction?.merchant ?? "");
      setAmount(transaction ? String(transaction.amount) : "");
      setCategory(transaction?.confirmed_category ?? NO_CATEGORY);
      setError(null);
    }
  }, [open, transaction]);

  const pending = createMutation.isPending || updateMutation.isPending;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const parsedAmount = Number(amount);
    if (!merchant.trim()) {
      setError("Merchant is required.");
      return;
    }
    if (Number.isNaN(parsedAmount)) {
      setError("Amount must be a number.");
      return;
    }

    const confirmed_category = category === NO_CATEGORY ? null : category;

    try {
      if (isEdit) {
        await updateMutation.mutateAsync({
          id: transaction.id,
          payload: { date, merchant, amount: parsedAmount, confirmed_category },
        });
        toast({ title: "Transaction updated" });
      } else {
        await createMutation.mutateAsync({ date, merchant, amount: parsedAmount, confirmed_category });
        toast({ title: "Transaction created" });
      }
      onOpenChange(false);
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) {
        setError(
          "The categorization model is unavailable right now, so a new transaction can't be created. Please try again later.",
        );
      } else if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("Something went wrong. Please try again.");
      }
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>{isEdit ? "Edit transaction" : "Add transaction"}</DialogTitle>
            <DialogDescription>
              {isEdit
                ? "Correcting the category here sets it as confirmed and it becomes the effective category everywhere else."
                : "Manually entered transactions are categorized automatically."}
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 py-4">
            <div className="grid gap-1.5">
              <Label htmlFor="txn-date">Date</Label>
              <Input id="txn-date" type="date" value={date} onChange={(e) => setDate(e.target.value)} required />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="txn-merchant">Merchant</Label>
              <Input
                id="txn-merchant"
                value={merchant}
                onChange={(e) => setMerchant(e.target.value)}
                placeholder="e.g. Loblaws"
                required
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="txn-amount">Amount</Label>
              <Input
                id="txn-amount"
                type="number"
                step="0.01"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                placeholder="42.50 (use a negative number for a refund/credit)"
                required
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="txn-category">Category (optional override)</Label>
              <Select value={category} onValueChange={setCategory}>
                <SelectTrigger id="txn-category">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NO_CATEGORY}>Let the model predict</SelectItem>
                  {CATEGORIES.map((c) => (
                    <SelectItem key={c} value={c}>
                      {c}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {error && <p className="text-sm text-destructive">{error}</p>}
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={pending}>
              Cancel
            </Button>
            <Button type="submit" disabled={pending}>
              {pending && <Loader2 className="h-4 w-4 animate-spin" />}
              {isEdit ? "Save changes" : "Add transaction"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
