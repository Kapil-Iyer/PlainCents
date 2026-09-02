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
import { useToast } from "@/components/shared/Toast";
import { useCreateHolding, useUpdateHolding } from "@/hooks/useHoldings";
import { ApiError } from "@/types/common";
import type { HoldingResponse } from "@/types/holding";

interface HoldingFormDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Present -> edit mode (shares/avg cost only); absent -> create mode. */
  holding?: HoldingResponse;
}

/** Add/Edit holding (Build Plan Phase 8). Ticker is only settable on create —
 * HoldingRepository.update() never accepts a ticker change, so the edit form
 * doesn't expose a field that would silently do nothing. */
export function HoldingFormDialog({ open, onOpenChange, holding }: HoldingFormDialogProps) {
  const isEdit = !!holding;
  const { toast } = useToast();
  const createMutation = useCreateHolding();
  const updateMutation = useUpdateHolding();

  const [ticker, setTicker] = useState("");
  const [shares, setShares] = useState("");
  const [avgCost, setAvgCost] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setTicker(holding?.ticker ?? "");
      setShares(holding ? String(holding.shares) : "");
      setAvgCost(holding ? String(holding.avg_cost) : "");
      setError(null);
    }
  }, [open, holding]);

  const pending = createMutation.isPending || updateMutation.isPending;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const parsedShares = Number(shares);
    const parsedAvgCost = Number(avgCost);

    if (!isEdit && !ticker.trim()) {
      setError("Ticker is required.");
      return;
    }
    if (!Number.isFinite(parsedShares) || parsedShares <= 0) {
      setError("Shares must be a positive number.");
      return;
    }
    if (!Number.isFinite(parsedAvgCost) || parsedAvgCost < 0) {
      setError("Average cost must be zero or a positive number.");
      return;
    }

    try {
      if (isEdit) {
        await updateMutation.mutateAsync({
          id: holding.id,
          payload: { shares: parsedShares, avg_cost: parsedAvgCost },
        });
        toast({ title: "Holding updated" });
      } else {
        await createMutation.mutateAsync({ ticker, shares: parsedShares, avg_cost: parsedAvgCost });
        toast({ title: "Holding added" });
      }
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>{isEdit ? "Edit holding" : "Add holding"}</DialogTitle>
            <DialogDescription>
              {isEdit
                ? "Update shares or average cost. Prices only change via Refresh Prices."
                : "Enter a ticker, shares held, and your average cost per share."}
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 py-4">
            <div className="grid gap-1.5">
              <Label htmlFor="holding-ticker">Ticker</Label>
              <Input
                id="holding-ticker"
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                placeholder="e.g. AAPL"
                disabled={isEdit}
                required
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="holding-shares">Shares</Label>
              <Input
                id="holding-shares"
                type="number"
                step="any"
                min="0"
                value={shares}
                onChange={(e) => setShares(e.target.value)}
                placeholder="10"
                required
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="holding-avg-cost">Average cost per share</Label>
              <Input
                id="holding-avg-cost"
                type="number"
                step="0.01"
                min="0"
                value={avgCost}
                onChange={(e) => setAvgCost(e.target.value)}
                placeholder="150.00"
                required
              />
            </div>
            {error && <p className="text-sm text-destructive">{error}</p>}
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={pending}>
              Cancel
            </Button>
            <Button type="submit" disabled={pending}>
              {pending && <Loader2 className="h-4 w-4 animate-spin" />}
              {isEdit ? "Save changes" : "Add holding"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
