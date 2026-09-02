import { useState } from "react";
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
import { useClearDemo } from "@/hooks/useImport";
import { ApiError } from "@/types/common";

interface DemoConflictDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Retried after the (attempted) demo clear, regardless of whether the
   * clear itself actually succeeded — see the 501 note below. */
  onRetry: () => void;
}

/**
 * TRD §5.2: a real import while mode === 'DEMO' returns 409 demo_conflict.
 * The sanctioned flow is DELETE /api/demo/clear then retry the import — but
 * that endpoint is still a Phase-9 501 stub (Build Plan §2.5), so this
 * dialog explains that plainly rather than claiming the clear worked.
 */
export function DemoConflictDialog({ open, onOpenChange, onRetry }: DemoConflictDialogProps) {
  const clearDemoMutation = useClearDemo();
  const [notice, setNotice] = useState<string | null>(null);

  const handleClearAndRetry = async () => {
    setNotice(null);
    try {
      await clearDemoMutation.mutateAsync();
      onOpenChange(false);
      onRetry();
    } catch (err) {
      if (err instanceof ApiError && err.status === 501) {
        setNotice(
          "Clearing demo data isn't implemented yet (it lands in a later phase). You can't import real data while demo data is loaded.",
        );
      } else {
        setNotice("Couldn't clear demo data. Please try again.");
      }
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Demo data is currently loaded</DialogTitle>
          <DialogDescription>
            You can't import real transactions while demo data is active. Clear the demo data
            first, then this import will be retried automatically.
          </DialogDescription>
        </DialogHeader>
        {notice && <p className="text-sm text-warning">{notice}</p>}
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={handleClearAndRetry} disabled={clearDemoMutation.isPending}>
            {clearDemoMutation.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
            Clear demo data &amp; retry
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
