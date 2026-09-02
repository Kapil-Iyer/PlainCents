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
import { useClearDemo } from "@/hooks/useDemo";
import { ApiError } from "@/types/common";

interface DemoConflictDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Retried after a successful demo clear. */
  onRetry: () => void;
}

/**
 * TRD §5.2/§14.4: a real import while mode === 'DEMO' returns 409
 * demo_conflict. The sanctioned flow is DELETE /api/demo/clear then retry
 * the import — both are real as of Phase 9.
 */
export function DemoConflictDialog({ open, onOpenChange, onRetry }: DemoConflictDialogProps) {
  const clearDemoMutation = useClearDemo();
  const [error, setError] = useState<string | null>(null);

  const handleClearAndRetry = async () => {
    setError(null);
    try {
      await clearDemoMutation.mutateAsync();
      onOpenChange(false);
      onRetry();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't clear demo data. Please try again.");
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
        {error && <p className="text-sm text-destructive">{error}</p>}
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
