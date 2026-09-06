import { useState } from "react";
import { AlertTriangle, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ConfirmDialog } from "@/components/shared/ConfirmDialog";
import { useToast } from "@/components/shared/Toast";
import { useAppState } from "@/context/AppStateContext";
import { useClearRealData } from "@/hooks/useDemo";
import { ApiError } from "@/types/common";

/**
 * In-app "clear all real data" action — the user-facing equivalent of the
 * developer-only scripts/reset_real_data.py maintenance script. Exists for
 * anywhere that script isn't reachable (no shell access to the running
 * instance, e.g. once deployed) and for anyone who'd rather not touch a
 * terminal at all.
 *
 * Only rendered while mode === "REAL" (nothing to clear otherwise) — see
 * ImportPage. Clearing returns the app to EMPTY, which is also what
 * unblocks Load Demo Data (DemoService.load_demo() rejects with 409 while
 * mode === "REAL").
 */
export function ClearRealDataCard() {
  const { mode } = useAppState();
  const clearRealDataMutation = useClearRealData();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const { toast } = useToast();

  if (mode !== "REAL") return null;

  const handleClear = async () => {
    try {
      await clearRealDataMutation.mutateAsync();
      toast({ title: "Real data cleared", description: "You can now import again or load demo data." });
    } catch (err) {
      toast({
        title: "Couldn't clear real data",
        description: err instanceof ApiError ? err.message : "Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <>
      <Card className="border-destructive/40">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <AlertTriangle className="h-4 w-4 text-destructive" />
            Danger zone
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-muted-foreground">
            Permanently delete every imported transaction, holding, and forecast, and return the
            app to empty — so you can start over or load demo data instead.
          </p>
          <Button variant="destructive" onClick={() => setConfirmOpen(true)}>
            <Trash2 className="h-4 w-4" />
            Clear all real data
          </Button>
        </CardContent>
      </Card>

      <ConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        title="Clear all real data?"
        description="This permanently deletes every real transaction, holding, and forecast run. This can't be undone."
        confirmLabel="Clear all real data"
        onConfirm={handleClear}
      />
    </>
  );
}
