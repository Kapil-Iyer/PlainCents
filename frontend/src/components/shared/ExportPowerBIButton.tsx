import { Download, Loader2 } from "lucide-react";
import { useMutation } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import { useToast } from "@/components/shared/Toast";
import { downloadPowerBIExport } from "@/api/export";
import { ApiError } from "@/types/common";

/**
 * PATCH D: on-demand Power BI "current-state" export. One click, one file
 * -- no export job to poll, no email, no background task. The ZIP is built
 * fresh from the live database on the same request that downloads it, so
 * it always reflects whatever the app is showing right now.
 */
export function ExportPowerBIButton() {
  const { toast } = useToast();
  const mutation = useMutation({
    mutationFn: downloadPowerBIExport,
    onError: (err) => {
      toast({
        title: "Couldn't generate the export",
        description: err instanceof ApiError ? err.message : "Please try again.",
        variant: "destructive",
      });
    },
  });

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={() => mutation.mutate()}
      disabled={mutation.isPending}
    >
      {mutation.isPending ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        <Download className="h-4 w-4" />
      )}
      Export for Power BI
    </Button>
  );
}
