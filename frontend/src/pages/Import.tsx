import { useState } from "react";

import { useToast } from "@/components/shared/Toast";
import { useConfirmImport, useCreateImport } from "@/hooks/useImport";
import { ApiError } from "@/types/common";
import type { ImportPreview, ImportResult } from "@/types/import";

import { DemoConflictDialog } from "@/pages/import/DemoConflictDialog";
import { FileUploadCard } from "@/pages/import/FileUploadCard";
import { ImportPreviewCard } from "@/pages/import/ImportPreviewCard";
import { ImportResultCard } from "@/pages/import/ImportResultCard";

type Stage = { step: "upload" } | { step: "preview"; preview: ImportPreview } | { step: "result"; result: ImportResult };

export function ImportPage() {
  const [stage, setStage] = useState<Stage>({ step: "upload" });
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [demoConflictOpen, setDemoConflictOpen] = useState(false);
  const [confirmError, setConfirmError] = useState<string | null>(null);
  const { toast } = useToast();

  const createImportMutation = useCreateImport();
  const confirmImportMutation = useConfirmImport();

  const runUpload = (file: File) => {
    setPendingFile(file);
    createImportMutation.mutate(
      { file, bank: "TD" },
      {
        onSuccess: (preview) => setStage({ step: "preview", preview }),
        onError: (err) => {
          if (err instanceof ApiError && err.error === "demo_conflict") {
            setDemoConflictOpen(true);
          } else if (err instanceof ApiError) {
            toast({ title: "Import failed", description: err.message, variant: "destructive" });
          } else {
            toast({ title: "Import failed", description: "Please try again.", variant: "destructive" });
          }
        },
      },
    );
  };

  const handleConfirm = () => {
    if (stage.step !== "preview") return;
    setConfirmError(null);
    confirmImportMutation.mutate(stage.preview.batch_id, {
      onSuccess: (result) => setStage({ step: "result", result }),
      onError: (err) => {
        if (err instanceof ApiError && err.status === 503) {
          setConfirmError(
            "The categorization model is unavailable, so this import can't be confirmed yet. No transactions were saved. Try again once the model is available.",
          );
        } else if (err instanceof ApiError) {
          setConfirmError(err.message);
        } else {
          setConfirmError("Something went wrong confirming the import.");
        }
      },
    });
  };

  const reset = () => {
    setStage({ step: "upload" });
    setPendingFile(null);
    setConfirmError(null);
  };

  return (
    <div className="flex flex-col gap-5">
      <div>
        <h1 className="text-xl font-semibold">Import</h1>
        <p className="text-sm text-muted-foreground">
          Bring in transactions from a TD CSV export. Nothing is saved until you confirm the
          preview.
        </p>
      </div>

      {stage.step === "upload" && (
        <FileUploadCard onUpload={runUpload} pending={createImportMutation.isPending} />
      )}

      {stage.step === "preview" && (
        <ImportPreviewCard
          preview={stage.preview}
          onConfirm={handleConfirm}
          onCancel={reset}
          pending={confirmImportMutation.isPending}
          error={confirmError}
        />
      )}

      {stage.step === "result" && <ImportResultCard result={stage.result} onImportAnother={reset} />}

      <DemoConflictDialog
        open={demoConflictOpen}
        onOpenChange={setDemoConflictOpen}
        onRetry={() => pendingFile && runUpload(pendingFile)}
      />
    </div>
  );
}
