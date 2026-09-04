import { apiClient } from "@/api/client";
import type { ImportBatchResponse, ImportPreview, ImportResult } from "@/types/import";

// Phase 12A.5/12B: `bank` is optional — omitted (or "Auto") means auto-detect
// across the four implemented banks (RBC/Scotiabank/TD/CIBC) server-side.
// An explicit bank name validates only that bank's own format.
export const createImport = (file: File, bank?: string) => {
  const formData = new FormData();
  formData.append("file", file);
  if (bank && bank !== "Auto") {
    formData.append("bank", bank);
  }
  return apiClient.post<ImportPreview>("/imports", formData);
};

export const confirmImport = (batchId: number) =>
  apiClient.post<ImportResult>(`/imports/${batchId}/confirm`);

export const listImports = () => apiClient.get<ImportBatchResponse[]>("/imports");

export const getImport = (batchId: number) =>
  apiClient.get<ImportBatchResponse>(`/imports/${batchId}`);
