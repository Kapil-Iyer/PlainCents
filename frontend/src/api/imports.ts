import { apiClient } from "@/api/client";
import type { ImportBatchResponse, ImportPreview, ImportResult } from "@/types/import";

export const createImport = (file: File, bank = "TD") => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("bank", bank);
  return apiClient.post<ImportPreview>("/imports", formData);
};

export const confirmImport = (batchId: number) =>
  apiClient.post<ImportResult>(`/imports/${batchId}/confirm`);

export const listImports = () => apiClient.get<ImportBatchResponse[]>("/imports");

export const getImport = (batchId: number) =>
  apiClient.get<ImportBatchResponse>(`/imports/${batchId}`);
