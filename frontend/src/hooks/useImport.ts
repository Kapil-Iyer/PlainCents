import { useMutation, useQueryClient } from "@tanstack/react-query";

import { clearDemo } from "@/api/demo";
import { confirmImport, createImport } from "@/api/imports";
import { APP_STATE_QUERY_KEY } from "@/context/AppStateContext";

export function useCreateImport() {
  return useMutation({
    mutationFn: ({ file, bank }: { file: File; bank?: string }) => createImport(file, bank),
  });
}

export function useConfirmImport() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (batchId: number) => confirmImport(batchId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["transactions"] });
      queryClient.invalidateQueries({ queryKey: APP_STATE_QUERY_KEY });
    },
  });
}

/**
 * DELETE /api/demo/clear is a 501 stub until Phase 9 (Build Plan §2.5) — this
 * hook still calls it so the demo-conflict UI flow is fully wired; callers
 * must handle the 501 gracefully rather than assuming demo data was cleared.
 */
export function useClearDemo() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => clearDemo(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: APP_STATE_QUERY_KEY });
    },
  });
}
