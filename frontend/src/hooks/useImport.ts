import { useMutation, useQueryClient } from "@tanstack/react-query";

import { confirmImport, createImport } from "@/api/imports";
import { APP_STATE_QUERY_KEY } from "@/context/AppStateContext";
import { DASHBOARD_QUERY_KEY } from "@/hooks/useDashboard";

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
      queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEY });
    },
  });
}
