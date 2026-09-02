import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  createHolding,
  deleteHolding,
  listHoldings,
  refreshPrices,
  updateHolding,
} from "@/api/holdings";
import { APP_STATE_QUERY_KEY } from "@/context/AppStateContext";
import { DASHBOARD_QUERY_KEY } from "@/hooks/useDashboard";
import type { HoldingCreate, HoldingUpdate } from "@/types/holding";

const HOLDINGS_KEY = "holdings";

export function useHoldingsQuery() {
  return useQuery({
    queryKey: [HOLDINGS_KEY],
    queryFn: listHoldings,
  });
}

export function useCreateHolding() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: HoldingCreate) => createHolding(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [HOLDINGS_KEY] });
      queryClient.invalidateQueries({ queryKey: APP_STATE_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEY });
    },
  });
}

export function useUpdateHolding() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, payload }: { id: number; payload: HoldingUpdate }) =>
      updateHolding(id, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [HOLDINGS_KEY] });
      queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEY });
    },
  });
}

export function useDeleteHolding() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => deleteHolding(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [HOLDINGS_KEY] });
      queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEY });
    },
  });
}

/** PRD §9.7/§11.9: opening Portfolio never triggers a fetch — this only
 * fires from an explicit "Refresh Prices" click. */
export function useRefreshPrices() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => refreshPrices(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [HOLDINGS_KEY] });
      queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEY });
    },
  });
}
