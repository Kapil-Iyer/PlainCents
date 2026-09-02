import { useMutation, useQueryClient } from "@tanstack/react-query";

import { clearDemo, loadDemo } from "@/api/demo";
import { APP_STATE_QUERY_KEY } from "@/context/AppStateContext";
import { DASHBOARD_QUERY_KEY } from "@/hooks/useDashboard";
import { FORECAST_LATEST_QUERY_KEY, FORECAST_STATUS_QUERY_KEY } from "@/hooks/useForecast";

/** Every currently-established query key whose results depend on app mode
 * or demo-backed data (TRD §9.3's "editing data should refetch the
 * dashboard" pattern, extended to every mode-scoped read). Transactions and
 * holdings use their hooks' own literal key prefixes ("transactions"/
 * "holdings") the same way useConfirmImport already does, since those
 * consts aren't exported from useTransactions.ts/useHoldings.ts. */
function invalidateModeScopedQueries(queryClient: ReturnType<typeof useQueryClient>) {
  queryClient.invalidateQueries({ queryKey: APP_STATE_QUERY_KEY });
  queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEY });
  queryClient.invalidateQueries({ queryKey: FORECAST_STATUS_QUERY_KEY });
  queryClient.invalidateQueries({ queryKey: FORECAST_LATEST_QUERY_KEY });
  queryClient.invalidateQueries({ queryKey: ["transactions"] });
  queryClient.invalidateQueries({ queryKey: ["holdings"] });
}

/** PRD §10a: the onboarding empty state's "Load demo data" action. */
export function useLoadDemo() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => loadDemo(),
    onSuccess: () => invalidateModeScopedQueries(queryClient),
  });
}

/** TRD §14.3/§14.4: the demo→real sequence's clear step, called from
 * DemoConflictDialog after the user confirms. */
export function useClearDemo() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => clearDemo(),
    onSuccess: () => invalidateModeScopedQueries(queryClient),
  });
}
