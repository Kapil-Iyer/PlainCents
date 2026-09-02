import { useQuery } from "@tanstack/react-query";

import { getDashboardSummary } from "@/api/dashboard";

export const DASHBOARD_QUERY_KEY = ["dashboard"] as const;

/**
 * TRD §9.3 names this exact scenario: "editing a transaction should refetch
 * the dashboard". useTransactions.ts's create/update/delete mutations
 * invalidate this same DASHBOARD_QUERY_KEY.
 */
export function useDashboardSummary() {
  return useQuery({
    queryKey: DASHBOARD_QUERY_KEY,
    queryFn: getDashboardSummary,
  });
}
