import { useQuery } from "@tanstack/react-query";

import { getAvailableMonths, getDashboardSummary } from "@/api/dashboard";

export const DASHBOARD_QUERY_KEY = ["dashboard"] as const;

/**
 * TRD §9.3 names this exact scenario: "editing a transaction should refetch
 * the dashboard". useTransactions.ts's create/update/delete mutations
 * invalidate this same DASHBOARD_QUERY_KEY.
 *
 * `month` ("YYYY-MM") is the ONE shared analysis-month clock (see
 * useAnalyticsMonth.ts) — omitted, this defaults to the current calendar
 * month, reproducing prior behavior exactly.
 */
export function useDashboardSummary(month?: string) {
  return useQuery({
    queryKey: [...DASHBOARD_QUERY_KEY, "summary", month ?? null],
    queryFn: () => getDashboardSummary(month),
  });
}

/** Backs the analysis-month selector — only months the user actually has
 * data in. Shares DASHBOARD_QUERY_KEY's root so a transaction create/delete
 * invalidation also refreshes which months are offered. */
export function useAvailableMonths() {
  return useQuery({
    queryKey: [...DASHBOARD_QUERY_KEY, "available-months"],
    queryFn: getAvailableMonths,
  });
}
