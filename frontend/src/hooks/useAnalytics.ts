import { useQuery } from "@tanstack/react-query";

import {
  getCategoryMovers,
  getCategoryTrend,
  getForecastAccuracy,
  getSpendPace,
  getTopMerchants,
} from "@/api/analytics";

/**
 * Analytics query keys all share the "analytics" root so a single
 * invalidation after a transaction edit refreshes every chart at once —
 * a category correction changes the trend, the movers, the pace and the
 * merchant table simultaneously, and they must never disagree on screen.
 */
export const ANALYTICS_QUERY_KEY = ["analytics"] as const;

export function useCategoryTrend(months: number) {
  return useQuery({
    queryKey: [...ANALYTICS_QUERY_KEY, "category-trend", months],
    queryFn: () => getCategoryTrend(months),
  });
}

export function useTopMerchants(limit: number, months: number) {
  return useQuery({
    queryKey: [...ANALYTICS_QUERY_KEY, "top-merchants", limit, months],
    queryFn: () => getTopMerchants(limit, months),
  });
}

/**
 * `month` ("YYYY-MM") is the ONE shared analysis-month clock also driving
 * the Dashboard's Change KPI and Spending Pace (see useAnalyticsMonth.ts) —
 * omitted, this defaults to the current calendar month.
 */
export function useCategoryMovers(month?: string) {
  return useQuery({
    queryKey: [...ANALYTICS_QUERY_KEY, "category-movers", month ?? null],
    queryFn: () => getCategoryMovers(month),
  });
}

export function useSpendPace(month?: string) {
  return useQuery({
    queryKey: [...ANALYTICS_QUERY_KEY, "spend-pace", month ?? null],
    queryFn: () => getSpendPace(month),
  });
}

export function useForecastAccuracy() {
  return useQuery({
    queryKey: [...ANALYTICS_QUERY_KEY, "forecast-accuracy"],
    queryFn: getForecastAccuracy,
  });
}
