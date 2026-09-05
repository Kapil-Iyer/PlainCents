import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { getForecastStatus, getLatestForecast, runForecast } from "@/api/forecasts";
import { ANALYTICS_QUERY_KEY } from "@/hooks/useAnalytics";
import { DASHBOARD_QUERY_KEY } from "@/hooks/useDashboard";

export const FORECAST_STATUS_QUERY_KEY = ["forecast", "status"] as const;
export const FORECAST_LATEST_QUERY_KEY = ["forecast", "latest"] as const;

export function useForecastStatus() {
  return useQuery({
    queryKey: FORECAST_STATUS_QUERY_KEY,
    queryFn: getForecastStatus,
  });
}

/** Only fetched once status is known and isn't cold_start — reading latest
 * during cold-start would just return {status: "no_forecast_yet"} anyway,
 * but skipping the request keeps the two reads sequenced sensibly and
 * avoids a request the ColdStartState screen doesn't need. */
export function useLatestForecast(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: FORECAST_LATEST_QUERY_KEY,
    queryFn: getLatestForecast,
    enabled: options?.enabled ?? true,
  });
}

export function useRunForecast() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => runForecast(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: FORECAST_STATUS_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: FORECAST_LATEST_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: ANALYTICS_QUERY_KEY });
    },
  });
}
