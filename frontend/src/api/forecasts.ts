import { apiClient } from "@/api/client";
import type {
  ForecastLatestResponse,
  ForecastRunResponse,
  ForecastStatusResponse,
} from "@/types/forecast";

export const getForecastStatus = () => apiClient.get<ForecastStatusResponse>("/forecasts/status");

export const getLatestForecast = () => apiClient.get<ForecastLatestResponse>("/forecasts/latest");

/** POST /api/forecasts/run — the only call that trains (TRD Section 5.6).
 * 422 (cold_start) if months_available < 12; the caller handles that via
 * ApiError, same pattern as Import's demo_conflict/503 handling. */
export const runForecast = () => apiClient.post<ForecastRunResponse>("/forecasts/run");
