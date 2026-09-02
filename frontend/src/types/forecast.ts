/** Mirrors backend/schemas/forecast.py (TRD Section 6, Section 5.6). */

export type ForecastStatusValue = "ready" | "cold_start" | "no_forecast_yet";

export interface ForecastStatusResponse {
  status: ForecastStatusValue;
  months_available: number;
  months_required: number;
  latest_run_id: number | null;
  is_stale: boolean | null;
}

export interface ForecastPrediction {
  category: string;
  forecast_month: string;
  month_offset: 1 | 2 | 3;
  predicted_amount: number | null;
  is_available: boolean;
  unavailable_reason: string | null;
}

export interface ForecastRunResponse {
  run_id: number;
  generated_at: string;
  is_stale: boolean;
  stale_reason: string | null;
  months_available: number;
  predictions: ForecastPrediction[];
}

/** GET /api/forecasts/latest: either the latest run, or this shape if none
 * has ever been generated (TRD Section 5.6). */
export type ForecastLatestResponse = ForecastRunResponse | { status: "no_forecast_yet" };

export function hasForecastRun(latest: ForecastLatestResponse | undefined): latest is ForecastRunResponse {
  return !!latest && "run_id" in latest;
}
