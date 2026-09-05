import { apiClient } from "@/api/client";
import type {
  CategoryMoversResponse,
  CategoryTrendResponse,
  ForecastAccuracyResponse,
  SpendPaceResponse,
  TopMerchantsResponse,
} from "@/types/analytics";

export const getCategoryTrend = (months: number) =>
  apiClient.get<CategoryTrendResponse>(`/analytics/category-trend?months=${months}`);

export const getTopMerchants = (limit: number, months: number) =>
  apiClient.get<TopMerchantsResponse>(
    `/analytics/top-merchants?limit=${limit}&months=${months}`,
  );

export const getCategoryMovers = () =>
  apiClient.get<CategoryMoversResponse>("/analytics/category-movers");

export const getSpendPace = () => apiClient.get<SpendPaceResponse>("/analytics/spend-pace");

export const getForecastAccuracy = () =>
  apiClient.get<ForecastAccuracyResponse>("/analytics/forecast-accuracy");
