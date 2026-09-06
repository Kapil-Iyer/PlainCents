import { apiClient } from "@/api/client";
import type { AvailableMonthsResponse, DashboardSummaryResponse } from "@/types/dashboard";

export const getDashboardSummary = (month?: string) =>
  apiClient.get<DashboardSummaryResponse>(
    month ? `/dashboard/summary?month=${month}` : "/dashboard/summary",
  );

export const getAvailableMonths = () =>
  apiClient.get<AvailableMonthsResponse>("/dashboard/available-months");
