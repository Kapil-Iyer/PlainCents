import { apiClient } from "@/api/client";
import type { DashboardSummaryResponse } from "@/types/dashboard";

export const getDashboardSummary = () =>
  apiClient.get<DashboardSummaryResponse>("/dashboard/summary");
