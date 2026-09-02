import { apiClient } from "@/api/client";
import type { DemoStatusResponse, HealthResponse } from "@/types/common";

export const getHealth = () => apiClient.get<HealthResponse>("/health");
export const getDemoStatus = () => apiClient.get<DemoStatusResponse>("/demo/status");
