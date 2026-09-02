import { apiClient } from "@/api/client";

/**
 * DELETE /api/demo/clear returns 501 until Phase 9 (Build Plan §2.5) — this
 * function is wired now so the Import page's demo-conflict flow is fully
 * built, but callers must handle the 501 ApiError gracefully rather than
 * assuming it succeeds.
 */
export const clearDemo = () => apiClient.delete<unknown>("/demo/clear");
