import { apiClient } from "@/api/client";
import type { DemoClearResponse, DemoLoadResponse, RealDataClearResponse } from "@/types/demo";

/** POST /api/demo/load — TRD §5.2/§14.2: 200 + summary on success, 409
 * demo_conflict if mode isn't EMPTY. Real data is never deleted to make
 * room for demo data. */
export const loadDemo = () => apiClient.post<DemoLoadResponse>("/demo/load");

/** DELETE /api/demo/clear — TRD §5.2: 200 on success, idempotent even if
 * already EMPTY. Full reset of demo-flagged data only; never touches real
 * data. */
export const clearDemo = () => apiClient.delete<DemoClearResponse>("/demo/clear");

/** DELETE /api/demo/clear-real-data — mirror image of clearDemo: full reset
 * of real-flagged data only, so Load Demo Data becomes available again.
 * The in-app equivalent of scripts/reset_real_data.py, for anywhere without
 * shell access to the running instance. Idempotent even if there's no real
 * data yet. */
export const clearRealData = () => apiClient.delete<RealDataClearResponse>("/demo/clear-real-data");
