/** Mirrors backend/schemas/common.py's DemoLoadResponse/DemoClearResponse
 * (TRD §5.2, §6). */

export interface DemoLoadResponse {
  mode: "DEMO";
  summary: Record<string, number>;
}

export interface DemoClearResponse {
  mode: "EMPTY" | "REAL";
  cleared: boolean;
  summary: Record<string, number>;
}

/** DELETE /api/demo/clear-real-data -- mirror image of DemoClearResponse. */
export interface RealDataClearResponse {
  mode: "EMPTY" | "DEMO";
  cleared: boolean;
  summary: Record<string, number>;
}
