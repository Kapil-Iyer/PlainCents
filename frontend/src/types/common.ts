/** Mirrors backend/schemas/common.py (TRD §6, §15). */

export interface ErrorResponse {
  error: string;
  message: string;
  details: Record<string, unknown>;
}

export type DataMode = "EMPTY" | "DEMO" | "REAL";

export interface HealthResponse {
  db: "ok" | "error";
  categorization_model: "loaded" | "missing" | "error";
  data_mode: DataMode;
}

export interface DemoStatusResponse {
  mode: DataMode;
  can_load_demo: boolean;
}

/** Thrown by the API client on any non-2xx response (TRD §15 envelope). */
export class ApiError extends Error {
  status: number;
  error: string;
  details: Record<string, unknown>;

  constructor(status: number, body: ErrorResponse) {
    super(body.message);
    this.name = "ApiError";
    this.status = status;
    this.error = body.error;
    this.details = body.details ?? {};
  }
}
