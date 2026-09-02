import { apiClient } from "@/api/client";
import type {
  HoldingCreate,
  HoldingResponse,
  HoldingUpdate,
  RefreshPricesResponse,
} from "@/types/holding";

export const listHoldings = () => apiClient.get<HoldingResponse[]>("/holdings");

export const createHolding = (payload: HoldingCreate) =>
  apiClient.post<HoldingResponse>("/holdings", payload);

export const updateHolding = (id: number, payload: HoldingUpdate) =>
  apiClient.patch<HoldingResponse>(`/holdings/${id}`, payload);

export const deleteHolding = (id: number) =>
  apiClient.delete<{ id: number; deleted: boolean }>(`/holdings/${id}`);

/** POST /api/holdings/refresh-prices — the only call that reaches yfinance
 * (TRD §5.7/§13.2). Never called automatically; only from an explicit click. */
export const refreshPrices = () => apiClient.post<RefreshPricesResponse>("/holdings/refresh-prices");
