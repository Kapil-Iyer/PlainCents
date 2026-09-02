import { apiClient } from "@/api/client";
import type {
  TransactionCreate,
  TransactionListParams,
  TransactionListResponse,
  TransactionResponse,
  TransactionUpdate,
} from "@/types/transaction";

function buildQuery(params: TransactionListParams): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, String(value));
    }
  }
  const qs = search.toString();
  return qs ? `?${qs}` : "";
}

export const listTransactions = (params: TransactionListParams = {}) =>
  apiClient.get<TransactionListResponse>(`/transactions${buildQuery(params)}`);

export const getTransaction = (id: number) =>
  apiClient.get<TransactionResponse>(`/transactions/${id}`);

export const createTransaction = (payload: TransactionCreate) =>
  apiClient.post<TransactionResponse>("/transactions", payload);

export const updateTransaction = (id: number, payload: TransactionUpdate) =>
  apiClient.patch<TransactionResponse>(`/transactions/${id}`, payload);

export const deleteTransaction = (id: number) =>
  apiClient.delete<{ id: number; deleted: boolean }>(`/transactions/${id}`);
