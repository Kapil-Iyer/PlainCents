import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  createTransaction,
  deleteTransaction,
  listTransactions,
  updateTransaction,
} from "@/api/transactions";
import { APP_STATE_QUERY_KEY } from "@/context/AppStateContext";
import { ANALYTICS_QUERY_KEY } from "@/hooks/useAnalytics";
import { DASHBOARD_QUERY_KEY } from "@/hooks/useDashboard";
import type {
  TransactionCreate,
  TransactionListParams,
  TransactionUpdate,
} from "@/types/transaction";

const TRANSACTIONS_KEY = "transactions";

export function useTransactionsQuery(params: TransactionListParams) {
  return useQuery({
    queryKey: [TRANSACTIONS_KEY, params],
    queryFn: () => listTransactions(params),
    placeholderData: (prev) => prev,
  });
}

export function useCreateTransaction() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: TransactionCreate) => createTransaction(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [TRANSACTIONS_KEY] });
      queryClient.invalidateQueries({ queryKey: APP_STATE_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: ANALYTICS_QUERY_KEY });
    },
  });
}

export function useUpdateTransaction() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, payload }: { id: number; payload: TransactionUpdate }) =>
      updateTransaction(id, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [TRANSACTIONS_KEY] });
      queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: ANALYTICS_QUERY_KEY });
    },
  });
}

export function useDeleteTransaction() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => deleteTransaction(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [TRANSACTIONS_KEY] });
      queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: ANALYTICS_QUERY_KEY });
    },
  });
}
