/** Mirrors backend/schemas/transaction.py (TRD §6, §5.4). */

export interface TransactionResponse {
  id: number;
  date: string;
  merchant: string;
  raw_description: string | null;
  amount: number;
  bank_source: string | null;
  predicted_category: string;
  confirmed_category: string | null;
  effective_category: string;
  is_manual_override: boolean;
  created_at: string;
  updated_at: string;
}

export interface TransactionListResponse {
  items: TransactionResponse[];
  total: number;
  page: number;
  page_size: number;
}

export interface TransactionCreate {
  date: string;
  merchant: string;
  amount: number;
  confirmed_category?: string | null;
}

export interface TransactionUpdate {
  date?: string;
  merchant?: string;
  amount?: number;
  confirmed_category?: string | null;
}

export interface TransactionListParams {
  date_from?: string;
  date_to?: string;
  category?: string;
  search?: string;
  sort?: string;
  page?: number;
  page_size?: number;
}
