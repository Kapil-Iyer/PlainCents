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
  /** Why `predicted_category` is what it is (backend/services/
   * category_decision.py's SOURCE_* constants). Null for a pre-migration row
   * or a manually-created transaction (no decision path ran). Never
   * overwritten by a later human correction -- see CategoryBadge.tsx for how
   * this and `is_manual_override` combine. */
  decision_source?:
    | "model"
    | "structural_other"
    | "low_confidence_other"
    | "gazetteer"
    | "ambiguous_e_transfer"
    | null;
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
