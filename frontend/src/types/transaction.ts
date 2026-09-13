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
   * category_decision.py's SOURCE_* constants). Set by the same shared
   * decision path for both Import and manually-created transactions; null
   * only for a pre-migration row. Never overwritten by a later human
   * correction -- see CategoryBadge.tsx for how this and
   * `is_manual_override` combine. */
  decision_source?:
    | "model"
    | "structural_other"
    | "low_confidence_other"
    | "gazetteer"
    | "ambiguous_e_transfer"
    | "internal_transfer"
    | null;
  /** Advisory only: what the classifier alone said, even when a
   * low-confidence abstention overrode it to "Other" (predicted_category).
   * Never affects predicted_category/confirmed_category/effective_category,
   * and is never touched by a later human correction. Null on structural/
   * ambiguous-e-transfer rows, or a pre-migration row. Drives
   * CategoryBadge.tsx's "Suggested: {model_category}" advisory chip and its
   * one-click "Use" accept. */
  model_category?: string | null;
  /** Spending eligibility, ORTHOGONAL to category (backend/services/
   * transfer_eligibility.py). "internal_transfer" means a structurally-
   * detected same-owner account transfer: still visible here, but excluded
   * from every spend total/forecast/category summary. CategoryBadge.tsx
   * renders this as a plain "Internal transfer" label instead of a category
   * badge -- it is never shown a "Suggested category" chip, since it was
   * never run through categorization at all. Null only for a pre-migration
   * row (treated the same as "spending"). */
  transaction_type?: "spending" | "internal_transfer" | null;
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
