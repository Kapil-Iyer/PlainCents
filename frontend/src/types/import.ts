/** Mirrors backend/schemas/import_.py (TRD §6, §5.3). */

export interface ImportSampleRow {
  date: string;
  merchant: string;
  amount: number;
  /** The SYSTEM's decision for this row. */
  predicted_category: string | null;
  /** A prior genuine correction of yours for this merchant, which will be
   * applied on confirm. Null when none is remembered. */
  remembered_category: string | null;
  /** What the row will actually count as — COALESCE(remembered, predicted),
   * the same rule the rest of the app uses. */
  effective_category: string | null;
  /** Why `predicted_category` is what it is. "gazetteer" is a deterministic
   * public-brand/service match (backend/services/gazetteer.py), not a model
   * guess. "ambiguous_e_transfer" is an E-Transfer with no purpose evidence
   * in its description (backend/services/e_transfer_policy.py) -- distinct
   * from "structural_other" (a transfer/withdrawal that names nothing at
   * all). */
  decision_source:
    | "model"
    | "structural_other"
    | "low_confidence_other"
    | "gazetteer"
    | "ambiguous_e_transfer"
    | "internal_transfer"
    | null;
  /** Advisory only: what the classifier alone said, even when a
   * low-confidence abstention overrode it to "Other". Null on structural/
   * ambiguous-e-transfer rows (the model is never called on those paths).
   * Never affects predicted_category/effective_category. Display-only here
   * (no "Use" action in Preview — no transaction id exists yet); see
   * CategoryBadge.tsx for the post-Confirm one-click accept. */
  model_category?: string | null;
  /** Spending eligibility (backend/services/transfer_eligibility.py),
   * ORTHOGONAL to category. "internal_transfer" rows will not count toward
   * the imported total's spend/forecast/category-summary figures even
   * though `effective_category` still reads "Other". */
  transaction_type?: "spending" | "internal_transfer" | null;
  is_duplicate: boolean;
}

export interface ImportPreview {
  batch_id: number;
  /** The bank actually resolved — an explicit selection, or the winning
   * auto-detect fingerprint (Phase 12A.5/12B). */
  detected_bank: string;
  rows_valid: number;
  rows_unparseable: number;
  rows_duplicate: number;
  /** Rows correctly recognized as credits/deposits and intentionally
   * excluded (not malformed) — Phase 12A.5 §17/§24. */
  rows_skipped_credit: number;
  /** RBC USD$-only rows excluded as an unsupported currency (no
   * conversion) — Phase 12A.5 §18. */
  rows_skipped_currency: number;
  /** Rows structurally detected as an internal/self account transfer
   * (backend/services/transfer_eligibility.py) — a SUBSET of rows_valid,
   * not a separate exclusive bucket (same relationship rows_duplicate has).
   * These rows ARE still imported and stay visible afterward, labeled
   * "Internal transfer"; this count exists so Preview can say up front how
   * many will be excluded from spend totals. */
  rows_internal_transfer: number;
  date_range: { from: string | null; to: string | null };
  sample_rows: ImportSampleRow[];
  status: string;
  categorization_available: boolean;
}

export interface ImportResult {
  batch_id: number;
  rows_imported: number;
  rows_skipped_unparseable: number;
  rows_skipped_duplicate: number;
  rows_skipped_credit: number;
  rows_skipped_currency: number;
  rows_internal_transfer: number;
  status: string;
}

export interface ImportBatchResponse {
  id: number;
  bank_source: string;
  original_filename: string | null;
  status: string;
  rows_valid: number;
  rows_unparseable: number;
  rows_duplicate: number;
  rows_imported: number;
  rows_skipped_credit: number;
  rows_skipped_currency: number;
  rows_internal_transfer: number;
  created_at: string;
  confirmed_at: string | null;
}

/** The six named export formats (Phase 12A.5 evidence gate). Phase 12B
 * closure patch: BMO and National Bank stay visible in the selector (so the
 * roadmap is honest) but are disabled/non-selectable — selecting an
 * unimplemented bank and only failing after upload is exactly what this
 * patch removes. The backend's own "not yet supported" guard for
 * bank="BMO"/"National Bank" stays in place regardless, as defense in
 * depth for any direct API call that bypasses the UI. */
export const SUPPORTED_BANKS = ["RBC", "Scotiabank", "TD", "CIBC"] as const;
export const COMING_SOON_BANKS = ["BMO", "National Bank"] as const;
export type BankName = (typeof SUPPORTED_BANKS)[number] | (typeof COMING_SOON_BANKS)[number];
