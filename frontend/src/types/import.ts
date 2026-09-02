/** Mirrors backend/schemas/import_.py (TRD §6, §5.3). */

export interface ImportSampleRow {
  date: string;
  merchant: string;
  amount: number;
  predicted_category: string | null;
  is_duplicate: boolean;
}

export interface ImportPreview {
  batch_id: number;
  rows_valid: number;
  rows_unparseable: number;
  rows_duplicate: number;
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
  created_at: string;
  confirmed_at: string | null;
}
