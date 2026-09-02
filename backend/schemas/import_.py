"""Import schemas (TRD §6, §5.3; Build Plan Phase 4)."""
from datetime import datetime

from pydantic import BaseModel


class ImportSampleRow(BaseModel):
    """One row of ImportPreview.sample_rows — enough to show the user what
    was parsed, without exposing staged_transactions' internal id/dedup_key."""

    date: str
    merchant: str
    amount: float
    predicted_category: str | None
    is_duplicate: bool


class ImportPreview(BaseModel):
    """TRD §6. `categorization_available` is an addition beyond §6's field
    list, required by TRD §10's explicit behavior: "preview itself still
    succeeds but reports that categorization could not run" — there must be
    a field for the frontend to read that state from."""

    batch_id: int
    rows_valid: int
    rows_unparseable: int
    rows_duplicate: int
    date_range: dict[str, str | None]
    sample_rows: list[ImportSampleRow]
    status: str
    categorization_available: bool


class ImportResult(BaseModel):
    """TRD §6."""

    batch_id: int
    rows_imported: int
    rows_skipped_unparseable: int
    rows_skipped_duplicate: int
    status: str


class ImportBatchResponse(BaseModel):
    """GET /api/imports, GET /api/imports/{batch_id} (TRD §5.3) — the user's
    own import history, mirroring the import_batches row shape."""

    id: int
    bank_source: str
    original_filename: str | None
    status: str
    rows_valid: int
    rows_unparseable: int
    rows_duplicate: int
    rows_imported: int
    created_at: datetime
    confirmed_at: datetime | None
