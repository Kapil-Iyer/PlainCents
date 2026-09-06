"""Import schemas (TRD §6, §5.3; Build Plan Phase 4)."""
from datetime import datetime

from pydantic import BaseModel


class ImportSampleRow(BaseModel):
    """One row of ImportPreview.sample_rows — enough to show the user what
    was parsed and WHY it was categorized the way it was, without exposing
    staged_transactions' internal id/dedup_key.

    ML-G: the decision fields below are what make Preview an honest preview.
    Previously this carried only `predicted_category` (the raw model output),
    while Confirm separately applied structural-ambiguity routing and
    remembered corrections — so the table showed a category that would not be
    the one stored.

      predicted_category   the SYSTEM's decision for this row
      remembered_category  a prior GENUINE user correction that will be
                           written to confirmed_category on confirm; None if
                           no correction is remembered
      effective_category   what the row will actually count as — exactly
                           COALESCE(remembered, predicted), the same rule the
                           v_transactions_effective view applies
      decision_source      'model' | 'structural_other' | 'low_confidence_other'
                           | 'gazetteer' (a deterministic public-brand/service
                           match, backend.services.gazetteer) |
                           'ambiguous_e_transfer' (an E-Transfer with no
                           purpose evidence in its description,
                           backend.services.e_transfer_policy -- distinct
                           from 'structural_other', which names nothing at
                           all) -- see backend.services.category_decision
    """

    date: str
    merchant: str
    amount: float
    predicted_category: str | None
    remembered_category: str | None = None
    effective_category: str | None = None
    decision_source: str | None = None
    is_duplicate: bool


class ImportPreview(BaseModel):
    """TRD §6. `categorization_available` is an addition beyond §6's field
    list, required by TRD §10's explicit behavior: "preview itself still
    succeeds but reports that categorization could not run" — there must be
    a field for the frontend to read that state from.

    `detected_bank`, `rows_skipped_credit`, `rows_skipped_currency` are
    Phase 12A.5/12B additions for six-bank support: `detected_bank` is the
    bank actually resolved (explicit selection, or the winning auto-detect
    fingerprint) — previously computed by the parser but discarded before
    reaching this schema. The two skip counts surface rows that were
    correctly recognized and intentionally excluded (credits/deposits;
    RBC USD$-only rows), distinct from `rows_unparseable` (genuinely
    malformed rows)."""

    batch_id: int
    detected_bank: str
    rows_valid: int
    rows_unparseable: int
    rows_duplicate: int
    rows_skipped_credit: int
    rows_skipped_currency: int
    date_range: dict[str, str | None]
    sample_rows: list[ImportSampleRow]
    status: str
    categorization_available: bool


class ImportResult(BaseModel):
    """TRD §6, extended Phase 12B with the same exclusion counts as
    ImportPreview (§24) so the result screen can show them too."""

    batch_id: int
    rows_imported: int
    rows_skipped_unparseable: int
    rows_skipped_duplicate: int
    rows_skipped_credit: int
    rows_skipped_currency: int
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
    rows_skipped_credit: int
    rows_skipped_currency: int
    created_at: datetime
    confirmed_at: datetime | None
