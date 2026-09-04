"""Import routes (TRD §5.3, Build Plan Phase 4)."""
import sqlite3

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.api.deps import get_categorization_service, get_db
from backend.api.errors import BadRequestError, NotFoundError
from backend.repositories.import_batch_repository import ImportBatchRepository
from backend.schemas.import_ import ImportBatchResponse, ImportPreview, ImportResult
from backend.services.categorization_service import CategorizationService
from backend.services.ingestion_service import IngestionService

router = APIRouter()

# Phase 12A found `await file.read()` fully unbounded. A practical cap (not
# a streaming rewrite -- Phase 12A.5 §28) rejects an oversized upload with a
# clear 400 before it reaches CSV parsing.
MAX_IMPORT_FILE_BYTES = 5 * 1024 * 1024  # 5 MB


def _service(
    conn: sqlite3.Connection, categorization_service: CategorizationService
) -> IngestionService:
    return IngestionService(conn, categorization_service)


@router.post("/api/imports", response_model=ImportPreview)
async def create_import(
    file: UploadFile = File(...),
    bank: str | None = Form(None),
    conn: sqlite3.Connection = Depends(get_db),
    categorization_service: CategorizationService = Depends(get_categorization_service),
) -> ImportPreview:
    # Phase 12A.5/12B: bank is optional -- omitted (or "Auto") means
    # auto-detect across the four implemented banks (RBC/Scotiabank/TD/CIBC);
    # an explicit bank name validates only that bank's own format and never
    # silently reinterprets the file as a different one.
    file_bytes = await file.read()
    if len(file_bytes) > MAX_IMPORT_FILE_BYTES:
        raise BadRequestError(
            f"File is too large ({len(file_bytes) / 1_000_000:.1f} MB). "
            f"The maximum supported size is {MAX_IMPORT_FILE_BYTES / 1_000_000:.0f} MB."
        )
    service = _service(conn, categorization_service)
    preview = service.parse_and_stage(file_bytes, bank=bank, original_filename=file.filename)
    return ImportPreview(**preview)


@router.post("/api/imports/{batch_id}/confirm", response_model=ImportResult)
def confirm_import(
    batch_id: int,
    conn: sqlite3.Connection = Depends(get_db),
    categorization_service: CategorizationService = Depends(get_categorization_service),
) -> ImportResult:
    service = _service(conn, categorization_service)
    result = service.commit_import(batch_id)
    return ImportResult(**result)


@router.get("/api/imports", response_model=list[ImportBatchResponse])
def list_imports(conn: sqlite3.Connection = Depends(get_db)) -> list[ImportBatchResponse]:
    batches = ImportBatchRepository(conn).list()
    return [ImportBatchResponse(**b) for b in batches]


@router.get("/api/imports/{batch_id}", response_model=ImportBatchResponse)
def get_import(batch_id: int, conn: sqlite3.Connection = Depends(get_db)) -> ImportBatchResponse:
    batch = ImportBatchRepository(conn).get(batch_id)
    if batch is None:
        raise NotFoundError(f"Import batch {batch_id} not found.")
    return ImportBatchResponse(**batch)
