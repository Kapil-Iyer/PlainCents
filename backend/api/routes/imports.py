"""Import routes (TRD §5.3, Build Plan Phase 4)."""
import sqlite3

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.api.deps import get_categorization_service, get_db
from backend.api.errors import NotFoundError
from backend.repositories.import_batch_repository import ImportBatchRepository
from backend.schemas.import_ import ImportBatchResponse, ImportPreview, ImportResult
from backend.services.categorization_service import CategorizationService
from backend.services.ingestion_service import IngestionService

router = APIRouter()


def _service(
    conn: sqlite3.Connection, categorization_service: CategorizationService
) -> IngestionService:
    return IngestionService(conn, categorization_service)


@router.post("/api/imports", response_model=ImportPreview)
async def create_import(
    file: UploadFile = File(...),
    bank: str = Form("TD"),
    conn: sqlite3.Connection = Depends(get_db),
    categorization_service: CategorizationService = Depends(get_categorization_service),
) -> ImportPreview:
    file_bytes = await file.read()
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
