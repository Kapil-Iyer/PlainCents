"""Power BI export route (PATCH D). One GET endpoint, one on-demand ZIP —
no export job, no stored artifact, no polling: the same request that asks
for the file gets it back."""
import sqlite3

from fastapi import APIRouter, Depends, Response

from backend.api.deps import get_db
from backend.repositories.mode_filter import resolve_data_mode_filter
from backend.services.app_state_service import AppStateService
from backend.services.powerbi_export_service import PowerBIExportService, export_filename

router = APIRouter()


@router.get("/api/export/powerbi")
def export_powerbi(conn: sqlite3.Connection = Depends(get_db)) -> Response:
    # Same mode-resolution convention as every other read endpoint
    # (dashboard.py, transactions.py, holdings.py): the route decides which
    # data_mode is currently visible, the service just executes the read.
    mode = AppStateService(conn).get_mode()
    data_mode = resolve_data_mode_filter(mode)
    service = PowerBIExportService(conn)
    zip_bytes = service.build_export_zip(data_mode)
    filename = export_filename()
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
