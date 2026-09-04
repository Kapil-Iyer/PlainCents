"""ImportBatchRepository — persistence only (TRD §8)."""
from __future__ import annotations

import sqlite3


class ImportBatchRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def create_preview(
        self, bank_source: str, original_filename: str | None = None, data_mode: str = "real"
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO import_batches (bank_source, original_filename, status, data_mode)
            VALUES (?, ?, 'previewing', ?)
            """,
            (bank_source, original_filename, data_mode),
        )
        return cur.lastrowid

    def get(self, batch_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM import_batches WHERE id = ?", (batch_id,)
        ).fetchone()
        return dict(row) if row else None

    def list(self) -> list[dict]:
        # id DESC tie-break for the same reason as ForecastRepository.get_latest_run:
        # SQLite's CURRENT_TIMESTAMP has 1-second resolution.
        rows = self._conn.execute(
            "SELECT * FROM import_batches ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_status(self, batch_id: int, status: str, counts: dict | None = None) -> bool:
        counts = counts or {}
        set_parts = ["status = ?"]
        params: list = [status]
        for key in (
            "rows_valid",
            "rows_unparseable",
            "rows_duplicate",
            "rows_imported",
            "rows_skipped_credit",
            "rows_skipped_currency",
        ):
            if key in counts:
                set_parts.append(f"{key} = ?")
                params.append(counts[key])
        if status == "confirmed":
            set_parts.append("confirmed_at = CURRENT_TIMESTAMP")
        params.append(batch_id)
        cur = self._conn.execute(
            f"UPDATE import_batches SET {', '.join(set_parts)} WHERE id = ?", params
        )
        return cur.rowcount > 0
