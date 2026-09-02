"""ForecastRepository — persistence only (TRD §8). Does not decide when a
forecast becomes stale, only records the flag when told to (TRD §12.4)."""
import sqlite3

from backend.repositories.money import round_money


class ForecastRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def create_run(self, data: dict) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO forecast_runs
                (months_available, months_required, data_mode, model_impl_version)
            VALUES (?, ?, ?, ?)
            """,
            (
                data["months_available"],
                data.get("months_required", 12),
                data["data_mode"],
                data.get("model_impl_version"),
            ),
        )
        return cur.lastrowid

    def save_predictions(self, run_id: int, predictions: list[dict]) -> int:
        if not predictions:
            return 0
        payload = [
            (
                run_id,
                p["category"],
                p["forecast_month"],
                p["month_offset"],
                round_money(p.get("predicted_amount")),
                int(p.get("is_available", True)),
                p.get("unavailable_reason"),
            )
            for p in predictions
        ]
        self._conn.executemany(
            """
            INSERT INTO forecast_predictions
                (forecast_run_id, category, forecast_month, month_offset,
                 predicted_amount, is_available, unavailable_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        return len(payload)

    def get_predictions(self, run_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM forecast_predictions WHERE forecast_run_id = ? "
            "ORDER BY month_offset, category",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_run(self, data_mode: str | None = None) -> dict | None:
        # Tie-break on id DESC: SQLite's CURRENT_TIMESTAMP has 1-second
        # resolution, so two runs created within the same second would
        # otherwise sort ambiguously. AUTOINCREMENT ids are monotonically
        # increasing and reflect true insertion order, which is exactly
        # "latest" when timestamps tie.
        sql = "SELECT * FROM forecast_runs"
        params: list = []
        if data_mode is not None:
            sql += " WHERE data_mode = ?"
            params.append(data_mode)
        sql += " ORDER BY generated_at DESC, id DESC LIMIT 1"
        row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def get_run(self, run_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM forecast_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def mark_run_stale(self, run_id: int, reason: str | None = None) -> bool:
        cur = self._conn.execute(
            "UPDATE forecast_runs SET is_stale = 1, stale_reason = ? WHERE id = ?",
            (reason, run_id),
        )
        return cur.rowcount > 0

    def delete_runs_by_data_mode(self, data_mode: str) -> int:
        """Bulk delete, scoped by data_mode (Build Plan Phase 9 —
        DemoService.clear_demo()'s full-reset deletion). forecast_predictions
        rows cascade via their FK's ON DELETE CASCADE (every V2 connection
        has PRAGMA foreign_keys = ON — backend/db/connection.py), so no
        separate predictions-deletion call is needed."""
        cur = self._conn.execute("DELETE FROM forecast_runs WHERE data_mode = ?", (data_mode,))
        return cur.rowcount
