"""StagedTransactionRepository — persistence only (TRD §8)."""
import sqlite3

from backend.repositories.money import round_money


class StagedTransactionRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def bulk_create(self, import_batch_id: int, rows: list[dict]) -> int:
        if not rows:
            return 0
        payload = [
            (
                import_batch_id,
                r["date"],
                r.get("raw_description"),
                r["merchant"],
                round_money(r["amount"]),
                r.get("predicted_category"),
                r["dedup_key"],
                int(r.get("is_duplicate", False)),
                int(r.get("is_valid", True)),
                r.get("invalid_reason"),
            )
            for r in rows
        ]
        self._conn.executemany(
            """
            INSERT INTO staged_transactions
                (import_batch_id, date, raw_description, merchant, amount,
                 predicted_category, dedup_key, is_duplicate, is_valid, invalid_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        return len(payload)

    def list_for_batch(self, import_batch_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM staged_transactions WHERE import_batch_id = ?", (import_batch_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_for_batch(self, import_batch_id: int) -> int:
        cur = self._conn.execute(
            "DELETE FROM staged_transactions WHERE import_batch_id = ?", (import_batch_id,)
        )
        return cur.rowcount
