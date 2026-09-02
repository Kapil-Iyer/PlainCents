"""
TransactionRepository — persistence only (TRD §8). No business decisions,
no ML calls, no forecast-staleness triggering, no EMPTY→REAL transitioning:
those are service-layer concerns (Phase 3+).
"""
import sqlite3

from backend.repositories.money import round_money

_SELECT_EFFECTIVE = "SELECT * FROM v_transactions_effective"


class TransactionRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def create(self, data: dict) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO transactions
                (date, raw_description, merchant, amount, bank_source,
                 predicted_category, confirmed_category, import_batch_id,
                 data_mode, dedup_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["date"],
                data.get("raw_description"),
                data["merchant"],
                round_money(data["amount"]),
                data.get("bank_source"),
                data["predicted_category"],
                data.get("confirmed_category"),
                data.get("import_batch_id"),
                data["data_mode"],
                data["dedup_key"],
            ),
        )
        return cur.lastrowid

    def get(self, transaction_id: int) -> dict | None:
        row = self._conn.execute(
            f"{_SELECT_EFFECTIVE} WHERE id = ?", (transaction_id,)
        ).fetchone()
        return dict(row) if row else None

    def list(
        self,
        data_mode: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        category: str | None = None,
        search: str | None = None,
        sort: str = "date",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list = []

        if data_mode is not None:
            clauses.append("data_mode = ?")
            params.append(data_mode)
        if date_from is not None:
            clauses.append("date >= ?")
            params.append(date_from)
        if date_to is not None:
            clauses.append("date <= ?")
            params.append(date_to)
        if category is not None:
            clauses.append("effective_category = ?")
            params.append(category)
        if search is not None:
            clauses.append("merchant LIKE ?")
            params.append(f"%{search}%")

        sql = _SELECT_EFFECTIVE
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        sort_column = sort.lstrip("-")
        direction = "DESC" if sort.startswith("-") else "ASC"
        allowed_sorts = {"date", "amount", "merchant", "created_at"}
        if sort_column not in allowed_sorts:
            sort_column = "date"
        sql += f" ORDER BY {sort_column} {direction}"

        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def update(self, transaction_id: int, fields: dict) -> bool:
        if not fields:
            return False
        allowed = {"date", "merchant", "amount", "confirmed_category", "raw_description"}
        set_parts = []
        params = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "amount":
                value = round_money(value)
            set_parts.append(f"{key} = ?")
            params.append(value)
        if not set_parts:
            return False
        set_parts.append("updated_at = CURRENT_TIMESTAMP")
        params.append(transaction_id)
        cur = self._conn.execute(
            f"UPDATE transactions SET {', '.join(set_parts)} WHERE id = ?", params
        )
        return cur.rowcount > 0

    def delete(self, transaction_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
        return cur.rowcount > 0

    def delete_by_data_mode(self, data_mode: str) -> int:
        """Bulk delete, scoped by data_mode (Build Plan Phase 9 —
        DemoService.clear_demo()'s full-reset deletion). Never used with a
        None data_mode; that would delete every transaction regardless of
        mode, which is not a case this method is meant to support."""
        cur = self._conn.execute("DELETE FROM transactions WHERE data_mode = ?", (data_mode,))
        return cur.rowcount

    def exists_by_dedup_key(self, dedup_key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM transactions WHERE dedup_key = ?", (dedup_key,)
        ).fetchone()
        return row is not None

    def count_distinct_months(self, data_mode: str | None = None) -> int:
        sql = "SELECT COUNT(DISTINCT substr(date, 1, 7)) FROM transactions"
        params: list = []
        if data_mode is not None:
            sql += " WHERE data_mode = ?"
            params.append(data_mode)
        row = self._conn.execute(sql, params).fetchone()
        return row[0] if row else 0

    def aggregate_by_month_category(
        self, data_mode: str | None = None, date_from: str | None = None, date_to: str | None = None
    ) -> list[dict]:
        clauses = []
        params: list = []
        if data_mode is not None:
            clauses.append("data_mode = ?")
            params.append(data_mode)
        if date_from is not None:
            clauses.append("date >= ?")
            params.append(date_from)
        if date_to is not None:
            clauses.append("date <= ?")
            params.append(date_to)

        sql = (
            "SELECT substr(date, 1, 7) AS month, effective_category AS category, "
            "SUM(amount) AS total_spend "
            "FROM v_transactions_effective"
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " GROUP BY month, category ORDER BY month, category"

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
