"""
TransactionRepository — persistence only (TRD §8). No business decisions,
no ML calls, no forecast-staleness triggering, no EMPTY→REAL transitioning:
those are service-layer concerns (Phase 3+).
"""
from __future__ import annotations

import sqlite3

from backend.repositories.money import round_money
from backend.services.merchant_identity import stable_merchant_key

# Sentinel distinguishing "caller did not supply merchant_key" (derive it)
# from "caller supplied None" (this row genuinely has no merchant identity).
_MISSING = object()

_SELECT_EFFECTIVE = "SELECT * FROM v_transactions_effective"


class TransactionRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @staticmethod
    def _resolve_merchant_key(data: dict) -> str | None:
        """Use the caller's merchant_key when it supplied one (the import
        path computes it as part of the shared decision, so it must not be
        recomputed and risk drifting); otherwise derive it here, so no
        insertion path can create a row correction memory is blind to.
        An explicit None means "this text names nothing" and is honoured."""
        supplied = data.get("merchant_key", _MISSING)
        if supplied is not _MISSING:
            return supplied
        return stable_merchant_key(data["merchant"], data.get("bank_source"))

    def create(self, data: dict) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO transactions
                (date, raw_description, merchant, amount, bank_source,
                 predicted_category, confirmed_category, import_batch_id,
                 data_mode, dedup_key, merchant_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                self._resolve_merchant_key(data),
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
        if "merchant" in fields:
            # The identity correction memory keys on is derived from the
            # merchant text, so editing the text must move the row's memory
            # identity with it -- otherwise a renamed transaction would keep
            # answering lookups for the merchant it used to be.
            existing = self._conn.execute(
                "SELECT bank_source FROM transactions WHERE id = ?", (transaction_id,)
            ).fetchone()
            bank = existing["bank_source"] if existing else None
            set_parts.append("merchant_key = ?")
            params.append(stable_merchant_key(fields["merchant"], bank))
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

    def find_latest_confirmed_category(self, merchant: str, bank_source: str) -> str | None:
        """Correction memory (ML-F brief §14/§15): the smallest-architecture
        lookup a future import can use to reuse a user's own prior manual
        category correction for the same exact (merchant, bank_source)
        identity -- no new table, just a query against the existing
        `transactions` table's own `confirmed_category` column. Exact string
        match only (no fuzzy/substring/semantic matching). If the user has
        confirmed different categories for this exact key over time, the most
        recent one wins (ORDER BY updated_at, then id, DESC) -- never an
        arbitrary or averaged choice. Returns None if no prior confirmation
        exists for this key, in which case the caller leaves
        confirmed_category unset and effective_category falls back to
        predicted_category as today."""
        row = self._conn.execute(
            """
            SELECT confirmed_category FROM transactions
            WHERE merchant = ? AND bank_source = ? AND confirmed_category IS NOT NULL
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (merchant, bank_source),
        ).fetchone()
        return row[0] if row else None

    def find_latest_confirmed_category_by_key(self, merchant_key: str) -> str | None:
        """Correction memory, keyed by STABLE merchant identity.

        Supersedes find_latest_confirmed_category() above, which matched the
        exact `merchant` string and therefore almost never fired on real bank
        data: every transaction carries a different card suffix / store
        number / reference code, so one merchant produced a different key
        every month. merchant_key (backend/services/merchant_identity.py) is
        the same merchant's identity through any payment rail, scoped by
        bank, and NULL whenever the text names nothing -- which is what stops
        unrelated generic transfers from sharing one memory entry.

        confirmed_category is only ever written by a genuine user action (a
        PATCH via TransactionService.update) or by propagation of one, so
        anything this returns traces back to a real human decision. A
        system-generated "Other" never appears here, because that path writes
        predicted_category and leaves confirmed_category NULL.

        Most recent correction wins (updated_at, then id, DESC) -- never an
        arbitrary or averaged choice. Exact key equality only: no fuzzy
        matching, no substring matching, no semantic similarity.
        """
        if not merchant_key:
            return None
        row = self._conn.execute(
            """
            SELECT confirmed_category FROM transactions
            WHERE merchant_key = ? AND confirmed_category IS NOT NULL
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (merchant_key,),
        ).fetchone()
        return row[0] if row else None

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
