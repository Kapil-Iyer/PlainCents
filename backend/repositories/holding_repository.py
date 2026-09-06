"""HoldingRepository — persistence only (TRD §8)."""
from __future__ import annotations

import sqlite3

from backend.repositories.money import round_money


class HoldingRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def create(self, data: dict) -> int:
        # avg_cost is optional (PRODUCT DECISION: a user may track "10 MSFT
        # shares" without knowing their cost basis yet) -- `.get()`, never
        # `data["avg_cost"]`, so omitting the key is not a KeyError, and
        # round_money(None) already returns None, which binds to SQL NULL.
        cur = self._conn.execute(
            "INSERT INTO holdings (ticker, shares, avg_cost, data_mode) VALUES (?, ?, ?, ?)",
            (data["ticker"], data["shares"], round_money(data.get("avg_cost")), data["data_mode"]),
        )
        return cur.lastrowid

    def get(self, holding_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM holdings WHERE id = ?", (holding_id,)
        ).fetchone()
        return dict(row) if row else None

    def list(self, data_mode: str | None = None) -> list[dict]:
        sql = "SELECT * FROM holdings"
        params: list = []
        if data_mode is not None:
            sql += " WHERE data_mode = ?"
            params.append(data_mode)
        sql += " ORDER BY ticker"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def update(self, holding_id: int, fields: dict) -> bool:
        # avg_cost may be explicitly set to None here (clearing a
        # previously-known cost basis, e.g. via the API's PATCH
        # {"avg_cost": null}) -- round_money(None) is None, which binds to
        # SQL NULL, same as create(). This is a genuine, deliberate "I no
        # longer know/want to record this" state, not a bug.
        allowed = {"shares", "avg_cost"}
        set_parts = []
        params = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "avg_cost":
                value = round_money(value)
            set_parts.append(f"{key} = ?")
            params.append(value)
        if not set_parts:
            return False
        set_parts.append("updated_at = CURRENT_TIMESTAMP")
        params.append(holding_id)
        cur = self._conn.execute(
            f"UPDATE holdings SET {', '.join(set_parts)} WHERE id = ?", params
        )
        return cur.rowcount > 0

    def delete(self, holding_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM holdings WHERE id = ?", (holding_id,))
        return cur.rowcount > 0

    def delete_by_data_mode(self, data_mode: str) -> int:
        """Bulk delete, scoped by data_mode (Build Plan Phase 9 —
        DemoService.clear_demo()'s full-reset deletion). Never used with a
        None data_mode; that would delete every holding regardless of mode,
        which is not a case this method is meant to support."""
        cur = self._conn.execute("DELETE FROM holdings WHERE data_mode = ?", (data_mode,))
        return cur.rowcount
