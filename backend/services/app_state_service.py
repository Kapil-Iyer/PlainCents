"""
AppStateService (TRD §4.5, §7; Build Plan Phase 2).

Thin service wrapping AppStateRepository. get_mode()/can_load_demo() are
fully functional in Phase 2. maybe_transition_to_real() is a stub here —
its real callers (TransactionService, IngestionService, PortfolioService,
after a durable real write) are introduced starting in Phase 3.
"""
import sqlite3

from backend.repositories.app_state_repository import AppStateRepository


class AppStateService:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._repo = AppStateRepository(conn)

    def get_mode(self) -> str:
        return self._repo.get_mode()

    def can_load_demo(self) -> bool:
        return self.get_mode() == "EMPTY"

    def maybe_transition_to_real(self) -> None:
        """
        TRD §4.5.1 EMPTY -> REAL transition: called immediately after a
        durable real-data write succeeds (manual transaction creation here in
        Phase 3; TD import commit and holding creation are added in later
        phases). Idempotent — a no-op if the mode is already REAL or DEMO.
        Callers must invoke this only after their own write has durably
        succeeded; a failed write must never call this (mode stays EMPTY).
        """
        if self.get_mode() == "EMPTY":
            self._repo.set_mode("REAL")

    def maybe_transition_to_empty(self) -> None:
        """
        The reverse of maybe_transition_to_real(): called after a durable
        real-data DELETE succeeds (a single transaction or holding removed
        one at a time — NOT the bulk "Clear Real Data" action, which already
        flips mode itself in DemoService.clear_real_data()). Without this,
        deleting every real row down to zero one-by-one left `mode` stuck on
        "REAL" forever (a stored flag, never re-derived from the data), which
        made "Load demo data" keep failing with "real data exists" even
        though no real row remained anywhere — confusing, since the fix
        (Clear Real Data) was never actually necessary at that point.

        REAL can be reached via EITHER a transaction or a holding (both call
        maybe_transition_to_real()), so the reverse must check BOTH tables —
        deleting the last transaction while a real holding still exists must
        NOT drop back to EMPTY, or that holding would misleadingly vanish
        from view (GET endpoints filter by the current mode).

        Idempotent — a no-op if mode isn't REAL, or if real data still
        exists in either table.
        """
        if self.get_mode() != "REAL":
            return
        remaining = self._conn.execute(
            "SELECT "
            "  (SELECT COUNT(*) FROM transactions WHERE data_mode = 'real') "
            "  + (SELECT COUNT(*) FROM holdings WHERE data_mode = 'real') AS n"
        ).fetchone()["n"]
        if remaining == 0:
            self._repo.set_mode("EMPTY")
