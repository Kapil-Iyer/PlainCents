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
