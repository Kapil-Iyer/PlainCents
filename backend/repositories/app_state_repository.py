"""AppStateRepository — persistence only (TRD §8, §4.5). Does not decide
WHEN to transition mode — that decision belongs to AppStateService
(Phase 2+); this repository only reads/writes the single-row app_state
table it's told to."""
import sqlite3


class AppStateRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def get_mode(self) -> str:
        row = self._conn.execute("SELECT mode FROM app_state WHERE id = 1").fetchone()
        return row["mode"] if row else "EMPTY"

    def set_mode(self, mode: str) -> None:
        if mode not in ("EMPTY", "DEMO", "REAL"):
            raise ValueError(f"Invalid app_state mode: {mode!r}")
        self._conn.execute(
            "UPDATE app_state SET mode = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
            (mode,),
        )
