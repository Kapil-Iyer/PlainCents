"""
V2 SQLite connection helper.

V2 uses its own database file, separate from V1's plaincents.db (TRD §18.2).
Every connection returned by this module has PRAGMA foreign_keys = ON set
explicitly — SQLite does not persist this setting across connections, so it
must never be assumed (TRD §4, item 4).
"""
import sqlite3
from pathlib import Path

from backend.db.migration_runner import MIGRATIONS_DIR, apply_migrations

DEFAULT_V2_DB_PATH = Path(__file__).resolve().parent.parent.parent / "plaincents_v2.db"


def get_connection(
    db_path: Path | str = DEFAULT_V2_DB_PATH,
    migrations_dir: Path = MIGRATIONS_DIR,
    run_migrations: bool = True,
) -> sqlite3.Connection:
    """
    Open (creating if needed) the V2 SQLite database at db_path, enable
    foreign key enforcement, apply any pending migrations, and return the
    connection.

    Parameters
    ----------
    db_path : Path or str
        Location of the V2 database file. Defaults to plaincents_v2.db at the
        repository root. Tests should pass an isolated temporary path here —
        never point automated tests at a developer's real plaincents_v2.db.
    migrations_dir : Path
        Directory containing numbered NNN_*.sql migration files.
    run_migrations : bool
        If True (default), pending migrations are applied on open. Callers
        that only need a raw connection to an already-migrated DB may pass
        False to skip the (idempotent, but non-free) migration check.
    """
    # check_same_thread=False: FastAPI (Phase 2+) runs sync route handlers in
    # a worker threadpool, while this connection is opened once in the
    # lifespan hook's thread and reused across requests. This is a single
    # local desktop process with no concurrent writers (TRD §1.8), so sharing
    # one connection across threads sequentially is safe.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if run_migrations:
        apply_migrations(conn, migrations_dir=migrations_dir)
    return conn
