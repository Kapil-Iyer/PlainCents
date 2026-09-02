"""
V2 migration runner.

db/migrations/*.sql is the SOLE schema source of truth for V2 (TRD §4.12).
This module applies pending numbered migration files in order, tracking
applied versions in a schema_migrations table, idempotently.
"""
import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "db" / "migrations"

_FILENAME_RE = re.compile(r"^(\d+)_.*\.sql$")

_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def _discover_migrations(migrations_dir: Path) -> list[tuple[int, Path]]:
    """Return [(version, path), ...] sorted by version, for every NNN_*.sql file."""
    found = []
    for path in migrations_dir.glob("*.sql"):
        m = _FILENAME_RE.match(path.name)
        if not m:
            continue
        found.append((int(m.group(1)), path))
    return sorted(found, key=lambda pair: pair[0])


def _highest_applied_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return row[0] if row and row[0] is not None else 0


def apply_migrations(conn: sqlite3.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[int]:
    """
    Apply every pending migration in migrations_dir, in ascending version order,
    each inside its own transaction. Idempotent: re-running applies nothing new
    if all migrations are already recorded in schema_migrations.

    Returns the list of versions actually applied this call (empty if none).
    """
    conn.execute(_SCHEMA_MIGRATIONS_DDL)
    conn.commit()

    current = _highest_applied_version(conn)
    applied_now: list[int] = []

    for version, path in _discover_migrations(migrations_dir):
        if version <= current:
            continue
        sql = path.read_text(encoding="utf-8")
        try:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (?)", (version,)
            )
            conn.commit()
            applied_now.append(version)
            logger.info("Applied migration %03d: %s", version, path.name)
        except Exception:
            conn.rollback()
            logger.error("Migration %03d (%s) failed; rolled back.", version, path.name)
            raise

    return applied_now
