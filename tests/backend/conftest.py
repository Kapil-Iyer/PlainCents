"""
Shared fixtures for backend tests.

Every test uses an isolated, temporary SQLite database file — never the
developer's real plaincents_v2.db (Build Plan Phase 1 requirement).
"""
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.db.migration_runner import apply_migrations  # noqa: E402

MIGRATIONS_DIR = ROOT / "db" / "migrations"


@pytest.fixture
def db_path(tmp_path) -> Path:
    """A unique temporary DB file path for one test."""
    return tmp_path / "test_plaincents_v2.db"


@pytest.fixture
def conn(db_path) -> sqlite3.Connection:
    """A fresh, fully-migrated connection to an isolated temporary database."""
    # check_same_thread=False: API tests (Phase 2+) exercise this connection
    # from FastAPI's TestClient, which runs sync route handlers in a worker
    # thread. See backend/db/connection.py for the matching production note.
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    apply_migrations(connection, migrations_dir=MIGRATIONS_DIR)
    yield connection
    connection.close()
