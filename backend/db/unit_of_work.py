"""
Generic unit-of-work / explicit-transaction helper (TRD §4, item 5).

Multi-repository operations that must succeed or fail as one unit (import
confirmation, forecast run + predictions persistence, full demo reset) use
this context manager rather than each reimplementing commit/rollback logic.
Commits only on success; rolls back fully on any exception, leaving no
partial write.
"""
import sqlite3
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """
    Usage:
        with transaction(conn):
            repo_a.create(...)
            repo_b.update(...)
        # commits here if no exception was raised; otherwise fully rolled back.
    """
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
