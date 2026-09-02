"""
FastAPI dependency injection for the shared DB connection (Build Plan Phase 2,
item 5: "modify backend/db/connection.py to wire into FastAPI DI").

A single connection is opened once in main.py's lifespan hook (migrations run
once at startup) and stored on app.state.db_connection. Each request depends
on get_db(), which yields that same connection — SQLite on a local, single
process, single-user desktop app (TRD §1.8) does not need a per-request
connection pool.

Concurrency fix (discovered during Phase 7 manual verification; a pre-existing
Phase 2 defect, not a Phase 7 regression — reproduced against Phase 3/6
endpoints identically, fixed here as an explicitly authorized exception to
Phase 7's file scope): FastAPI runs sync route handlers in a threadpool
(Starlette's run_in_threadpool), so two requests arriving close together
execute on DIFFERENT OS threads against the SAME sqlite3.Connection object.
Python's sqlite3 module does not make one connection safe for concurrent
statement execution from multiple threads — check_same_thread=False only
disables the same-thread check, it adds no thread-safety — and concurrent
access reliably produced `sqlite3.InterfaceError: bad parameter or other API
misuse`. _db_lock, held for the full duration of get_db()'s yielded value
(acquired before yield, released only once the route handler using it has
returned), serializes access to the single shared connection without
changing the single-connection design TRD §1.8 already commits to for this
local, single-user app — it makes the existing "no concurrent writers"
assumption actually true instead of merely assumed.
"""
import sqlite3
import threading
from typing import Iterator

from fastapi import Request

from backend.services.categorization_service import CategorizationService

# One process-wide lock guarding the single shared sqlite3.Connection — not
# per-connection state, since there is exactly one shared connection for the
# process's lifetime (main.py's lifespan hook opens it once).
_db_lock = threading.Lock()


def get_db(request: Request) -> Iterator[sqlite3.Connection]:
    with _db_lock:
        yield request.app.state.db_connection


def get_categorization_service(request: Request) -> CategorizationService:
    return request.app.state.categorization_service
