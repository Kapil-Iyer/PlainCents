"""
FastAPI dependency injection for the shared DB connection (Build Plan Phase 2,
item 5: "modify backend/db/connection.py to wire into FastAPI DI").

A single connection is opened once in main.py's lifespan hook (migrations run
once at startup) and stored on app.state.db_connection. Each request depends
on get_db(), which yields that same connection — SQLite on a local, single
process, single-user desktop app (TRD §1.8) does not need a per-request
connection pool.
"""
from typing import Iterator

import sqlite3

from fastapi import Request

from backend.services.categorization_service import CategorizationService


def get_db(request: Request) -> Iterator[sqlite3.Connection]:
    yield request.app.state.db_connection


def get_categorization_service(request: Request) -> CategorizationService:
    return request.app.state.categorization_service
