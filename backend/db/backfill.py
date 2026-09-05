"""
Idempotent post-migration backfills that need application logic, not SQL.

Migration 003 adds `transactions.merchant_key`, whose value is derived by
backend/services/merchant_identity.stable_merchant_key(). That derivation
involves boilerplate stripping and token filtering, which is not expressible
in SQLite DDL, so the column lands NULL for pre-existing rows and is filled
in here.

Run once at startup, after migrations. Cheap: the WHERE clause matches
nothing on an already-backfilled database, so the common case is a single
indexed count query.
"""
from __future__ import annotations

import logging
import sqlite3

from backend.services.merchant_identity import stable_merchant_key

logger = logging.getLogger("backend")


def backfill_merchant_keys(conn: sqlite3.Connection) -> int:
    """Populate merchant_key for every transaction still missing one.

    Returns the number of rows updated. Rows whose text carries no merchant
    identity keep merchant_key NULL by design -- and because the query
    filters on `merchant_key IS NULL`, those rows are re-examined on each
    startup. That is intentional and harmless: they are few, the work is a
    pure string transform, and the alternative (a sentinel value) would make
    "no identity" and "not yet computed" indistinguishable in the data.
    """
    rows = conn.execute(
        "SELECT id, merchant, bank_source FROM transactions WHERE merchant_key IS NULL"
    ).fetchall()
    if not rows:
        return 0

    updates = [
        (key, row["id"] if isinstance(row, sqlite3.Row) else row[0])
        for row in rows
        if (key := stable_merchant_key(
            row["merchant"] if isinstance(row, sqlite3.Row) else row[1],
            row["bank_source"] if isinstance(row, sqlite3.Row) else row[2],
        )) is not None
    ]
    if updates:
        conn.executemany("UPDATE transactions SET merchant_key = ? WHERE id = ?", updates)
        conn.commit()
        logger.info("Backfilled merchant_key for %d transaction(s)", len(updates))
    return len(updates)
