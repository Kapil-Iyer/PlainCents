"""
backend.api.deps.get_db() concurrency regression test.

Discovered during Phase 7 manual browser verification: FastAPI runs sync
route handlers in a threadpool, so concurrent requests race on the single
shared sqlite3.Connection (opened once in main.py's lifespan hook) and
intermittently raise `sqlite3.InterfaceError: bad parameter or other API
misuse`. Reproduced identically against pre-existing Phase 3/6 endpoints
(/api/dashboard/summary, /api/transactions) — a Category B pre-existing
Phase 2 defect, not a Phase 7 regression — and fixed in deps.py (a
process-wide lock serializing access to the shared connection) as an
explicitly authorized exception to Phase 7's declared file scope.

This test drives get_db() directly from many threads (rather than through
the tests/backend/api/conftest.py `client` fixture, whose dependency
override replaces get_db entirely and so would never exercise the real
lock) to prove concurrent access no longer raises and no writes are lost.
"""
import sqlite3
import threading
from types import SimpleNamespace

from backend.api import deps


def _fake_request(conn: sqlite3.Connection):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(db_connection=conn)))


def test_get_db_serializes_concurrent_access_without_sqlite_errors(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "concurrency_test.db"), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)")
    conn.commit()

    errors: list[Exception] = []
    inserts_per_worker = 20
    worker_count = 12

    def worker(n: int) -> None:
        try:
            for yielded_conn in deps.get_db(_fake_request(conn)):
                # Emulate a route handler doing real work with the connection
                # while holding it, the same way a real request would.
                for i in range(inserts_per_worker):
                    yielded_conn.execute("INSERT INTO t (v) VALUES (?)", (n * 1000 + i,))
                yielded_conn.commit()
                yielded_conn.execute("SELECT COUNT(*) FROM t").fetchone()
        except Exception as exc:  # pragma: no cover - failure path under test
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(worker_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent get_db() access raised: {errors}"
    total = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert total == worker_count * inserts_per_worker  # no lost/corrupted writes
    conn.close()
