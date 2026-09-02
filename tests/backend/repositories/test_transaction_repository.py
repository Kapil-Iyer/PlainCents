"""Phase 1 tests: TransactionRepository. Requirements #7, #10, #11, #12, #13."""
import sqlite3
import time

import pytest

from backend.repositories.transaction_repository import TransactionRepository

SAMPLE = {
    "date": "2026-01-05",
    "merchant": "TIM HORTONS",
    "amount": 6.75,
    "predicted_category": "Food & Dining",
    "data_mode": "real",
    "dedup_key": "2026-01-05|6.75|TIM HORTONS|TD|0",
}


def test_7_dedup_key_unique_constraint(conn):
    repo = TransactionRepository(conn)
    repo.create(SAMPLE)
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        repo.create(SAMPLE)  # same dedup_key


def test_10_updated_at_changes_on_update(conn):
    repo = TransactionRepository(conn)
    tid = repo.create(SAMPLE)
    conn.commit()
    before = repo.get(tid)["updated_at"]

    time.sleep(1.1)  # SQLite CURRENT_TIMESTAMP has 1-second resolution
    repo.update(tid, {"amount": 9.99})
    conn.commit()
    after = repo.get(tid)["updated_at"]

    assert after != before


def test_11_updated_at_does_not_change_on_read(conn):
    repo = TransactionRepository(conn)
    tid = repo.create(SAMPLE)
    conn.commit()
    first_read = repo.get(tid)["updated_at"]

    time.sleep(1.1)
    repo.get(tid)  # read only, no write
    second_read = repo.get(tid)["updated_at"]

    assert first_read == second_read


def test_12_crud_round_trip(conn):
    repo = TransactionRepository(conn)
    tid = repo.create(SAMPLE)
    conn.commit()

    fetched = repo.get(tid)
    assert fetched["merchant"] == "TIM HORTONS"
    assert fetched["amount"] == 6.75
    assert fetched["effective_category"] == "Food & Dining"  # no confirmed_category yet
    assert fetched["is_manual_override"] == 0

    ok = repo.update(tid, {"confirmed_category": "Shopping"})
    conn.commit()
    assert ok
    fetched = repo.get(tid)
    assert fetched["predicted_category"] == "Food & Dining"  # never overwritten
    assert fetched["confirmed_category"] == "Shopping"
    assert fetched["effective_category"] == "Shopping"  # confirmed takes precedence
    assert fetched["is_manual_override"] == 1

    ok = repo.delete(tid)
    conn.commit()
    assert ok
    assert repo.get(tid) is None


def test_13_empty_demo_real_read_mapping(conn):
    repo = TransactionRepository(conn)
    demo_row = dict(SAMPLE, data_mode="demo", dedup_key="demo-key-1")
    real_row = dict(SAMPLE, data_mode="real", dedup_key="real-key-1")
    repo.create(demo_row)
    repo.create(real_row)
    conn.commit()

    # EMPTY -> data_mode=None -> no filter (all rows visible; per TRD §4.5.1,
    # a well-behaved app never has rows to see while truly EMPTY, but the
    # repository-level contract is simply "no filter applied")
    assert len(repo.list(data_mode=None)) == 2
    # DEMO -> only demo rows
    demo_results = repo.list(data_mode="demo")
    assert len(demo_results) == 1
    assert demo_results[0]["data_mode"] == "demo"
    # REAL -> only real rows
    real_results = repo.list(data_mode="real")
    assert len(real_results) == 1
    assert real_results[0]["data_mode"] == "real"
