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


def test_12b_decision_source_persists_and_survives_a_correction(conn):
    """Migration 005: decision_source is stored, comes back on read, and is
    NEVER touched by a later human correction (TransactionRepository.update's
    `allowed` set does not include it) -- it explains the SYSTEM's original
    reasoning, not something a correction should overwrite."""
    repo = TransactionRepository(conn)
    tid = repo.create({**SAMPLE, "dedup_key": "dk-decision-source", "decision_source": "gazetteer"})
    conn.commit()

    fetched = repo.get(tid)
    assert fetched["decision_source"] == "gazetteer"

    repo.update(tid, {"confirmed_category": "Shopping"})
    conn.commit()
    fetched = repo.get(tid)
    assert fetched["decision_source"] == "gazetteer"  # unchanged by the correction
    assert fetched["confirmed_category"] == "Shopping"


def test_12c_decision_source_defaults_to_none_for_manual_entries(conn):
    """A caller that doesn't supply decision_source (e.g. TransactionService.
    create_manual(), which never runs decide()/decide_batch()) gets NULL, not
    a fabricated reason."""
    repo = TransactionRepository(conn)
    tid = repo.create({**SAMPLE, "dedup_key": "dk-no-decision-source"})
    conn.commit()

    fetched = repo.get(tid)
    assert fetched["decision_source"] is None


def test_12h_model_category_persists_and_survives_a_correction(conn):
    """Migration 006: model_category is stored, comes back on read, and is
    NEVER touched by a later human correction -- same "frozen at decide-time,
    advisory only" rule as decision_source. Accepting the suggestion happens
    through confirmed_category (the normal correction path), never by this
    column changing."""
    repo = TransactionRepository(conn)
    tid = repo.create({
        **SAMPLE, "dedup_key": "dk-model-category",
        "decision_source": "low_confidence_other", "model_category": "Subscriptions",
    })
    conn.commit()

    fetched = repo.get(tid)
    assert fetched["model_category"] == "Subscriptions"
    assert fetched["predicted_category"] == "Food & Dining"  # SAMPLE's own predicted_category, unaffected

    repo.update(tid, {"confirmed_category": "Subscriptions"})
    conn.commit()
    fetched = repo.get(tid)
    assert fetched["model_category"] == "Subscriptions"  # unchanged by the correction
    assert fetched["confirmed_category"] == "Subscriptions"
    assert fetched["effective_category"] == "Subscriptions"


def test_12i_model_category_defaults_to_none_for_manual_entries(conn):
    """A caller that doesn't supply model_category (e.g. a manual entry, or
    a structural/ambiguous-e-transfer row where the model is never called)
    gets NULL, not a fabricated opinion."""
    repo = TransactionRepository(conn)
    tid = repo.create({**SAMPLE, "dedup_key": "dk-no-model-category"})
    conn.commit()

    fetched = repo.get(tid)
    assert fetched["model_category"] is None


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


# -- ML-F correction memory: find_latest_confirmed_category -----------------


def _txn(**overrides) -> dict:
    base = dict(SAMPLE, bank_source="RBC")
    base.update(overrides)
    return base


def test_find_latest_confirmed_category_exact_recurring_merchant_match(conn):
    repo = TransactionRepository(conn)
    tid = repo.create(_txn(dedup_key="k1", merchant="ACME SUB SERVICE"))
    conn.commit()
    repo.update(tid, {"confirmed_category": "Subscriptions"})
    conn.commit()

    found = repo.find_latest_confirmed_category("ACME SUB SERVICE", "RBC")
    assert found == "Subscriptions"


def test_find_latest_confirmed_category_no_match_returns_none(conn):
    repo = TransactionRepository(conn)
    assert repo.find_latest_confirmed_category("NEVER SEEN MERCHANT", "RBC") is None


def test_find_latest_confirmed_category_different_bank_source_does_not_collide(conn):
    repo = TransactionRepository(conn)
    tid = repo.create(_txn(dedup_key="k2", merchant="ACME SUB SERVICE", bank_source="RBC"))
    conn.commit()
    repo.update(tid, {"confirmed_category": "Subscriptions"})
    conn.commit()

    # Same merchant text, different bank -- must not reuse RBC's correction.
    assert repo.find_latest_confirmed_category("ACME SUB SERVICE", "Scotiabank") is None


def test_find_latest_confirmed_category_different_merchant_does_not_collide(conn):
    repo = TransactionRepository(conn)
    tid = repo.create(_txn(dedup_key="k3", merchant="ACME SUB SERVICE"))
    conn.commit()
    repo.update(tid, {"confirmed_category": "Subscriptions"})
    conn.commit()

    assert repo.find_latest_confirmed_category("OTHER MERCHANT", "RBC") is None


def test_find_latest_confirmed_category_newest_correction_wins(conn):
    repo = TransactionRepository(conn)
    t1 = repo.create(_txn(dedup_key="k4", merchant="ACME SUB SERVICE"))
    conn.commit()
    repo.update(t1, {"confirmed_category": "Subscriptions"})
    conn.commit()

    time.sleep(1.1)  # SQLite CURRENT_TIMESTAMP has 1-second resolution
    t2 = repo.create(_txn(dedup_key="k5", merchant="ACME SUB SERVICE"))
    conn.commit()
    repo.update(t2, {"confirmed_category": "Entertainment"})
    conn.commit()

    assert repo.find_latest_confirmed_category("ACME SUB SERVICE", "RBC") == "Entertainment"


def test_find_latest_confirmed_category_ignores_unconfirmed_rows(conn):
    repo = TransactionRepository(conn)
    # A row with only a predicted_category (never manually confirmed) must
    # never be treated as a remembered correction.
    repo.create(_txn(dedup_key="k6", merchant="ACME SUB SERVICE"))
    conn.commit()

    assert repo.find_latest_confirmed_category("ACME SUB SERVICE", "RBC") is None


# -- list_distinct_months (backs the analysis-month selector) ----------------


def test_list_distinct_months_returns_newest_first_deduplicated(conn):
    repo = TransactionRepository(conn)
    repo.create(_txn(dedup_key="m1", date="2026-06-05"))
    repo.create(_txn(dedup_key="m2", date="2026-06-20"))  # same month as m1 -- not a duplicate row
    repo.create(_txn(dedup_key="m3", date="2026-04-01"))
    repo.create(_txn(dedup_key="m4", date="2026-08-15"))
    conn.commit()

    assert repo.list_distinct_months() == ["2026-08", "2026-06", "2026-04"]


def test_list_distinct_months_filters_by_data_mode(conn):
    repo = TransactionRepository(conn)
    repo.create(_txn(dedup_key="m1", date="2026-06-05", data_mode="real"))
    repo.create(_txn(dedup_key="m2", date="2026-07-05", data_mode="demo"))
    conn.commit()

    assert repo.list_distinct_months(data_mode="real") == ["2026-06"]
    assert repo.list_distinct_months(data_mode="demo") == ["2026-07"]


def test_list_distinct_months_empty_database_returns_empty_list(conn):
    repo = TransactionRepository(conn)

    assert repo.list_distinct_months() == []
