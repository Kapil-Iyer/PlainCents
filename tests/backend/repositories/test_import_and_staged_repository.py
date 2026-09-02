"""Phase 1 tests: ImportBatchRepository, StagedTransactionRepository, AppStateRepository. Requirement #12."""
from backend.repositories.app_state_repository import AppStateRepository
from backend.repositories.import_batch_repository import ImportBatchRepository
from backend.repositories.staged_transaction_repository import StagedTransactionRepository


def test_import_batch_crud_round_trip(conn):
    repo = ImportBatchRepository(conn)
    batch_id = repo.create_preview(bank_source="TD", original_filename="statement.csv")
    conn.commit()

    batch = repo.get(batch_id)
    assert batch["status"] == "previewing"
    assert batch["bank_source"] == "TD"

    ok = repo.update_status(
        batch_id, "confirmed",
        counts={"rows_valid": 10, "rows_unparseable": 1, "rows_duplicate": 2, "rows_imported": 8},
    )
    conn.commit()
    assert ok
    batch = repo.get(batch_id)
    assert batch["status"] == "confirmed"
    assert batch["rows_imported"] == 8
    assert batch["confirmed_at"] is not None

    assert len(repo.list()) == 1


def test_staged_transaction_bulk_create_and_cleanup(conn):
    import_repo = ImportBatchRepository(conn)
    staged_repo = StagedTransactionRepository(conn)

    batch_id = import_repo.create_preview(bank_source="TD")
    conn.commit()

    rows = [
        {"date": "2026-01-05", "merchant": "TIM HORTONS", "amount": 6.75,
         "predicted_category": "Food & Dining", "dedup_key": "k1", "is_duplicate": False, "is_valid": True},
        {"date": "2026-01-06", "merchant": "SHELL", "amount": 45.30,
         "predicted_category": "Transport", "dedup_key": "k2", "is_duplicate": True, "is_valid": True},
    ]
    n = staged_repo.bulk_create(batch_id, rows)
    conn.commit()
    assert n == 2

    fetched = staged_repo.list_for_batch(batch_id)
    assert len(fetched) == 2
    assert any(r["is_duplicate"] == 1 for r in fetched)

    deleted = staged_repo.delete_for_batch(batch_id)
    conn.commit()
    assert deleted == 2
    assert staged_repo.list_for_batch(batch_id) == []


def test_staged_transactions_cascade_delete_with_batch(conn):
    """ON DELETE CASCADE: deleting an import_batches row removes its staged rows."""
    import_repo = ImportBatchRepository(conn)
    staged_repo = StagedTransactionRepository(conn)

    batch_id = import_repo.create_preview(bank_source="TD")
    conn.commit()
    staged_repo.bulk_create(batch_id, [
        {"date": "2026-01-05", "merchant": "X", "amount": 1.0,
         "predicted_category": "Other", "dedup_key": "kx"},
    ])
    conn.commit()

    conn.execute("DELETE FROM import_batches WHERE id = ?", (batch_id,))
    conn.commit()

    assert staged_repo.list_for_batch(batch_id) == []


def test_app_state_get_and_set_mode(conn):
    repo = AppStateRepository(conn)
    assert repo.get_mode() == "EMPTY"  # default per migration's INSERT OR IGNORE

    repo.set_mode("REAL")
    conn.commit()
    assert repo.get_mode() == "REAL"

    repo.set_mode("DEMO")
    conn.commit()
    assert repo.get_mode() == "DEMO"
