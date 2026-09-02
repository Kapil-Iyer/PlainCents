"""Phase 1 test: unit-of-work rollback. Requirement #14."""
import pytest

from backend.db.unit_of_work import transaction
from backend.repositories.holding_repository import HoldingRepository
from backend.repositories.transaction_repository import TransactionRepository


def test_unit_of_work_commits_on_success(conn):
    txn_repo = TransactionRepository(conn)
    holding_repo = HoldingRepository(conn)

    with transaction(conn):
        txn_repo.create({
            "date": "2026-01-05", "merchant": "TIM HORTONS", "amount": 6.75,
            "predicted_category": "Food & Dining", "data_mode": "real", "dedup_key": "uow-1",
        })
        holding_repo.create({"ticker": "AAPL", "shares": 10, "avg_cost": 150.0, "data_mode": "real"})

    assert len(txn_repo.list()) == 1
    assert len(holding_repo.list()) == 1


def test_unit_of_work_rolls_back_fully_on_forced_failure(conn):
    """
    Simulates a multi-step operation (e.g., import confirm: insert several
    transactions, then a later step fails) — the entire unit must leave no
    partial write, not just the failing statement.
    """
    txn_repo = TransactionRepository(conn)
    holding_repo = HoldingRepository(conn)

    with pytest.raises(RuntimeError):
        with transaction(conn):
            txn_repo.create({
                "date": "2026-01-05", "merchant": "TIM HORTONS", "amount": 6.75,
                "predicted_category": "Food & Dining", "data_mode": "real", "dedup_key": "uow-2",
            })
            holding_repo.create({"ticker": "AAPL", "shares": 10, "avg_cost": 150.0, "data_mode": "real"})
            raise RuntimeError("forced failure mid-transaction")

    # Neither the transaction row nor the holding row survived the rollback,
    # even though the transaction insert happened before the forced failure.
    assert txn_repo.list() == []
    assert holding_repo.list() == []
