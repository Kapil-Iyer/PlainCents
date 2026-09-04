"""
ML-F correction memory + structurally-ambiguous-row routing, integrated at
IngestionService.commit_import() (ML-F-A audit §14-16; ML-F brief §14-16).

Uses a minimal, self-authored TD-shaped CSV (Date,Description,Amount) per
test for full control over merchant text/date/amount, rather than the
shared TD fixtures, so each scenario is unambiguous and self-contained.
"""
from pathlib import Path

import pytest

from backend.repositories.app_state_repository import AppStateRepository
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.ambiguity import is_structurally_ambiguous
from backend.services.app_state_service import AppStateService
from backend.services.categorization_service import CategorizationService
from backend.services.ingestion_service import IngestionService

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
TEST_MODEL_PATH = FIXTURES_DIR / "logreg_model_test.pkl"


def _csv(rows: list[tuple[str, str, str]]) -> bytes:
    lines = ["Date,Description,Amount"] + [f"{d},{desc},{amt}" for d, desc, amt in rows]
    return ("\n".join(lines) + "\n").encode()


class FakeForecastService:
    def mark_stale(self, reason):
        pass


@pytest.fixture
def categorization_service():
    return CategorizationService(TEST_MODEL_PATH)


@pytest.fixture
def service(conn, categorization_service):
    return IngestionService(
        conn,
        categorization_service,
        app_state_service=AppStateService(conn),
        forecast_service=FakeForecastService(),
    )


def _import(service, rows):
    preview = service.parse_and_stage(_csv(rows), bank="TD")
    result = service.commit_import(preview["batch_id"])
    return result


# -- correction memory --------------------------------------------------


def test_recurring_merchant_reuses_prior_confirmed_category(service, conn):
    repo = TransactionRepository(conn)

    _import(service, [("1/5/2026", "ACME SUB SERVICE", "9.99")])
    first = repo.list(data_mode="real")[0]
    original_predicted = first["predicted_category"]
    repo.update(first["id"], {"confirmed_category": "Subscriptions"})
    conn.commit()

    # A second, distinct transaction (different date/amount so it is not a
    # duplicate) from the same merchant + bank.
    _import(service, [("2/5/2026", "ACME SUB SERVICE", "9.99")])
    rows = [r for r in repo.list(data_mode="real") if r["date"] == "2026-02-05"]
    assert len(rows) == 1
    second = rows[0]

    # predicted_category is ALWAYS the raw ML output -- never overwritten by
    # correction memory.
    assert second["predicted_category"] == original_predicted
    # confirmed_category is pre-filled from the remembered correction, and
    # effective_category reflects it.
    assert second["confirmed_category"] == "Subscriptions"
    assert second["effective_category"] == "Subscriptions"


def test_different_bank_source_does_not_reuse_correction(service, conn, categorization_service):
    repo = TransactionRepository(conn)

    _import(service, [("1/5/2026", "ACME SUB SERVICE", "9.99")])
    first = repo.list(data_mode="real")[0]
    assert first["bank_source"] == "TD"
    repo.update(first["id"], {"confirmed_category": "Subscriptions"})
    conn.commit()

    # A different bank's import of the "same" merchant text must not collide.
    from backend.services.ingestion_service import IngestionService as _IS

    scotia_csv = (
        "Filter,Date,Description,Sub-description,Type of Transaction,Amount,Balance\n"
        '"","2026-02-05","ACME SUB SERVICE"," ","Debit","-9.99","100.00"\n'
    ).encode()
    preview = service.parse_and_stage(scotia_csv, bank="Scotiabank")
    service.commit_import(preview["batch_id"])

    scotia_rows = [r for r in repo.list(data_mode="real") if r["bank_source"] == "Scotiabank"]
    assert len(scotia_rows) == 1
    assert scotia_rows[0]["confirmed_category"] is None


def test_different_merchant_does_not_reuse_correction(service, conn):
    repo = TransactionRepository(conn)

    _import(service, [("1/5/2026", "ACME SUB SERVICE", "9.99")])
    first = repo.list(data_mode="real")[0]
    repo.update(first["id"], {"confirmed_category": "Subscriptions"})
    conn.commit()

    _import(service, [("2/5/2026", "COMPLETELY DIFFERENT MERCHANT", "9.99")])
    other = [r for r in repo.list(data_mode="real") if r["merchant"] == "COMPLETELY DIFFERENT MERCHANT"][0]
    assert other["confirmed_category"] is None


def test_newest_correction_wins_on_next_import(service, conn):
    import time

    repo = TransactionRepository(conn)

    _import(service, [("1/5/2026", "ACME SUB SERVICE", "9.99")])
    t1 = repo.list(data_mode="real")[0]
    repo.update(t1["id"], {"confirmed_category": "Subscriptions"})
    conn.commit()

    time.sleep(1.1)  # SQLite CURRENT_TIMESTAMP has 1-second resolution
    _import(service, [("2/5/2026", "ACME SUB SERVICE", "9.99")])
    t2 = [r for r in repo.list(data_mode="real") if r["date"] == "2026-02-05"][0]
    repo.update(t2["id"], {"confirmed_category": "Entertainment"})
    conn.commit()

    _import(service, [("3/5/2026", "ACME SUB SERVICE", "9.99")])
    t3 = [r for r in repo.list(data_mode="real") if r["date"] == "2026-03-05"][0]
    assert t3["confirmed_category"] == "Entertainment"


def test_manual_later_correction_overrides_remembered_value(service, conn):
    repo = TransactionRepository(conn)

    _import(service, [("1/5/2026", "ACME SUB SERVICE", "9.99")])
    t1 = repo.list(data_mode="real")[0]
    repo.update(t1["id"], {"confirmed_category": "Subscriptions"})
    conn.commit()

    _import(service, [("2/5/2026", "ACME SUB SERVICE", "9.99")])
    t2 = [r for r in repo.list(data_mode="real") if r["date"] == "2026-02-05"][0]
    assert t2["confirmed_category"] == "Subscriptions"  # remembered

    # User manually overrides the pre-filled value on t2 itself.
    repo.update(t2["id"], {"confirmed_category": "Entertainment"})
    conn.commit()
    t2_after = repo.get(t2["id"])
    assert t2_after["confirmed_category"] == "Entertainment"
    assert t2_after["predicted_category"] == t2["predicted_category"]  # still untouched


# -- structurally-ambiguous-row routing ----------------------------------


def test_generic_etransfer_routes_to_other():
    assert is_structurally_ambiguous("E-TRANSFER SENT JOHN DOE ABC123") is True
    assert is_structurally_ambiguous("ETRANSFER SENT") is True


def test_generic_abm_atm_routes_to_other():
    assert is_structurally_ambiguous("ABM WITHDRAWAL") is True
    assert is_structurally_ambiguous("ATM WITHDRAWAL") is True


def test_normal_merchant_is_not_ambiguous():
    assert is_structurally_ambiguous("ACME SUB SERVICE") is False
    assert is_structurally_ambiguous("VISA DEBIT PURCHASE - 4521 GROCERY STORE") is False


def test_ambiguous_row_gets_other_confirmed_category_on_import(service, conn):
    repo = TransactionRepository(conn)
    _import(service, [("1/5/2026", "E-TRANSFER SENT JOHN DOE ABC123", "50.00")])
    row = repo.list(data_mode="real")[0]

    assert row["confirmed_category"] is not None
    assert row["confirmed_category"] == "Other"
    assert row["effective_category"] == "Other"
    # predicted_category is preserved as whatever the ML model actually
    # output -- never silently forced to "Other" itself.
    assert row["predicted_category"] is not None


def test_correction_memory_takes_priority_over_ambiguous_routing(service, conn):
    """An ambiguous-shaped merchant string that the user has ALREADY
    manually confirmed to a specific category must keep using that
    correction on a later import, not the generic Other-routing default —
    exact user intent always wins over the generic fallback."""
    repo = TransactionRepository(conn)

    _import(service, [("1/5/2026", "E-TRANSFER SENT JOHN DOE ABC123", "50.00")])
    first = repo.list(data_mode="real")[0]
    repo.update(first["id"], {"confirmed_category": "Rent & Utilities"})
    conn.commit()

    _import(service, [("2/5/2026", "E-TRANSFER SENT JOHN DOE ABC123", "50.00")])
    second = [r for r in repo.list(data_mode="real") if r["date"] == "2026-02-05"][0]
    assert second["confirmed_category"] == "Rent & Utilities"
