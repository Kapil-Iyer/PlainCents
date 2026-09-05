"""
ML-G: Preview must show the decision Confirm will store, and Preview must
change nothing.

The bug: Preview staged the raw model output while Confirm independently
applied structural-ambiguity routing and remembered corrections. So the
category shown in the Preview table was not the category persisted on
Confirm -- specifically on ambiguous rows and on merchants the user had
already corrected, i.e. exactly the rows where being wrong is most
noticeable.
"""
from pathlib import Path

import pytest

from backend.repositories.staged_transaction_repository import StagedTransactionRepository
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.app_state_service import AppStateService
from backend.services.categorization_service import CategorizationService
from backend.services.ingestion_service import IngestionService

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
TEST_MODEL_PATH = FIXTURES_DIR / "categorizer_model_test.pkl"

ROWS = [
    ("1/5/2026", "VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY", "34.82"),
    ("1/6/2026", "E-TRANSFER SENT", "250.00"),
    ("1/7/2026", "ABM WITHDRAWAL", "100.00"),
    ("1/8/2026", "NORTHSIDE PIZZA #0042", "22.40"),
    ("1/9/2026", "BRIGHTWAVE INTERNET PREAUTH PYMT 858490", "86.96"),
    ("1/10/2026", "E-TRANSFER SENT MAPLEWOOD DINER REF44120", "31.00"),
]


def _csv(rows):
    lines = ["Date,Description,Amount"] + [f"{d},{desc},{amt}" for d, desc, amt in rows]
    return ("\n".join(lines) + "\n").encode()


class FakeForecastService:
    def __init__(self):
        self.calls = []

    def mark_stale(self, reason):
        self.calls.append(reason)


@pytest.fixture
def forecast():
    return FakeForecastService()


@pytest.fixture
def service(conn, forecast):
    return IngestionService(
        conn,
        CategorizationService(TEST_MODEL_PATH),
        app_state_service=AppStateService(conn),
        forecast_service=forecast,
    )


def _staged_by_merchant(conn, batch_id):
    return {r["merchant"]: r for r in StagedTransactionRepository(conn).list_for_batch(batch_id)}


def _stored_by_merchant(conn):
    return {r["merchant"]: r for r in TransactionRepository(conn).list(data_mode="real")}


# -- agreement ----------------------------------------------------------------


def test_every_previewed_row_stores_the_category_it_showed(service, conn):
    preview = service.parse_and_stage(_csv(ROWS), bank="TD")
    staged = _staged_by_merchant(conn, preview["batch_id"])

    service.commit_import(preview["batch_id"])
    stored = _stored_by_merchant(conn)

    assert set(staged) == set(stored)
    for merchant, staged_row in staged.items():
        stored_row = stored[merchant]
        assert stored_row["predicted_category"] == staged_row["predicted_category"], merchant
        assert stored_row["confirmed_category"] == staged_row["remembered_category"], merchant
        # The effective category -- what every chart and the forecast use.
        expected_effective = (
            staged_row["remembered_category"] or staged_row["predicted_category"]
        )
        assert stored_row["effective_category"] == expected_effective, merchant


def test_preview_shows_remembered_corrections_before_confirming(service, conn):
    """A user who corrected this merchant before should see their own
    category in the Preview table, not the model's -- otherwise Preview
    misrepresents the import they are about to approve."""
    TransactionRepository(conn).create({
        "date": "2025-12-01",
        "merchant": "CAREWELL PHARMACY",
        "amount": 20.0,
        "bank_source": "TD",
        "predicted_category": "Healthcare",
        "confirmed_category": "Shopping",
        "data_mode": "real",
        "dedup_key": "prior",
    })
    conn.commit()

    preview = service.parse_and_stage(
        _csv([("1/5/2026", "VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY", "34.82")]), bank="TD")
    staged = _staged_by_merchant(conn, preview["batch_id"])
    row = next(iter(staged.values()))

    assert row["remembered_category"] == "Shopping"
    assert row["predicted_category"] == "Healthcare"  # system view preserved

    service.commit_import(preview["batch_id"])
    stored = TransactionRepository(conn).list(data_mode="real")
    imported = [t for t in stored if t["dedup_key"] != "prior"][0]
    assert imported["confirmed_category"] == "Shopping"
    assert imported["predicted_category"] == "Healthcare"
    assert imported["effective_category"] == "Shopping"


def test_preview_records_why_each_category_was_chosen(service, conn):
    preview = service.parse_and_stage(_csv(ROWS), bank="TD")
    staged = _staged_by_merchant(conn, preview["batch_id"])

    assert staged["E-TRANSFER SENT"]["decision_source"] == "structural_other"
    assert staged["E-TRANSFER SENT"]["predicted_category"] == "Other"
    assert staged["ABM WITHDRAWAL"]["decision_source"] == "structural_other"
    assert staged["VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY"]["decision_source"] in (
        "model", "low_confidence_other")


# -- preview is read-only -----------------------------------------------------


def test_preview_creates_no_transactions_and_seeds_no_memory(service, conn, forecast):
    before_mode = AppStateService(conn).get_mode()

    service.parse_and_stage(_csv(ROWS), bank="TD")

    assert TransactionRepository(conn).list(data_mode="real") == []
    # Mode is unchanged: previewing is not importing.
    assert AppStateService(conn).get_mode() == before_mode
    # And nothing downstream was told the world changed.
    assert forecast.calls == []


def test_preview_twice_is_idempotent_for_stored_state(service, conn):
    service.parse_and_stage(_csv(ROWS), bank="TD")
    service.parse_and_stage(_csv(ROWS), bank="TD")

    assert TransactionRepository(conn).list(data_mode="real") == []


# -- confirm re-checks what can genuinely change ------------------------------


def test_correction_made_after_preview_is_honoured_at_confirm(service, conn):
    """Preview is read-only, but it is also not a lock: if the user corrects
    the same merchant between previewing and confirming, the newer human
    decision wins over the one preview captured."""
    preview = service.parse_and_stage(
        _csv([("1/5/2026", "VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY", "34.82")]), bank="TD")

    repo = TransactionRepository(conn)
    repo.create({
        "date": "2025-12-01",
        "merchant": "CAREWELL PHARMACY",
        "amount": 20.0,
        "bank_source": "TD",
        "predicted_category": "Healthcare",
        "confirmed_category": "Entertainment",
        "data_mode": "real",
        "dedup_key": "later-correction",
    })
    conn.commit()

    service.commit_import(preview["batch_id"])
    imported = [t for t in repo.list(data_mode="real") if t["dedup_key"] != "later-correction"][0]
    assert imported["confirmed_category"] == "Entertainment"
