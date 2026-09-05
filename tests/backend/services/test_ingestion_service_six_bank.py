"""
IngestionService six-bank tests (Phase 12A.5/12B): full preview/confirm
pipeline for RBC, Scotiabank, CIBC via auto-detect and explicit selection,
BLOCKED-bank behavior, and cross-bank dedup (no collision via bank_source).
"""
from pathlib import Path

import pytest

from backend.api.errors import BadRequestError
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.app_state_service import AppStateService
from backend.services.categorization_service import CategorizationService
from backend.services.ingestion_service import IngestionService

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
TEST_MODEL_PATH = FIXTURES_DIR / "categorizer_model_test.pkl"


def _read(subdir: str, name: str) -> bytes:
    return (FIXTURES_DIR / subdir / name).read_bytes()


class FakeForecastService:
    def __init__(self):
        self.calls = []

    def mark_stale(self, reason):
        self.calls.append(reason)


@pytest.fixture
def categorization_service():
    return CategorizationService(TEST_MODEL_PATH)


@pytest.fixture
def forecast_service():
    return FakeForecastService()


@pytest.fixture
def service(conn, categorization_service, forecast_service):
    return IngestionService(
        conn,
        categorization_service,
        app_state_service=AppStateService(conn),
        forecast_service=forecast_service,
    )


@pytest.mark.parametrize(
    "subdir,bank,expected_valid,expected_credit,expected_currency",
    [
        ("rbc_csv", "RBC", 3, 1, 1),
        ("scotia_csv", "Scotiabank", 4, 1, 0),
        ("cibc_csv", "CIBC", 2, 1, 0),
    ],
)
def test_preview_explicit_bank(service, subdir, bank, expected_valid, expected_credit, expected_currency):
    preview = service.parse_and_stage(_read(subdir, "clean_valid.csv"), bank=bank)
    assert preview["detected_bank"] == bank
    assert preview["rows_valid"] == expected_valid
    assert preview["rows_skipped_credit"] == expected_credit
    assert preview["rows_skipped_currency"] == expected_currency
    assert preview["status"] == "previewing"


@pytest.mark.parametrize(
    "subdir,bank",
    [("rbc_csv", "RBC"), ("scotia_csv", "Scotiabank"), ("cibc_csv", "CIBC")],
)
def test_preview_auto_detect_matches_explicit(service, subdir, bank):
    preview = service.parse_and_stage(_read(subdir, "clean_valid.csv"), bank=None)
    assert preview["detected_bank"] == bank


def test_confirm_persists_resolved_bank_source(service, conn):
    preview = service.parse_and_stage(_read("rbc_csv", "clean_valid.csv"), bank=None)
    result = service.commit_import(preview["batch_id"])
    assert result["rows_imported"] == 3
    rows = TransactionRepository(conn).list(data_mode="real")
    assert all(r["bank_source"] == "RBC" for r in rows)
    # Account Number never reached the transactions table.
    assert all("account" not in str(r).lower() for r in rows)


def test_confirm_populates_raw_description(service, conn):
    preview = service.parse_and_stage(_read("scotia_csv", "clean_valid.csv"), bank="Scotiabank")
    service.commit_import(preview["batch_id"])
    rows = TransactionRepository(conn).list(data_mode="real")
    assert any(r["raw_description"] == "GROCERY MART LOYALTY POINTS EARNED" for r in rows)


def test_blocked_bank_raises_bad_request_not_500(service):
    with pytest.raises(BadRequestError, match="not yet supported"):
        service.parse_and_stage(_read("rbc_csv", "clean_valid.csv"), bank="BMO")


def test_cross_bank_same_date_amount_merchant_no_dedup_collision(service, conn):
    # Two different banks' rows sharing (date, amount, merchant) must not be
    # flagged as duplicates of each other -- bank_source is part of the key.
    rbc_preview = service.parse_and_stage(_read("rbc_csv", "clean_valid.csv"), bank="RBC")
    service.commit_import(rbc_preview["batch_id"])

    # A same-shape TD file with a coincidentally identical (date, amount,
    # merchant) is easiest to construct inline rather than a new fixture.
    identical_td_csv = b"Date,Description,Amount\n08/03/2026,TIM HORTONS #123,6.75\n"
    td_preview = service.parse_and_stage(identical_td_csv, bank="TD")
    assert td_preview["rows_duplicate"] == 0  # different bank_source -> not a duplicate

    td_result = service.commit_import(td_preview["batch_id"])
    assert td_result["rows_imported"] == 1

    rows = TransactionRepository(conn).list(data_mode="real")
    bank_sources = {r["bank_source"] for r in rows}
    assert bank_sources == {"RBC", "TD"}
