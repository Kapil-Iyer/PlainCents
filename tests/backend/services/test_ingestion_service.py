"""
IngestionService tests (Build Plan Phase 4, item 8): TD fixture parsing via
the full preview/confirm pipeline, dedup (cross-batch + idempotent confirm),
model-missing preview-200/confirm-503, atomicity, and DEMO-mode 409.
"""
from pathlib import Path

import pytest

from backend.api.errors import CategorizationUnavailableError, ConflictError, DemoConflictError
from backend.repositories.app_state_repository import AppStateRepository
from backend.repositories.staged_transaction_repository import StagedTransactionRepository
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.app_state_service import AppStateService
from backend.services.categorization_service import CategorizationService
from backend.services.ingestion_service import IngestionService

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
TD_CSV_DIR = FIXTURES_DIR / "td_csv"
RBC_CSV_DIR = FIXTURES_DIR / "rbc_csv"
SCOTIA_CSV_DIR = FIXTURES_DIR / "scotia_csv"
TEST_MODEL_PATH = FIXTURES_DIR / "categorizer_model_test.pkl"


def _read(name: str) -> bytes:
    return (TD_CSV_DIR / name).read_bytes()


def _read_rbc(name: str) -> bytes:
    return (RBC_CSV_DIR / name).read_bytes()


def _read_scotia(name: str) -> bytes:
    return (SCOTIA_CSV_DIR / name).read_bytes()


class FakeForecastService:
    def __init__(self):
        self.calls = []

    def mark_stale(self, reason):
        self.calls.append(reason)


@pytest.fixture
def categorization_service():
    return CategorizationService(TEST_MODEL_PATH)


@pytest.fixture
def missing_categorization_service(tmp_path):
    return CategorizationService(tmp_path / "does_not_exist.pkl")


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


# -- preview / parse_and_stage -----------------------------------------------


def test_preview_clean_valid_fixture(service):
    preview = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    assert preview["rows_valid"] == 12
    assert preview["rows_unparseable"] == 0
    assert preview["rows_duplicate"] == 0
    assert preview["status"] == "previewing"
    assert preview["categorization_available"] is True
    assert len(preview["sample_rows"]) == 10
    assert all(r["predicted_category"] for r in preview["sample_rows"])


def test_preview_unparseable_dates_fixture(service):
    preview = service.parse_and_stage(_read("unparseable_dates.csv"), bank="TD")
    assert preview["rows_valid"] == 4
    assert preview["rows_unparseable"] == 3
    assert preview["rows_duplicate"] == 0


def test_preview_unrecognized_format_raises_bad_request(service):
    from backend.api.errors import BadRequestError

    with pytest.raises(BadRequestError):
        service.parse_and_stage(_read("unrecognized_format.csv"), bank="TD")


def test_preview_duplicate_rows_fixture_preserves_intrafile_occurrences(service):
    # Phase 12B dedup fix: all 8 rows survive parsing (see
    # test_ingest_bytes.py); IngestionService's occurrence_index then lets
    # each repeated (date, amount, merchant) group get distinct dedup keys,
    # so none of them are flagged as duplicates of each other within the
    # same first-ever import.
    preview = service.parse_and_stage(_read("duplicate_rows.csv"), bank="TD")
    assert preview["rows_valid"] == 8
    assert preview["rows_duplicate"] == 0


def test_duplicate_rows_fixture_survive_reimport_via_distinct_occurrence_index(service, conn):
    # The core Phase 12B regression: two (or three) legitimate identical
    # transactions in one file must both/all survive on first import, each
    # getting a distinct occurrence_index, and then both/all be correctly
    # recognized as duplicates -- with the SAME occurrence sequence -- on a
    # second import of the same file.
    first = service.parse_and_stage(_read("duplicate_rows.csv"), bank="TD")
    assert first["rows_valid"] == 8
    assert first["rows_duplicate"] == 0

    result = service.commit_import(first["batch_id"])
    assert result["rows_imported"] == 8

    rows = TransactionRepository(conn).list(data_mode="real")
    dedup_keys = {r["dedup_key"] for r in rows}
    assert len(dedup_keys) == 8  # every occurrence got a distinct key
    occurrence_suffixes = sorted(int(k.rsplit("|", 1)[1]) for k in dedup_keys if k.startswith("2026-01-05|6.75"))
    assert occurrence_suffixes == [0, 1]  # the two identical $6.75 TIM HORTONS rows

    second = service.parse_and_stage(_read("duplicate_rows.csv"), bank="TD")
    assert second["rows_valid"] == 8
    assert second["rows_duplicate"] == 8  # every occurrence reconstructs the same key, all flagged

    second_result = service.commit_import(second["batch_id"])
    assert second_result["rows_imported"] == 0
    assert second_result["rows_skipped_duplicate"] == 8


def test_preview_headerless_td_fixture(service):
    preview = service.parse_and_stage(_read("headerless_positional.csv"), bank="TD")
    assert preview["rows_valid"] == 4
    assert preview["rows_unparseable"] == 0
    assert preview["rows_skipped_credit"] == 1
    assert preview["detected_bank"] == "TD"


def test_preview_does_not_touch_transactions_table(service, conn):
    service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    row = conn.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()
    assert row["n"] == 0


def test_preview_succeeds_200_when_model_missing(conn, missing_categorization_service, forecast_service):
    service = IngestionService(
        conn,
        missing_categorization_service,
        app_state_service=AppStateService(conn),
        forecast_service=forecast_service,
    )
    preview = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    assert preview["categorization_available"] is False
    assert preview["rows_valid"] == 12
    assert all(r["predicted_category"] is None for r in preview["sample_rows"])


def test_preview_raises_demo_conflict_when_mode_is_demo(service, conn):
    AppStateRepository(conn).set_mode("DEMO")
    with pytest.raises(DemoConflictError):
        service.parse_and_stage(_read("clean_valid.csv"), bank="TD")


# -- multi-bank real imports -------------------------------------------------
#
# Product decision: PlainCents supports multiple banks in the same REAL
# dataset (RBC, Scotiabank, TD, CIBC, in any combination, subject to the
# usual per-file parser/dedup rules). There is no "real data is locked to
# one bank" restriction -- see backend/api/errors.py (no BankMismatchError)
# and IngestionService.parse_and_stage() (no established-bank gate).


def test_preview_allows_a_second_file_from_the_same_bank(service, conn):
    """Incremental import: a second file from the SAME bank (e.g. next
    month's statement) succeeds."""
    first = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    service.commit_import(first["batch_id"])

    second = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    assert second["status"] == "previewing"


def test_preview_allows_rbc_then_scotiabank(service, conn):
    first = service.parse_and_stage(_read_rbc("clean_valid.csv"), bank="RBC")
    service.commit_import(first["batch_id"])

    second = service.parse_and_stage(_read_scotia("clean_valid.csv"), bank="Scotiabank")
    assert second["status"] == "previewing"
    assert second["detected_bank"] == "Scotiabank"

    service.commit_import(second["batch_id"])
    bank_sources = {r["bank_source"] for r in TransactionRepository(conn).list(data_mode="real")}
    assert bank_sources == {"RBC", "Scotiabank"}


def test_preview_allows_scotiabank_then_td(service, conn):
    first = service.parse_and_stage(_read_scotia("clean_valid.csv"), bank="Scotiabank")
    service.commit_import(first["batch_id"])

    second = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    assert second["status"] == "previewing"
    assert second["detected_bank"] == "TD"

    service.commit_import(second["batch_id"])
    bank_sources = {r["bank_source"] for r in TransactionRepository(conn).list(data_mode="real")}
    assert bank_sources == {"Scotiabank", "TD"}


def test_preview_allows_first_import_of_any_bank(service, conn):
    preview = service.parse_and_stage(_read_scotia("clean_valid.csv"), bank="Scotiabank")
    assert preview["detected_bank"] == "Scotiabank"


# -- confirm / commit_import -------------------------------------------------


def test_confirm_persists_transactions_with_predicted_category(service, conn):
    preview = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    result = service.commit_import(preview["batch_id"])

    assert result["rows_imported"] == 12
    assert result["status"] == "confirmed"
    rows = TransactionRepository(conn).list(data_mode="real")
    assert len(rows) == 12
    assert all(r["predicted_category"] for r in rows)
    assert all(r["import_batch_id"] == preview["batch_id"] for r in rows)


def test_confirm_persists_decision_source_matching_preview(service, conn):
    """decision_source (migration 005) must survive Confirm exactly as
    Preview staged it -- it was only ever transient (staged_transactions)
    before this migration, so a confirmed transaction had no way to explain
    its own predicted_category after reload. This is the Preview/Confirm/
    reload consistency this feature needs."""
    preview = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    staged = StagedTransactionRepository(conn).list_for_batch(preview["batch_id"])
    staged_sources = {row["merchant"]: row["decision_source"] for row in staged}

    service.commit_import(preview["batch_id"])

    rows = TransactionRepository(conn).list(data_mode="real")
    assert len(rows) == 12
    # Never None/blank -- every row here was decided by the shared path.
    assert all(r["decision_source"] for r in rows)
    for row in rows:
        assert row["decision_source"] == staged_sources[row["merchant"]]
    # At least one row should be gazetteer-served (this fixture's brand
    # names -- TIM HORTONS, SHELL, NETFLIX, SPOTIFY, etc. -- were chosen for
    # an older model-only test, but several now also match
    # backend.services.gazetteer, which is exactly the kind of row this
    # persistence exists to explain later).
    assert any(r["decision_source"] == "gazetteer" for r in rows)


def test_confirm_persists_model_category_for_a_low_confidence_abstention(
    conn, categorization_service, forecast_service
):
    """model_category (migration 006) must survive Confirm for a
    low-confidence abstention row -- it's the advisory "Suggested:
    {model_category}" chip's data source, and it has to outlive Preview the
    same way decision_source does."""
    # Force abstention deterministically (same technique
    # test_category_decision.py uses) rather than relying on a particular
    # string staying below threshold as the fixture model evolves.
    categorization_service.min_margin = 1.1  # no margin can ever reach this
    service = IngestionService(
        conn, categorization_service,
        app_state_service=AppStateService(conn), forecast_service=forecast_service,
    )
    csv = b"Date,Description,Amount\n01/05/2026,SOME UNSEEN BRAND,12.34\n"

    preview = service.parse_and_stage(csv, bank="TD")
    assert preview["sample_rows"][0]["decision_source"] == "low_confidence_other"
    assert preview["sample_rows"][0]["predicted_category"] == "Other"
    staged_model_category = preview["sample_rows"][0]["model_category"]
    assert staged_model_category is not None  # the model still has AN opinion

    service.commit_import(preview["batch_id"])

    row = TransactionRepository(conn).list(data_mode="real")[0]
    assert row["decision_source"] == "low_confidence_other"
    assert row["predicted_category"] == "Other"  # served decision unaffected
    assert row["model_category"] == staged_model_category  # advisory opinion persisted
    assert row["confirmed_category"] is None  # never auto-accepted


def test_confirm_leaves_model_category_null_on_structural_and_e_transfer_rows(service, conn):
    """The model is never called on structural/ambiguous-e-transfer rows
    (CategoryDecision.model_category is already None on those paths) -- that
    must survive Confirm as NULL, not a fabricated value, so the frontend
    correctly never shows a suggestion for them."""
    csv = (
        b"Date,Description,Amount\n"
        b"01/05/2026,ABM WITHDRAWAL,40.00\n"
        b"01/06/2026,E-TRANSFER SENT JANE SMITH,25.00\n"
    )
    preview = service.parse_and_stage(csv, bank="TD")
    sources = {r["decision_source"] for r in preview["sample_rows"]}
    assert sources == {"structural_other", "ambiguous_e_transfer"}
    assert all(r["model_category"] is None for r in preview["sample_rows"])

    service.commit_import(preview["batch_id"])

    rows = TransactionRepository(conn).list(data_mode="real")
    assert len(rows) == 2
    assert all(r["model_category"] is None for r in rows)
    assert {r["decision_source"] for r in rows} == {"structural_other", "ambiguous_e_transfer"}


def test_confirm_transitions_empty_to_real(service, conn):
    preview = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    assert AppStateRepository(conn).get_mode() == "EMPTY"
    service.commit_import(preview["batch_id"])
    assert AppStateRepository(conn).get_mode() == "REAL"


def test_confirm_marks_forecast_stale(service, forecast_service):
    preview = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    service.commit_import(preview["batch_id"])
    assert "import_confirmed" in forecast_service.calls


def test_confirm_cross_batch_dedup_skips_reimport(service, conn):
    first = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    service.commit_import(first["batch_id"])

    second = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    assert second["rows_duplicate"] == 12

    result = service.commit_import(second["batch_id"])
    assert result["rows_imported"] == 0
    assert result["rows_skipped_duplicate"] == 12

    rows = TransactionRepository(conn).list(data_mode="real")
    assert len(rows) == 12  # still just the first import's rows


def test_confirm_is_idempotent_on_double_confirm(service, conn):
    preview = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")
    first_result = service.commit_import(preview["batch_id"])
    second_result = service.commit_import(preview["batch_id"])

    assert first_result == second_result
    rows = TransactionRepository(conn).list(data_mode="real")
    assert len(rows) == 12  # no double-insert


def test_confirm_returns_503_and_commits_nothing_when_model_missing(
    conn, missing_categorization_service, forecast_service
):
    # Stage while the model is available, then swap in a missing service to
    # simulate the model becoming unavailable between preview and confirm.
    available = CategorizationService(TEST_MODEL_PATH)
    staging_service = IngestionService(
        conn, available, app_state_service=AppStateService(conn), forecast_service=forecast_service
    )
    preview = staging_service.parse_and_stage(_read("clean_valid.csv"), bank="TD")

    confirming_service = IngestionService(
        conn,
        missing_categorization_service,
        app_state_service=AppStateService(conn),
        forecast_service=forecast_service,
    )
    with pytest.raises(CategorizationUnavailableError):
        confirming_service.commit_import(preview["batch_id"])

    row = conn.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()
    assert row["n"] == 0
    assert AppStateRepository(conn).get_mode() == "EMPTY"


def test_confirm_missing_batch_raises_not_found(service):
    from backend.api.errors import NotFoundError

    with pytest.raises(NotFoundError):
        service.commit_import(999999)


def test_confirm_atomicity_rolls_back_on_mid_batch_failure(service, conn, monkeypatch):
    preview = service.parse_and_stage(_read("clean_valid.csv"), bank="TD")

    real_create = TransactionRepository.create
    calls = {"n": 0}

    def _flaky_create(self, data):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated mid-batch failure")
        return real_create(self, data)

    monkeypatch.setattr(TransactionRepository, "create", _flaky_create)

    with pytest.raises(RuntimeError):
        service.commit_import(preview["batch_id"])

    # No partial rows: the whole batch's insert attempt rolled back.
    row = conn.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()
    assert row["n"] == 0
    batch = conn.execute(
        "SELECT status FROM import_batches WHERE id = ?", (preview["batch_id"],)
    ).fetchone()
    assert batch["status"] == "previewing"  # never got to "confirmed"
    assert AppStateRepository(conn).get_mode() == "EMPTY"
