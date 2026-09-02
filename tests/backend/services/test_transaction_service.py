"""TransactionService unit tests (Build Plan Phase 3, item 8): the
staleness-mutation table (merchant edit does NOT mark stale; amount/date/
category/create/delete DO), plus dedup_key + EMPTY->REAL transition
behavior, using a stubbed ForecastService so calls can be asserted.

Also covers the Phase 3 closure patch: the EMPTY->REAL transition and the
transaction insert that triggers it must both survive a connection
close/reopen (they must be committed as one durable unit), and the dedup_key
field order must match TRD §4.4's canonical
date|amount|merchant|bank_source|occurrence_index."""
import sqlite3
from pathlib import Path

import pytest

from backend.api.errors import CategorizationUnavailableError, NotFoundError
from backend.repositories.app_state_repository import AppStateRepository
from backend.services.app_state_service import AppStateService
from backend.services.categorization_service import CategorizationService
from backend.services.dedup import compute_dedup_key
from backend.services.transaction_service import TransactionService

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
TEST_MODEL_PATH = FIXTURES_DIR / "logreg_model_test.pkl"


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
    return TransactionService(
        conn,
        categorization_service,
        app_state_service=AppStateService(conn),
        forecast_service=forecast_service,
    )


def _sample(**overrides):
    data = {"date": "2026-01-15", "merchant": "TIM HORTONS", "amount": 4.50}
    data.update(overrides)
    return data


def test_create_manual_sets_predicted_category_and_real_mode(service, conn):
    row = service.create_manual(_sample())
    assert row["predicted_category"]
    assert row["data_mode"] == "real"
    assert row["import_batch_id"] is None
    assert row["effective_category"] == row["predicted_category"]
    assert row["is_manual_override"] == 0 or row["is_manual_override"] is False


def test_create_manual_transitions_empty_to_real(service, conn):
    assert AppStateRepository(conn).get_mode() == "EMPTY"
    service.create_manual(_sample())
    assert AppStateRepository(conn).get_mode() == "REAL"


def test_create_manual_marks_forecast_stale(service, forecast_service):
    service.create_manual(_sample())
    assert "transaction_created" in forecast_service.calls


def test_create_manual_with_confirmed_category_sets_override(service):
    row = service.create_manual(_sample(confirmed_category="Food & Dining"))
    assert row["confirmed_category"] == "Food & Dining"
    assert row["effective_category"] == "Food & Dining"


def test_create_manual_aborts_and_leaves_mode_empty_when_model_missing(conn, forecast_service, tmp_path):
    missing_service = CategorizationService(tmp_path / "does_not_exist.pkl")
    txn_service = TransactionService(
        conn, missing_service, app_state_service=AppStateService(conn), forecast_service=forecast_service
    )
    with pytest.raises(CategorizationUnavailableError):
        txn_service.create_manual(_sample())

    assert AppStateRepository(conn).get_mode() == "EMPTY"
    assert forecast_service.calls == []


def test_create_manual_persists_across_connection_close_and_reopen(
    db_path, service, conn, categorization_service, forecast_service
):
    """Regression test for the persistence defect found by external audit:
    same-connection reads previously made the mode transition *look*
    committed when it wasn't. Verify durability the only way that actually
    proves it — close the connection and reopen the database file fresh."""
    row = service.create_manual(_sample())
    conn.close()

    reopened = sqlite3.connect(str(db_path))
    reopened.row_factory = sqlite3.Row
    try:
        txn_row = reopened.execute(
            "SELECT * FROM transactions WHERE id = ?", (row["id"],)
        ).fetchone()
        assert txn_row is not None
        assert txn_row["merchant"] == "TIM HORTONS"

        mode_row = reopened.execute("SELECT mode FROM app_state WHERE id = 1").fetchone()
        assert mode_row["mode"] == "REAL"
    finally:
        reopened.close()


class _FailingAppStateService:
    """A fake whose mode-transition step raises, to prove the insert rolls
    back with it rather than being left durably committed on its own while
    app_state silently failed to transition."""

    def get_mode(self):
        return "EMPTY"

    def maybe_transition_to_real(self):
        raise RuntimeError("simulated mode-transition failure")


def test_create_manual_rolls_back_insert_if_mode_transition_fails(
    conn, categorization_service, forecast_service
):
    failing_service = TransactionService(
        conn,
        categorization_service,
        app_state_service=_FailingAppStateService(),
        forecast_service=forecast_service,
    )

    with pytest.raises(RuntimeError):
        failing_service.create_manual(_sample())

    count_row = conn.execute("SELECT COUNT(*) AS n FROM transactions").fetchone()
    assert count_row["n"] == 0
    # The downstream/optional forecast-staleness call must not have run
    # either — the whole create never durably succeeded.
    assert forecast_service.calls == []


def test_dedup_key_uses_canonical_field_order():
    # TRD §4.4: date + amount + merchant + bank_source + occurrence_index.
    key = compute_dedup_key("2026-01-05", 6.75, "TIM HORTONS", "TD", 0)
    assert key == "2026-01-05|6.75|TIM HORTONS|TD|0"


def test_create_manual_duplicate_gets_distinct_dedup_key(service, conn):
    row1 = service.create_manual(_sample())
    row2 = service.create_manual(_sample())
    cur = conn.execute("SELECT dedup_key FROM transactions WHERE id IN (?, ?)", (row1["id"], row2["id"]))
    keys = {r["dedup_key"] for r in cur.fetchall()}
    assert len(keys) == 2


def test_no_transaction_ever_has_null_predicted_category(service, conn):
    service.create_manual(_sample())
    row = conn.execute("SELECT predicted_category FROM transactions").fetchone()
    assert row["predicted_category"] is not None


@pytest.mark.parametrize(
    "fields,should_mark_stale",
    [
        ({"merchant": "STARBUCKS"}, False),
        ({"raw_description": "note"}, False),
        ({"amount": 9.99}, True),
        ({"date": "2026-02-01"}, True),
        ({"confirmed_category": "Shopping"}, True),
    ],
)
def test_update_staleness_mutation_table(service, forecast_service, fields, should_mark_stale):
    row = service.create_manual(_sample())
    forecast_service.calls.clear()  # ignore the create's own mark_stale call

    service.update(row["id"], fields)

    if should_mark_stale:
        assert "transaction_updated" in forecast_service.calls
    else:
        assert forecast_service.calls == []


def test_update_missing_transaction_raises_not_found(service):
    with pytest.raises(NotFoundError):
        service.update(999999, {"amount": 1.0})


def test_update_with_no_fields_raises_bad_request(service):
    row = service.create_manual(_sample())
    from backend.api.errors import BadRequestError

    with pytest.raises(BadRequestError):
        service.update(row["id"], {})


def test_delete_marks_forecast_stale(service, forecast_service):
    row = service.create_manual(_sample())
    forecast_service.calls.clear()

    service.delete(row["id"])

    assert "transaction_deleted" in forecast_service.calls


def test_delete_missing_transaction_raises_not_found(service):
    with pytest.raises(NotFoundError):
        service.delete(999999)


def test_correcting_category_does_not_overwrite_predicted_category(service):
    row = service.create_manual(_sample())
    original_predicted = row["predicted_category"]

    updated = service.update(row["id"], {"confirmed_category": "Healthcare"})

    assert updated["predicted_category"] == original_predicted
    assert updated["confirmed_category"] == "Healthcare"
    assert updated["effective_category"] == "Healthcare"
