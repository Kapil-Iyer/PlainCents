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
TEST_MODEL_PATH = FIXTURES_DIR / "categorizer_model_test.pkl"


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
    """Correction memory is bank-scoped (merchant_identity.py) -- a
    correction made on one bank's transaction must NOT be reused for the
    "same" merchant text imported from a DIFFERENT bank. Exercised through
    the full import pipeline for both banks (PlainCents supports multiple
    banks in one REAL dataset; a second bank's import is not restricted)."""
    repo = TransactionRepository(conn)

    # Import + confirm a TD transaction, then correct it.
    _import(service, [("1/5/2026", "ACME SUB SERVICE", "9.99")])
    first = repo.list(data_mode="real")[0]
    assert first["bank_source"] == "TD"
    repo.update(first["id"], {"confirmed_category": "Subscriptions"})
    conn.commit()

    # Import the "same" merchant text from a DIFFERENT bank (Scotiabank).
    scotia_csv = (
        "Filter,Date,Description,Sub-description,Type of Transaction,Amount,Balance\n"
        '"","2026-02-05","ACME SUB SERVICE"," ","Debit","-9.99","100.00"\n'
    ).encode()
    preview = service.parse_and_stage(scotia_csv, bank="Scotiabank")
    service.commit_import(preview["batch_id"])

    scotia_rows = [r for r in repo.list(data_mode="real") if r["bank_source"] == "Scotiabank"]
    assert len(scotia_rows) == 1
    # TD's correction must not have leaked onto the Scotiabank row.
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
    """Text that names NOTHING is structurally ambiguous."""
    assert is_structurally_ambiguous("ETRANSFER SENT") is True
    assert is_structurally_ambiguous("E-TRANSFER SENT") is True
    assert is_structurally_ambiguous("FREE INTERAC E-TRANSFER") is True
    assert is_structurally_ambiguous("ONLINE BANKING TRANSFER") is True
    assert is_structurally_ambiguous("PREAUTH PYMT 774120") is True


def test_etransfer_carrying_a_name_is_not_structurally_ambiguous():
    """ML-G over-routing fix.

    The previous rule matched a bare E-?TRANSFER anywhere in the text, so
    payment-method boilerplate wrapped around a perfectly usable merchant
    identity was routed to "Other" and never shown to the classifier. On the
    ML-G benchmark's FINAL_TEST partition that fired on 27 of 195 (13.8%) of
    legitimate rows.

    Structural ambiguity now means "this text names nothing". A row that does
    name something stays ML-eligible — and if the model then cannot place it,
    the abstention policy is what routes it to Other (see the test below).
    Both paths reach predicted_category="Other", confirmed_category=NULL, but
    via the mechanism that is actually true of the row.
    """
    assert is_structurally_ambiguous("E-TRANSFER SENT JOHN DOE ABC123") is False
    assert is_structurally_ambiguous("E-TRANSFER SENT TO SUMMIT PROPERTY MGMT RENT") is False
    assert is_structurally_ambiguous("WIRE TRANSFER SERVICE FEE") is False
    assert is_structurally_ambiguous("INTERAC ACCESS FEE") is False


def test_generic_abm_atm_routes_to_other():
    assert is_structurally_ambiguous("ABM WITHDRAWAL") is True
    assert is_structurally_ambiguous("ATM WITHDRAWAL") is True


def test_etransfer_completion_boilerplate_with_no_name_routes_to_other():
    """Product-semantics fix. "REQUEST"/"FULFILLED"/"RECEIVED"/"AUTODEPOSIT"
    are e-transfer STATUS/mechanism words, never a merchant identity — a row
    that is PURELY this boilerplate plus a reference code (no actual
    recipient name) must route to Other exactly like "ETRANSFER SENT" does,
    not slip past the ambiguity check because those status words looked like
    identity tokens."""
    assert is_structurally_ambiguous("E-TRANSFER REQUEST FULFILLED REF88213") is True
    assert is_structurally_ambiguous("E-TRANSFER - AUTODEPOSIT REF88213") is True
    # A genuine name alongside that same boilerplate must still stay
    # ML-eligible -- unchanged from test_etransfer_carrying_a_name_is_not_
    # structurally_ambiguous above; this is the disclosed limitation, not a
    # regression.
    assert is_structurally_ambiguous("E-TRANSFER REQUEST FULFILLED JOHN DOE REF88213") is False


def test_investment_or_brokerage_language_routes_to_other():
    """Product-semantics fix. Investment/brokerage/registered-account
    vocabulary names a genuine financial event, but not a purchase in any of
    the eight spending categories -- routing it to Other (rather than
    letting the classifier force a confident but meaningless spending-
    category guess) is a disclosed policy choice, not a claim the model
    itself learned this distinction."""
    assert is_structurally_ambiguous("INVESTMENT SAMPLE BROKERAGE INVESTMENTS") is True
    assert is_structurally_ambiguous("RRSP CONTRIBUTION SAMPLE BROKERAGE") is True
    assert is_structurally_ambiguous("TFSA CONTRIBUTION") is True
    # A merchant name that merely CONTAINS "investment"-adjacent words as
    # part of a normal business name is rare enough, and the vocabulary
    # closed/specific enough (INVESTMENT(S), BROKERAGE, SECURITIES, MUTUAL
    # FUND, RRSP, RESP, TFSA), that this is a deliberate, narrow trade-off --
    # not attempting to also catch every possible brokerage provider name.


def test_normal_merchant_is_not_ambiguous():
    assert is_structurally_ambiguous("ACME SUB SERVICE") is False
    assert is_structurally_ambiguous("VISA DEBIT PURCHASE - 4521 GROCERY STORE") is False


def test_ambiguous_row_is_a_system_prediction_not_a_manual_confirmation(service, conn):
    """HITL semantics fix: a structurally-ambiguous row's "Other" routing is
    a SYSTEM decision (predicted_category), never a stand-in for a genuine
    user confirmation (confirmed_category) — nobody looked at this row."""
    repo = TransactionRepository(conn)
    _import(service, [("1/5/2026", "E-TRANSFER SENT JOHN DOE ABC123", "50.00")])
    row = repo.list(data_mode="real")[0]

    assert row["predicted_category"] == "Other"
    assert row["confirmed_category"] is None
    assert row["effective_category"] == "Other"
    assert row["is_manual_override"] == 0


def test_normal_ml_row_has_no_confirmed_category(service, conn):
    """A recoverable (non-ambiguous) merchant is categorized by the model
    alone on first import — confirmed_category is untouched, exactly as
    before ML-F's correction memory/ambiguous-routing additions."""
    repo = TransactionRepository(conn)
    _import(service, [("1/5/2026", "ACME SUB SERVICE", "9.99")])
    row = repo.list(data_mode="real")[0]

    assert row["predicted_category"] is not None
    assert row["predicted_category"] != "Other" or not is_structurally_ambiguous(row["merchant"])
    assert row["confirmed_category"] is None
    assert row["is_manual_override"] == 0


def test_correction_memory_takes_priority_over_ambiguous_routing(service, conn):
    """An ambiguous-shaped merchant string that the user has ALREADY
    manually confirmed to a specific category must keep using that
    correction on a later import, not the generic Other-routing default —
    exact user intent always wins over the generic fallback. The system's
    own predicted_category is still "Other" (its honest, current decision on
    this row shape) — only confirmed_category/effective_category/
    is_manual_override reflect the remembered human correction."""
    repo = TransactionRepository(conn)

    _import(service, [("1/5/2026", "E-TRANSFER SENT JOHN DOE ABC123", "50.00")])
    first = repo.list(data_mode="real")[0]
    assert first["predicted_category"] == "Other"  # the system's own decision, pre-correction
    assert first["confirmed_category"] is None      # nobody has confirmed anything yet
    repo.update(first["id"], {"confirmed_category": "Rent & Utilities"})
    conn.commit()

    _import(service, [("2/5/2026", "E-TRANSFER SENT JOHN DOE ABC123", "50.00")])
    second = [r for r in repo.list(data_mode="real") if r["date"] == "2026-02-05"][0]
    assert second["predicted_category"] == "Other"              # system decision, preserved
    assert second["confirmed_category"] == "Rent & Utilities"   # remembered genuine correction
    assert second["effective_category"] == "Rent & Utilities"
    assert second["is_manual_override"] == 1


def test_auto_routed_ambiguous_row_does_not_seed_correction_memory(service, conn):
    """An ambiguous row that nobody has ever manually confirmed must NOT
    make a later identical import look like it has a remembered human
    correction — confirmed_category stays None on the second import too,
    since the first row's confirmed_category was never set by a real user
    action."""
    repo = TransactionRepository(conn)

    _import(service, [("1/5/2026", "E-TRANSFER SENT JOHN DOE ABC123", "50.00")])
    first = repo.list(data_mode="real")[0]
    assert first["confirmed_category"] is None  # auto-routed, not user-confirmed

    _import(service, [("2/5/2026", "E-TRANSFER SENT JOHN DOE ABC123", "50.00")])
    second = [r for r in repo.list(data_mode="real") if r["date"] == "2026-02-05"][0]
    assert second["confirmed_category"] is None  # nothing genuine to remember yet
    assert second["predicted_category"] == "Other"  # still the system's own honest decision
    assert second["effective_category"] == "Other"
    assert second["is_manual_override"] == 0


def test_explicit_correction_of_an_ambiguous_row_can_later_seed_memory(service, conn):
    """Once a human actually confirms an ambiguous row's category, THAT
    genuine action (not the earlier auto-routing) becomes a valid
    correction-memory source for a future exact matching import."""
    repo = TransactionRepository(conn)

    _import(service, [("1/5/2026", "E-TRANSFER SENT JOHN DOE ABC123", "50.00")])
    first = repo.list(data_mode="real")[0]
    assert first["confirmed_category"] is None

    # A human now genuinely confirms this specific transfer was rent.
    repo.update(first["id"], {"confirmed_category": "Rent & Utilities"})
    conn.commit()

    _import(service, [("2/5/2026", "E-TRANSFER SENT JOHN DOE ABC123", "50.00")])
    second = [r for r in repo.list(data_mode="real") if r["date"] == "2026-02-05"][0]
    assert second["confirmed_category"] == "Rent & Utilities"
    assert second["is_manual_override"] == 1
