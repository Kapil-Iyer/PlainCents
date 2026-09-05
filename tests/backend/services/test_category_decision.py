"""
ML-G: the single categorization decision path.

These tests exist because the bug they guard against was not a crash -- it
was two code paths quietly disagreeing. Preview staged the raw model output;
Confirm independently applied structural-ambiguity routing and correction
memory. The category a user saw was therefore not the category that got
stored, on exactly the rows where it mattered most.
"""
from pathlib import Path

import pytest

from backend.repositories.transaction_repository import TransactionRepository
from backend.services.ambiguity import is_structurally_ambiguous
from backend.services.categorization_service import CategorizationService
from backend.services.category_decision import (
    SOURCE_LOW_CONFIDENCE_OTHER,
    SOURCE_MODEL,
    SOURCE_STRUCTURAL_OTHER,
    SYSTEM_OTHER,
    CorrectionMemory,
    decide,
    decide_batch,
)
from backend.services.merchant_identity import stable_merchant_key

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"
TEST_MODEL_PATH = FIXTURES_DIR / "categorizer_model_test.pkl"


@pytest.fixture
def categorization():
    return CategorizationService(TEST_MODEL_PATH)


# -- structural ambiguity -----------------------------------------------------


@pytest.mark.parametrize("merchant", [
    "E-TRANSFER SENT",
    "ETRANSFER SENT",
    "FREE INTERAC E-TRANSFER",
    "INTERAC E-TRANSFER SENT",
    "ABM WITHDRAWAL",
    "abm withdrawal",
    "ATM WITHDRAWAL",
    "ATM WITHDRAWAL 8821",
    "CASH WITHDRAWAL",
    "ONLINE BANKING TRANSFER",
    "ONLINE TRANSFER TO DEPOSIT ACCOUNT",
    "TRANSFER TO SAVINGS ACCOUNT",
    "PREAUTH PYMT",
    "PREAUTH PYMT 774120",
    "MISC DEBIT TRANSACTION",
])
def test_text_that_names_nothing_routes_to_other(merchant, categorization):
    decision = decide(merchant, "RBC", categorization)

    assert decision.predicted_category == SYSTEM_OTHER
    assert decision.source == SOURCE_STRUCTURAL_OTHER
    # A SYSTEM decision, never a human one.
    assert decision.confirmed_category is None
    # And it yields no memory identity, so unrelated transfers can never
    # collapse into one shared correction-memory entry.
    assert decision.merchant_key is None


@pytest.mark.parametrize("merchant", [
    "E-TRANSFER SENT MAPLEWOOD DINER REF44120",
    "E-TRANSFER SENT TO SUMMIT PROPERTY MGMT RENT",
    "WIRE TRANSFER SERVICE FEE",
    "INTERAC ACCESS FEE",
    "BRIGHTWAVE INTERNET PREAUTH PYMT 858490",
    "VISA DEBIT PURCHASE - 9947 CEDAR GROCERS",
    "CAREWELL PHARMACY 0284",
])
def test_payment_boilerplate_with_a_real_identity_stays_ml_eligible(merchant, categorization):
    """ML-G over-routing fix.

    The previous rule matched a bare e-transfer/withdrawal regex anywhere in
    the text, so boilerplate wrapped around a usable merchant identity was
    routed to Other and never reached the classifier at all. Measured on the
    ML-G benchmark's FINAL_TEST partition it fired on 27 of 195 (13.8%) of
    legitimate rows.
    """
    assert is_structurally_ambiguous(merchant) is False
    decision = decide(merchant, "RBC", categorization)
    assert decision.source != SOURCE_STRUCTURAL_OTHER
    assert decision.merchant_key is not None


def test_recognizable_merchant_is_served_by_the_model(categorization):
    decision = decide("VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY", "RBC", categorization)

    assert decision.source == SOURCE_MODEL
    assert decision.predicted_category == "Healthcare"
    assert decision.n_active_features > 0


def test_abstention_is_a_system_other_not_a_confirmation(categorization):
    """A row the model cannot separate gets Other as a SYSTEM decision.

    This is what replaces the old behaviour where an evidence-free row got
    argmax(intercept_) -- a confident-looking answer with nothing behind it.
    """
    # Force the abstention branch deterministically rather than relying on a
    # particular string staying below threshold as the fixture evolves.
    categorization.min_margin = 1.1  # no margin can ever reach this
    decision = decide("SOME UNSEEN BRAND", "RBC", categorization)

    assert decision.predicted_category == SYSTEM_OTHER
    assert decision.source == SOURCE_LOW_CONFIDENCE_OTHER
    assert decision.confirmed_category is None
    # The classifier's own answer is preserved for auditability.
    assert decision.model_category is not None


# -- correction memory --------------------------------------------------------


def _insert(conn, merchant, bank, predicted, confirmed=None, dedup="k"):
    repo = TransactionRepository(conn)
    return repo.create({
        "date": "2026-01-05",
        "merchant": merchant,
        "amount": 10.0,
        "bank_source": bank,
        "predicted_category": predicted,
        "confirmed_category": confirmed,
        "data_mode": "real",
        "dedup_key": dedup,
    })


def test_remembered_correction_applies_across_card_suffix_variants(conn, categorization):
    """The bug this fixes: a real bank embeds a different card suffix on
    every transaction, so exact-string memory essentially never fired."""
    _insert(conn, "VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY", "RBC",
            "Healthcare", confirmed="Shopping", dedup="a")
    conn.commit()

    memory = CorrectionMemory(TransactionRepository(conn))
    decision = decide("VISA DEBIT PURCHASE - 9137 CAREWELL PHARMACY", "RBC",
                      categorization, memory)

    assert decision.confirmed_category == "Shopping"
    assert decision.effective_category == "Shopping"
    # predicted_category is preserved untouched -- system and human stay
    # separately auditable for the life of the row.
    assert decision.predicted_category == "Healthcare"
    assert decision.is_remembered_correction is True


def test_correction_memory_is_isolated_per_bank(conn, categorization):
    _insert(conn, "CAREWELL PHARMACY", "RBC", "Healthcare", confirmed="Shopping", dedup="a")
    conn.commit()

    memory = CorrectionMemory(TransactionRepository(conn))
    same_bank = decide("CAREWELL PHARMACY 0284", "RBC", categorization, memory)
    other_bank = decide("CAREWELL PHARMACY 0284", "Scotiabank", categorization, memory)

    assert same_bank.confirmed_category == "Shopping"
    assert other_bank.confirmed_category is None


def test_system_other_never_seeds_correction_memory(conn, categorization):
    """An auto-routed row writes predicted_category and leaves
    confirmed_category NULL, so it can never be mistaken for -- or
    propagated as -- a human decision."""
    _insert(conn, "E-TRANSFER SENT", "RBC", "Other", confirmed=None, dedup="a")
    conn.commit()

    memory = CorrectionMemory(TransactionRepository(conn))
    decision = decide("E-TRANSFER SENT", "RBC", categorization, memory)

    assert decision.confirmed_category is None
    assert memory.lookup(stable_merchant_key("E-TRANSFER SENT", "RBC")) is None


def test_generic_transfers_do_not_collide_in_memory(conn, categorization):
    """Two unrelated generic transfers must not share a memory entry.

    If they did, correcting one e-transfer would teach the system a category
    for "transfers in general" and silently re-label every future one.
    """
    keys = {
        stable_merchant_key(m, "RBC")
        for m in ("E-TRANSFER SENT", "ABM WITHDRAWAL", "ONLINE BANKING TRANSFER",
                  "PREAUTH PYMT 774120", "TRANSFER TO SAVINGS ACCOUNT")
    }
    assert keys == {None}


def test_distinct_merchants_get_distinct_keys():
    a = stable_merchant_key("VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY", "RBC")
    b = stable_merchant_key("VISA DEBIT PURCHASE - 4821 CEDARVALE PHARMACY", "RBC")
    assert a is not None and b is not None and a != b


# -- batch vs single ----------------------------------------------------------


def test_decide_batch_matches_decide_row_by_row(conn, categorization):
    """decide_batch exists only to make one vectorize+predict call instead of
    N. If it ever produced a different answer than decide(), the Preview /
    Confirm agreement this module guarantees would be false again."""
    _insert(conn, "CAREWELL PHARMACY", "RBC", "Healthcare", confirmed="Shopping", dedup="a")
    conn.commit()
    memory = CorrectionMemory(TransactionRepository(conn))

    merchants = [
        "VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY",
        "E-TRANSFER SENT",
        "NORTHSIDE PIZZA #0042",
        "ABM WITHDRAWAL",
        "BRIGHTWAVE INTERNET PREAUTH PYMT 858490",
        "E-TRANSFER SENT MAPLEWOOD DINER REF44120",
    ]
    rows = [(m, "RBC") for m in merchants]

    batch = decide_batch(rows, categorization, memory)
    singles = [decide(m, "RBC", categorization, memory) for m in merchants]

    assert len(batch) == len(singles)
    for b, s in zip(batch, singles):
        assert b == s
