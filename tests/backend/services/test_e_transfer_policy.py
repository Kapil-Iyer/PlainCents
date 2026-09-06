"""
Tests for backend/services/e_transfer_policy.py. All merchant text is
fabricated (fake names like "JANE SMITH", "JOHN DOE") -- never a real
private recipient.
"""
import pytest

from backend.services.e_transfer_policy import has_purpose_evidence, is_purposeless_e_transfer


# -- person-like E-Transfer, no purpose evidence -> True (route to Other) --


@pytest.mark.parametrize("merchant", [
    "E-TRANSFER SENT JANE SMITH",
    "E-TRANSFER RECEIVED JOHN DOE",
    "ETRANSFER SENT ALEX CHEN REF88213",
    "INTERAC E-TRANSFER SENT TO MARIA GARCIA",
    "E-TRANSFER SENT MAPLEWOOD DINER",  # a business name is ALSO purposeless
                                        # under this narrower policy -- see
                                        # test_category_decision.py's
                                        # disclosed-trade-off test for why.
])
def test_person_like_e_transfer_with_no_purpose_evidence_is_purposeless(merchant):
    assert is_purposeless_e_transfer(merchant) is True


# -- E-Transfer WITH genuine purpose evidence -> False (stays eligible) ----


@pytest.mark.parametrize("merchant", [
    "E-TRANSFER SENT JANE SMITH RENT",
    "E-TRANSFER SENT TO LANDLORD",
    "E-TRANSFER SENT JOHN DOE MORTGAGE PAYMENT",
    "E-TRANSFER SENT HYDRO BILL",
    "E-TRANSFER SENT FOR UTILITIES",
    "E-TRANSFER SENT INTERNET BILL",
    "E-TRANSFER SENT DAYCARE FEE",
    "E-TRANSFER SENT INSURANCE PREMIUM",
])
def test_e_transfer_with_purpose_evidence_is_not_suppressed(merchant):
    assert is_purposeless_e_transfer(merchant) is False
    assert has_purpose_evidence(merchant) is True


def test_e_transfer_matching_a_gazetteer_brand_counts_as_purpose_evidence():
    # A gazetteer hit is also purpose evidence -- an e-transfer that happens
    # to name a recognized public brand is not blindly suppressed either.
    merchant = "E-TRANSFER SENT SPOTIFY REFUND"
    assert has_purpose_evidence(merchant) is True
    assert is_purposeless_e_transfer(merchant) is False


# -- not an E-Transfer at all -> always False (this module is a no-op) -----


@pytest.mark.parametrize("merchant", [
    "ABM WITHDRAWAL",
    "ATM WITHDRAWAL",
    "TRANSFER TO SAVINGS ACCOUNT",
    "TIM HORTONS COFFEE",
    "",
    "   ",
])
def test_non_e_transfer_text_is_never_flagged(merchant):
    assert is_purposeless_e_transfer(merchant) is False


def test_hyphen_and_no_hyphen_spellings_both_detected():
    assert is_purposeless_e_transfer("E-TRANSFER SENT JANE SMITH") is True
    assert is_purposeless_e_transfer("ETRANSFER SENT JANE SMITH") is True


def test_case_insensitive():
    assert is_purposeless_e_transfer("e-transfer sent jane smith") is True
    assert is_purposeless_e_transfer("e-transfer sent jane smith rent") is False
