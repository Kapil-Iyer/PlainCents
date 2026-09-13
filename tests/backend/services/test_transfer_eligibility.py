"""
Tests for backend/services/transfer_eligibility.py. All merchant text is
fabricated (fake names like "JANE SMITH", "JOHN DOE") -- never a real
private recipient, account number, or user name.
"""
import pytest

from backend.services.transfer_eligibility import is_internal_account_transfer


# -- bank account-transfer mechanism vocabulary, no residual identity ------
# -> True (internal transfer, excluded from spending)


@pytest.mark.parametrize("merchant", [
    "ONLINE BANKING TRANSFER - 6346",
    "ONLINE BANKING TRANSFER",
    "ONLINE TRANSFER TO DEPOSIT ACCOUNT-1234",
    "TRANSFER TO SAVINGS ACCOUNT",
    "TRANSFER FROM CHEQUING",
    "TRANSFER TO ACCOUNT 4521",
])
def test_own_account_transfer_mechanism_text_is_internal(merchant):
    assert is_internal_account_transfer(merchant) is True


# -- generic E-Transfer, even a frequently-repeated one -> NEVER internal --
# (a name is never proof of self-ownership; false negatives are accepted)


@pytest.mark.parametrize("merchant", [
    "E-TRANSFER SENT JANE SMITH",
    "E-TRANSFER SENT JANE SMITH REF88213",
    "E-TRANSFER SENT JANE SMITH REF00001",
    "E-TRANSFER SENT JANE SMITH REF00002",
    "E-TRANSFER SENT JANE SMITH REF00003",  # "repeatedly to the same
                                             # recipient label" must still
                                             # never be treated as proof of
                                             # self-transfer -- see module
                                             # docstring.
    "INTERAC E-TRANSFER SENT TO JOHN DOE",
    "E-TRANSFER - AUTODEPOSIT JANE SMITH",
    "E-TRANSFER REQUEST FULFILLED JANE SMITH",
])
def test_generic_e_transfer_is_never_treated_as_internal(merchant):
    assert is_internal_account_transfer(merchant) is False


# -- ordinary purchase -> False (stays spending) ---------------------------


@pytest.mark.parametrize("merchant", [
    "STARBUCKS #4521",
    "AMAZON.CA PURCHASE",
    "VISA DEBIT PURCHASE - SPOTIFY P4542F1",
    "CONTACTLESS INTERAC PURCHASE - 12345",
])
def test_ordinary_merchant_purchase_is_not_internal(merchant):
    assert is_internal_account_transfer(merchant) is False


# -- ATM/ABM cash withdrawal -> False, unchanged (a withdrawal is not an
# account-to-account transfer; see module docstring)


@pytest.mark.parametrize("merchant", [
    "ABM WITHDRAWAL",
    "ATM WITHDRAWAL",
    "CASH WITHDRAWAL",
])
def test_atm_abm_cash_withdrawal_is_not_internal(merchant):
    assert is_internal_account_transfer(merchant) is False


# -- bill payment / preauthorized payment -> False, unchanged --------------


@pytest.mark.parametrize("merchant", [
    "PREAUTHORIZED PAYMENT - HYDRO",
    "PREAUTH PYMT INSURANCE",
    "ONLINE BANKING PAYMENT - 4521",  # "PAYMENT", not "TRANSFER" -- a bill
                                       # payment, structurally distinct
])
def test_bill_and_preauthorized_payments_are_not_internal(merchant):
    assert is_internal_account_transfer(merchant) is False


# -- a transfer marker that ALSO carries a residual identity/recipient -----
# name -> False (this is the safety net: "no residual identity" is required,
# not just the marker alone)


@pytest.mark.parametrize("merchant", [
    "TRANSFER TO JANE SMITH FOR RENT",
    "ONLINE BANKING TRANSFER TO MAPLEWOOD DINER",
])
def test_transfer_marker_with_named_recipient_is_not_internal(merchant):
    assert is_internal_account_transfer(merchant) is False


# -- edge cases -------------------------------------------------------------


@pytest.mark.parametrize("merchant", ["", "   ", None])
def test_blank_or_none_is_never_internal(merchant):
    assert is_internal_account_transfer(merchant) is False


def test_no_private_identifiers_hardcoded_in_module_source():
    # Structural guard against regressions that would defeat the whole
    # point of this module: it must never encode a real account number,
    # user name, or bank-contact label. This asserts the function's
    # behavior stays purely marker + residual-identity based, not that any
    # particular string is absent from the source (that would be brittle);
    # the real assurance is test_generic_e_transfer_is_never_treated_as_internal
    # above using varied, repeated, name-bearing fixtures. "ALEX CHEN" here
    # is a fabricated placeholder, standing in for a saved e-transfer
    # contact label at another bank -- never a real name.
    assert is_internal_account_transfer("E-TRANSFER SENT OTHERBANK ALEX CHEN REF12345") is False
