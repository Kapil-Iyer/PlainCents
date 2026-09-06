"""
Tests for backend/services/gazetteer.py -- the small, deterministic
public-brand/service knowledge layer. Every merchant string here is a
publicly recognizable brand name (Spotify, Amazon, Tim Hortons, ...), the
same category of "fabricated for this test" text this project's own
existing fixtures (tests/fixtures/build_test_categorizer_model.py) already
use -- never a real private transaction description.
"""
import pytest

from backend.services.gazetteer import match_gazetteer


@pytest.mark.parametrize("merchant,expected_category", [
    ("SPOTIFY PREMIUM", "Subscriptions"),
    ("NETFLIX.COM", "Subscriptions"),
    ("DISNEY PLUS", "Subscriptions"),
    ("DISNEY+", "Subscriptions"),
    ("MICROSOFT SUBSCRIPTION", "Subscriptions"),
    ("ADOBE CREATIVE CLOUD", "Subscriptions"),
    ("OPENAI CHATGPT", "Subscriptions"),
    ("CURSOR AI SUBSCRIPTION", "Subscriptions"),
    ("GOODLIFE FITNESS CLUB", "Subscriptions"),
    ("LYFT RIDE", "Transport"),
    ("PRESTO FARE LOAD", "Transport"),
    ("GO TRANSIT FARE", "Transport"),
    ("METROLINX FARE", "Transport"),
    ("TTC METROPASS", "Transport"),
    ("VIA RAIL TICKET", "Transport"),
    ("TRANSLINK FARE", "Transport"),
    ("SHELL GAS STATION", "Transport"),
    ("PETRO-CANADA FUEL", "Transport"),
    ("ESSO STATION", "Transport"),
    ("TIM HORTONS #042", "Food & Dining"),
    ("STARBUCKS COFFEE", "Food & Dining"),
    ("MCDONALD'S", "Food & Dining"),
    ("LOBLAWS GROCERY", "Food & Dining"),
    ("SOBEYS MARKET", "Food & Dining"),
    ("NO FRILLS", "Food & Dining"),
    ("FRESHCO", "Food & Dining"),
    ("EBAY PURCHASE", "Shopping"),
    ("WALMART SUPERCENTER", "Shopping"),
    ("IKEA FURNITURE", "Shopping"),
    ("HOME DEPOT", "Shopping"),
    ("BEST BUY ELECTRONICS", "Shopping"),
    ("SHOPPERS DRUG MART", "Healthcare"),
    ("REXALL PHARMACY", "Healthcare"),
])
def test_known_public_brand_matches(merchant, expected_category):
    rule = match_gazetteer(merchant)
    assert rule is not None
    assert rule.category == expected_category


@pytest.mark.parametrize("merchant,expected_category", [
    ("AMAZON PRIME MEMBERSHIP", "Subscriptions"),
    ("AMAZON.CA MARKETPLACE ORDER", "Shopping"),
    ("AMAZON MKTPLACE PMTS", "Shopping"),
    ("UBER EATS DELIVERY", "Food & Dining"),
    ("UBER TRIP", "Transport"),
    ("GOOGLE ONE STORAGE", "Subscriptions"),
    ("GOOGLE PLAY STORE", "Shopping"),
])
def test_contextual_same_company_distinction(merchant, expected_category):
    """The SAME company can sell products in different categories -- the
    more specific rule must win over the generic one, in both directions."""
    rule = match_gazetteer(merchant)
    assert rule is not None
    assert rule.category == expected_category


@pytest.mark.parametrize("merchant", [
    "SOME UNRECOGNIZED LOCAL SHOP",
    "JANE SMITH",
    "GENERIC MERCHANT 12345",
    "",
    "   ",
])
def test_unknown_merchant_does_not_match(merchant):
    assert match_gazetteer(merchant) is None


@pytest.mark.parametrize("merchant", [
    "SHELLEY'S BAKERY",     # must not fire the "shell_fuel" rule
    "MYADOBESTORE",         # must not fire "adobe" as a mid-word substring
    "THE ESSOTERIC SHOP",   # must not fire "esso" as a mid-word substring
    "ONESIE STORE",         # must not fire a bare "ONE"-shaped rule (none exists)
    "METROPOLITAN BAKERY",  # "METRO" is deliberately excluded from the gazetteer
])
def test_substring_collision_safety(merchant):
    """Whole-word/whole-phrase matching must never fire on a substring of an
    unrelated word."""
    assert match_gazetteer(merchant) is None


def test_no_bare_paypal_rule_exists():
    """Deliberate design choice (see module docstring): a payment
    intermediary can carry a transaction of ANY category, so there is no
    rule for bare 'PAYPAL' -- only for a recognizable brand token, wherever
    it appears (including inside a PayPal-wrapped description)."""
    assert match_gazetteer("PAYPAL PAYMENT TO JANE SMITH") is None
    assert match_gazetteer("PAYPAL SEND MONEY") is None


def test_no_bare_google_rule_exists():
    """Same guardrail as bare PAYPAL: a bare 'GOOGLE' mention (no PLAY, no
    ONE, no PAYPAL co-occurrence) matches nothing -- GAZETTEER V2's
    paypal_google_checkout rule requires BOTH tokens, never just one."""
    assert match_gazetteer("GOOGLE ADS CAMPAIGN") is None
    assert match_gazetteer("GOOGLE FI WIRELESS") is None


def test_paypal_wrapped_known_brand_still_matches_on_the_brand():
    rule = match_gazetteer("PAYPAL *SPOTIFY")
    assert rule is not None
    assert rule.category == "Subscriptions"


def test_paypal_google_cooccurrence_matches_shopping():
    """GAZETTEER V2 (B1): co-occurrence of PAYPAL and GOOGLE, in either
    order, with arbitrary wrapper punctuation/spacing -- matches Shopping.
    Neither token alone would match anything (see the two guardrail tests
    above)."""
    for merchant in [
        "PAYPAL *GOOGLE STOREFRONT123",
        "PAYPAL * GOOGLE-SHOP99",
        "GOOGLE CHECKOUT VIA PAYPAL",
        "paypal google shop",
    ]:
        rule = match_gazetteer(merchant)
        assert rule is not None, merchant
        assert rule.category == "Shopping", merchant
        assert rule.name == "paypal_google_checkout"


def test_paypal_google_cooccurrence_yields_to_google_subscription_signal():
    """google_one/google_play are checked BEFORE paypal_google_checkout, so
    a more specific Google subscription signal wins even when PAYPAL is also
    present."""
    rule = match_gazetteer("PAYPAL *GOOGLE ONE STORAGE")
    assert rule is not None
    assert rule.name == "google_one"
    assert rule.category == "Subscriptions"


def test_audible_rule_was_removed():
    """GAZETTEER V2 (B3): the audible rule was removed -- it was wrong both
    times it fired in the frozen development-informed counterfactual. A
    bare 'AUDIBLE' (or 'AUDIBLE MEMBERSHIP') must not match anything now."""
    assert match_gazetteer("AUDIBLE MEMBERSHIP") is None
    assert match_gazetteer("AUDIBLE.COM") is None


def test_cursor_requires_the_ai_compound_not_bare_cursor():
    """Bare 'CURSOR' is an ordinary English/computing word (mouse cursor,
    text cursor) -- must not match. Only the compound 'CURSOR AI' does."""
    assert match_gazetteer("CURSOR BLINKING ISSUE") is None
    assert match_gazetteer("MOUSE CURSOR REPLACEMENT") is None
    rule = match_gazetteer("CURSOR AI MONTHLY")
    assert rule is not None
    assert rule.category == "Subscriptions"


def test_case_insensitive_matching():
    assert match_gazetteer("spotify premium") is not None
    assert match_gazetteer("Tim Hortons") is not None
    assert match_gazetteer("presto fare") is not None
    assert match_gazetteer("openai chatgpt") is not None
