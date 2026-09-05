"""
Deterministic structural-ambiguity detection.

Some bank rows carry NO spending-purpose signal at all, by construction, no
matter how good the categorizer's representation gets: a generic Interac
e-transfer send, an ABM/ATM cash withdrawal, an internal account transfer, a
bare reference-code-only debit. Forcing any of those into one of the eight
spending categories is a guess, not a classification. This module routes
exactly those rows -- and only those -- to "Other" as an explicit SYSTEM
decision (predicted_category = "Other", confirmed_category = NULL).

ML-G FIX -- THE OVER-ROUTING BUG
--------------------------------
The previous implementation matched three bare regexes against the raw
merchant text:

    \\bE-?TRANSFER\\b   \\bABM WITHDRAWAL\\b   \\bATM WITHDRAWAL\\b

which meant "E-TRANSFER SENT MAPLEWOOD DINER REF44120" -- payment-method
boilerplate wrapped around a perfectly usable merchant identity -- was routed
to "Other" and never shown to the classifier at all. Measured on the ML-G
benchmark's FINAL_TEST partition, that rule fired on 27 of 195 (13.8%)
non-ambiguous rows. That is the "over-route everything to Other" failure mode
the brief explicitly warns against, and it silently destroyed categorization
for a whole payment rail.

The rule here is two-part instead:

  1. A structural MARKER must be present (transfer / withdrawal / bare
     pre-authorized-payment / miscellaneous-debit boilerplate). No marker ->
     an ordinary row, handed to the classifier as always.
  2. AND, after stripping that boilerplate plus reference codes, card
     suffixes and other generic banking vocabulary, NO merchant-identity
     token may remain. If anything identifiable survives, the row keeps its
     signal and stays ML-eligible.

So "ABM WITHDRAWAL" is ambiguous, "E-TRANSFER SENT MAPLEWOOD DINER" is not,
and "WIRE TRANSFER SERVICE FEE" is not (it is a genuine, identifiable bank
fee, correctly classified as Other by the model rather than routed there
blindly).

Pure deterministic string matching throughout: no fuzzy matching, no
semantic similarity, no embeddings, no ML call, no network access. Operates
on the already-cleaned, uppercased `merchant` text.
"""
from __future__ import annotations

import re

from ml.categorization.text_normalize_v2 import normalize_deployment_text_v2

# ---------------------------------------------------------------------------
# Part 1: structural markers. A row must match one of these to even be a
# candidate for ambiguity routing.
# ---------------------------------------------------------------------------
_MARKER_PATTERNS = [
    r"\bE-?TRANSFER\b",
    r"\bINTERAC E-?TRANSFER\b",
    r"\bABM\b",
    r"\bATM\b",
    r"\bCASH WITHDRAWAL\b",
    r"\bWITHDRAWAL\b",
    r"\bONLINE BANKING TRANSFER\b",
    r"\bTRANSFER (?:TO|FROM)\b",
    r"\bMISC(?:ELLANEOUS)? DEBIT\b",
    r"\bPREAUTH PYMT\b",
    r"\bPREAUTHORIZED PAYMENT\b",
]
_MARKER_RE = re.compile("|".join(_MARKER_PATTERNS))

# ---------------------------------------------------------------------------
# Part 2: generic banking vocabulary. A token that is ONLY one of these is
# not a merchant identity. Deliberately a closed, small, boring list of
# transaction-mechanics words -- never a merchant dictionary, never a
# category keyword list.
# ---------------------------------------------------------------------------
_GENERIC_TOKENS = frozenset({
    "ABM", "ACCOUNT", "ATM", "AUTH", "BANKING", "CASH", "CHEQUING", "CREDIT",
    "DEBIT", "DEPOSIT", "E", "ETRANSFER", "FREE", "FROM", "INTERAC", "MISC",
    "MISCELLANEOUS", "NO", "ONLINE", "PAYMENT", "PMT", "POS", "PREAUTH",
    "PREAUTHORIZED", "PURCHASE", "PYMT", "REF", "REFERENCE", "SAVINGS", "SENT",
    "SERVICE", "TFR", "TO", "TRANSACTION", "TRANSFER", "VISA", "WITHDRAWAL",
})

# A token needs at least this many characters to count as merchant identity.
# Two-letter fragments left behind by truncation are not an identity.
_MIN_IDENTITY_TOKEN_LEN = 3

_NON_ALPHA_RE = re.compile(r"[^A-Z ]")


def residual_identity_tokens(merchant: str) -> list[str]:
    """The merchant-identity tokens left after removing bank boilerplate,
    reference/card-suffix noise, and generic banking vocabulary.

    Empty means "this text names nothing" -- the precise condition that makes
    a marked row genuinely unclassifiable. Exposed (rather than kept private)
    because both the ambiguity rule below and the correction-memory identity
    key in backend/services/merchant_identity.py need exactly this notion,
    and they must not drift apart.
    """
    stripped = normalize_deployment_text_v2(merchant or "")
    stripped = _NON_ALPHA_RE.sub(" ", stripped.upper())
    return [
        token for token in stripped.split()
        if len(token) >= _MIN_IDENTITY_TOKEN_LEN and token not in _GENERIC_TOKENS
    ]


def is_structurally_ambiguous(merchant: str) -> bool:
    """True only when the text carries a transfer/withdrawal/bare-preauth
    marker AND no merchant-identity token survives normalization.

    Deterministic and side-effect-free. Safe on empty input (returns False:
    an empty merchant is a parsing problem, not a structural-ambiguity one,
    and is handled upstream)."""
    text = (merchant or "").upper()
    if not text.strip():
        return False
    if not _MARKER_RE.search(text):
        return False
    return not residual_identity_tokens(text)
