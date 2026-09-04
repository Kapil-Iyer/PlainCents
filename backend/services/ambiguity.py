"""
Deterministic structurally-ambiguous-row detection (ML-F-A audit §14/§16;
ML-F brief §16).

The audit found two real-export row shapes that carry NO spending-purpose
signal at all, by construction, no matter how good the categorizer's
representation gets: a generic Interac e-transfer send, and an ABM/ATM cash
withdrawal. Forcing either into one of the 8 spending categories is a guess,
not a classification. This module implements the smallest defensible
behavior the ML-F brief allows (§16): a deterministic, cheap, taxonomy-free
check on already-cleaned `merchant` text, used only to route these specific
row shapes to the existing "Other" category -- never a new category, never a
new UI, never fuzzy/semantic matching.

This is intentionally narrow. It does not attempt to resolve every
low-information row (a bare, non-descriptive PREAUTH/reference code with no
merchant name is still handed to the ML classifier as today) -- only the two
shapes the audit specifically confirmed have zero recoverable signal.
"""
from __future__ import annotations

import re

_AMBIGUOUS_PATTERNS = [
    re.compile(r"\bE-?TRANSFER\b"),
    re.compile(r"\bABM WITHDRAWAL\b"),
    re.compile(r"\bATM WITHDRAWAL\b"),
]


def is_structurally_ambiguous(merchant: str) -> bool:
    """True if the (already-cleaned, uppercase) `merchant` text matches one
    of the two audit-confirmed no-purpose-signal shapes: a generic Interac
    e-transfer send, or an ABM/ATM cash withdrawal. Pure deterministic
    string matching -- no fuzzy/semantic logic, no ML call."""
    text = (merchant or "").upper()
    return any(pattern.search(text) for pattern in _AMBIGUOUS_PATTERNS)
