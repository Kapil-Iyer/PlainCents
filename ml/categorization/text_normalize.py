"""
ML-F Candidate C: deterministic bank-boilerplate normalization.

Purpose (ML-F-A audit §11/§13): several deployment-shaped prefixes/suffixes
observed in real exports -- "VISA DEBIT PURCHASE -", "POS PURCHASE"/"Opos",
"CONTACTLESS INTERAC PURCHASE -", a trailing card-suffix/reference digit
group -- crowd the fixed-size TF-IDF vocabulary with transaction-METHOD
words ("debit", "purchase") that carry no category signal, at the expense of
merchant-IDENTITY words that do. This function removes exactly those
patterns and nothing else.

Design constraints (frozen by the ML-F-A audit and this phase's brief):
  - Deterministic, no ML, no fitting -- pure string transformation.
  - Conservative: only structural transaction-method/reference-code noise is
    removed. No merchant word is ever removed, and it never encodes any
    literal real (private) merchant name.
  - SHARED between training and inference: the categorization bake-off
    (ml/categorization/run_deployment_bakeoff.py) calls this on TRAIN before
    fitting Candidate C/E, and -- only if that candidate wins -- production
    inference (backend/services/categorization_service.py) calls the exact
    same function on the `merchant` text before vectorizing. It is
    deliberately NOT wired into pipeline/ingest.py's merchant cleaning: that
    would also change the stored `merchant` column (dedup keys, transaction
    list display, search), which is a much larger blast radius than the ML
    text representation this function exists to improve. This function is
    applied only to the copy of the text handed to the vectorizer.

Input is assumed already run through pipeline.ingest._clean_merchant_text
(uppercase, punctuation stripped except hyphen/ampersand, whitespace
collapsed) -- the same shape CategorizationService's `merchant` column
always has.
"""
from __future__ import annotations

import re

# Transaction-method / channel boilerplate. Matched as whole tokens/phrases
# so a merchant that happens to contain one of these words as part of its own
# name is not silently mangled elsewhere in the string -- these patterns only
# ever occur as bank-added prefixes in the observed deployment structure.
_BOILERPLATE_PATTERNS = [
    r"\bVISA DEBIT PURCHASE\b",
    r"\bVISA DEBIT REFUND\b",
    r"\bCONTACTLESS INTERAC PURCHASE\b",
    r"\bINTERAC PURCHASE\b",
    r"\bPOS PURCHASE\b",
    r"\bOPOS\b",
    r"\bPREAUTH PYMT\b",
    r"\bPREAUTH\b",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS))

# A lone hyphen left behind once a prefix is stripped (e.g. "PURCHASE - 4521
# NAME" -> "- 4521 NAME"), plus any standalone numeric-only token (card
# suffix, store number, reference code) -- never a token that mixes letters
# and digits, since that could be part of a genuine merchant identifier.
_LEADING_HYPHEN_RE = re.compile(r"(?:^|\s)-(?=\s|$)")
_STANDALONE_DIGITS_RE = re.compile(r"(?<!\S)\d+(?!\S)")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_deployment_text(merchant: str) -> str:
    """Strip known bank transaction-method boilerplate and bare numeric
    reference/card-suffix tokens from an already-cleaned `merchant` string.
    Deterministic, side-effect-free. Safe to call on an empty string."""
    text = merchant or ""
    text = _BOILERPLATE_RE.sub(" ", text)
    text = _STANDALONE_DIGITS_RE.sub(" ", text)
    text = _LEADING_HYPHEN_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
