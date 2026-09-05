"""
ML-G: deterministic bank-boilerplate normalization, v2.

Supersedes ml/categorization/text_normalize.py's normalize_deployment_text
for the ML-G production recipe. v1 is preserved untouched as ML-F evidence
(the ML-F selection record's Candidate C referenced it by name).

WHAT v1 MISSED, and why it lost to the no-normalization baseline in ML-F:
  * It stripped VISA DEBIT PURCHASE / POS PURCHASE / Opos / PREAUTH, but not
    ONLINE PURCHASE, not "E-TRANSFER SENT", and not alphanumeric reference
    tokens (REF44120, TXN9931). Those remaining tokens are pure
    transaction-method noise and, on a 50-200 term vocabulary, they crowd out
    merchant-identity terms.
  * It stripped bare numeric tokens only. A card/store suffix glued to a
    letter prefix ("REF10233") survived as its own vocabulary entry.

SHARED BETWEEN TRAINING AND INFERENCE. This is the single most important
property of this module. The ML-G bake-off calls it on TRAIN before fitting,
scripts/build_production_categorizer.py records which normalizer the winning
recipe used inside the artifact payload, and
backend/services/categorization_service.py resolves that recorded name back
to this exact function at inference time. ML-F's production path did NOT do
that -- CategorizationService vectorized the raw `merchant` column with no
normalizer at all -- so a normalizing candidate could never have been served
correctly even if it had won. That train/serve skew is closed in ML-G.

It is deliberately NOT wired into pipeline/ingest.py's merchant cleaning:
that would change the stored `merchant` column (dedup keys, display text,
search), a far larger blast radius than the ML text representation this
exists to improve. It applies only to the copy of the text handed to the
vectorizer -- and, separately and deliberately, to the correction-memory
identity key (backend/services/merchant_identity.py), which needs the same
"strip the varying reference noise" behavior for a different reason.

Input is assumed already run through pipeline.ingest._clean_merchant_text
(uppercase, punctuation stripped except hyphen/ampersand, whitespace
collapsed) -- the shape the `merchant` column always has.

Conservative by construction: only structural transaction-method/channel/
reference noise is removed. No merchant word is ever removed, and no literal
real (private) merchant name is encoded anywhere in this file.
"""
from __future__ import annotations

import re

# Transaction-method / channel boilerplate, matched as whole phrases so a
# merchant whose own name contains one of these words elsewhere is not
# mangled. Ordered longest-first so the multi-word forms win.
_BOILERPLATE_PATTERNS = [
    r"\bVISA DEBIT PURCHASE\b",
    r"\bVISA DEBIT REFUND\b",
    r"\bCONTACTLESS INTERAC PURCHASE\b",
    r"\bINTERAC E-TRANSFER SENT\b",
    r"\bINTERAC E-TRANSFER\b",
    r"\bINTERAC PURCHASE\b",
    r"\bFREE INTERAC E-TRANSFER\b",
    r"\bE-TRANSFER SENT TO\b",
    r"\bE-TRANSFER SENT\b",
    r"\bE-TRANSFER\b",
    r"\bETRANSFER\b",
    r"\bONLINE PURCHASE\b",
    r"\bPOS PURCHASE\b",
    r"\bOPOS\b",
    r"\bPREAUTH PYMT\b",
    r"\bPREAUTH\b",
    r"\bWWW\b",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS))

# Reference/transaction-id tokens: a short alphabetic prefix immediately
# followed by digits (REF44120, TXN9931, INV00231). Deliberately requires
# BOTH a recognized prefix and a digit run, so a genuine alphanumeric
# merchant identifier is not destroyed.
_REFERENCE_TOKEN_RE = re.compile(r"(?<!\S)(?:REF|TXN|TRN|INV|AUTH|SEQ|ID)\d+(?!\S)")

# Bare numeric tokens: card suffix, store number, reference code. Never a
# token mixing letters and digits.
_STANDALONE_DIGITS_RE = re.compile(r"(?<!\S)\d+(?!\S)")

# A lone hyphen or ampersand left stranded once a prefix is stripped.
_STRANDED_PUNCT_RE = re.compile(r"(?:^|\s)[-&](?=\s|$)")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_deployment_text_v2(merchant: str) -> str:
    """Strip bank transaction-method boilerplate, reference/transaction-id
    tokens, and bare numeric card/store suffixes from an already-cleaned
    `merchant` string.

    Deterministic, side-effect-free, safe on an empty string. Returns "" if
    the input was nothing but boilerplate (a generic e-transfer, for
    instance) -- callers treat an empty result as "no usable merchant
    identity", which is exactly what it is.
    """
    text = (merchant or "").upper()
    text = _BOILERPLATE_RE.sub(" ", text)
    text = _REFERENCE_TOKEN_RE.sub(" ", text)
    text = _STANDALONE_DIGITS_RE.sub(" ", text)
    text = _STRANDED_PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# Registry so a fitted artifact can record WHICH normalizer it was trained
# with by name, and inference can resolve that name back to the identical
# function -- closing the ML-F train/serve skew described in the module
# docstring. Never resolve a normalizer any other way in production code.
NORMALIZERS = {
    "normalize_deployment_text_v2": normalize_deployment_text_v2,
}


def resolve_normalizer(name: str | None):
    """Return the normalizer function registered under `name`, or None if
    `name` is None/empty. Raises on an unknown name rather than silently
    falling back to no normalization -- a mismatch here is exactly the
    train/serve skew this registry exists to prevent."""
    if not name:
        return None
    try:
        from ml.categorization.text_normalize import normalize_deployment_text

        legacy = {"normalize_deployment_text": normalize_deployment_text}
    except Exception:  # pragma: no cover - legacy module is always importable
        legacy = {}
    if name in NORMALIZERS:
        return NORMALIZERS[name]
    if name in legacy:
        return legacy[name]
    raise ValueError(
        f"Unknown text normalizer {name!r}. Known: "
        f"{sorted(set(NORMALIZERS) | set(legacy))}"
    )
