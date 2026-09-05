"""
Stable merchant identity for correction memory.

THE PROBLEM THIS SOLVES
-----------------------
Correction memory looked up a prior user correction by the exact `merchant`
string plus bank. Real bank descriptions embed a value that changes on every
single transaction -- a card suffix, a store number, a reference code:

    VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY
    VISA DEBIT PURCHASE - 9137 CAREWELL PHARMACY
    CAREWELL PHARMACY #0284

Those are three distinct `merchant` strings for one merchant, so an exact
match essentially never fired. A user could correct the same pharmacy every
month forever and the system would never learn. The feature was, in practice,
inert on real data.

THE KEY
-------
`stable_merchant_key(merchant, bank_source)` returns a deterministic
identity built from the merchant-identity tokens that survive boilerplate and
reference-noise removal, sorted and joined, scoped by bank. Same merchant
through different payment rails -> same key. Different merchants -> different
keys.

CONSTRAINTS THIS DELIBERATELY RESPECTS
--------------------------------------
  * Deterministic. Pure string transformation, no state, no randomness.
  * Bank-aware. The key is always scoped by bank_source, so a correction made
    on an RBC row never leaks onto a Scotiabank row.
  * No fuzzy matching, no edit distance, no embeddings, no LLM, no external
    lookup. Two merchants match only if their identity token SETS are equal.
  * No broad accidental merging. A row whose text names nothing -- a generic
    e-transfer, an ABM withdrawal, a bare reference code -- yields NO key at
    all (None), so unrelated transfers can never collapse into one shared
    memory entry and teach the system a category for "transfers in general".
    This is the single most important safety property here.
  * The original `merchant` text is never modified. This key is stored
    alongside it for lookup only; display, search and dedup keep using the
    exact text the bank sent.

TOKEN SORTING is deliberate: "CAREWELL PHARMACY" and a truncated-then-
reordered rendering of the same name produce the same key. It is also why
the key is a poor choice for display, and it is never used for that.
"""
from __future__ import annotations

from backend.services.ambiguity import residual_identity_tokens

# A key needs at least this many identity tokens. One short token is too thin
# an identity to safely merge transactions on -- "OPOS BRIGHTW+4821" reduced
# to a single fragment should not become a memory entry that later swallows a
# different merchant sharing that fragment.
_MIN_IDENTITY_TOKENS = 1
_MIN_TOTAL_IDENTITY_CHARS = 5


def stable_merchant_key(merchant: str, bank_source: str | None) -> str | None:
    """A deterministic, bank-scoped identity for correction-memory matching.

    Returns None when the text carries no usable merchant identity -- the
    caller must then neither store nor look up a correction for the row.
    """
    tokens = residual_identity_tokens(merchant or "")
    if len(tokens) < _MIN_IDENTITY_TOKENS:
        return None
    if sum(len(t) for t in tokens) < _MIN_TOTAL_IDENTITY_CHARS:
        return None
    bank = (bank_source or "UNKNOWN").strip().upper()
    return f"{bank}|{' '.join(sorted(set(tokens)))}"
