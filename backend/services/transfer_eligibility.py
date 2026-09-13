"""
Spending eligibility: internal/self account-transfer detection.

WHY THIS EXISTS
---------------
"Persisted as a negative-amount debit" is not the same fact as "this was
spending." Moving money between accounts the same person owns (RBC chequing
-> RBC savings, an "online banking transfer" to your own deposit account,
and so on) leaves no trace of consumption at all -- but until this module
existed, every such row was summed into total spend exactly like a
purchase, because nothing in the pipeline distinguished the two.

THIS IS DELIBERATELY A SEPARATE, NARROW CONCERN
-------------------------------------------------
This module does not touch categorization (backend.services.ambiguity,
backend.services.e_transfer_policy) and is not consulted by
is_structurally_ambiguous. The two questions are orthogonal:

  * "What category is this row filed under?" -- ambiguity.py /
    e_transfer_policy.py / the ML classifier, unchanged.
  * "Does this row count as spending at all?" -- this module, new.

A row can be structurally ambiguous (routed to category "Other") WITHOUT
being an internal transfer -- an ABM cash withdrawal is still spending and
still counted, exactly as before. A row this module flags as an internal
transfer never reaches the ambiguity check, the e-transfer policy, the
gazetteer, or the ML classifier at all: backend.services.category_decision
checks this FIRST and skips categorization entirely for a row it flags (see
that module's own comment), so an internal transfer can never seed
correction memory or pollute categorization diagnostics.

WHAT COUNTS AS "INTERNAL", CONSERVATIVELY
------------------------------------------
Only the bank's own account-to-account transfer MECHANISM vocabulary --
"ONLINE BANKING TRANSFER", "TRANSFER TO/FROM <account/account type>" (this
covers RBC's "ONLINE TRANSFER TO DEPOSIT ACCOUNT" too, since it contains
"TRANSFER TO") -- combined with the same "names nothing else" test
backend.services.ambiguity already uses (no residual merchant/recipient
identity token survives normalization). That combination is structurally
different from an Interac e-Transfer: a plain bank "transfer," with no
e-Transfer marker, is in every mainstream bank's own product design only
ever available between accounts already linked under one login (i.e.
accounts the same person owns) -- reaching a DIFFERENT person requires
Interac e-Transfer, a cheque, or a wire, each of which carries its own,
distinct marker text.

Interac e-Transfers (E-TRANSFER / INTERAC E-TRANSFER) are DELIBERATELY never
matched here, no matter how often they repeat to what looks like the same
recipient label. A person's name is not proof of self-ownership, and
inferring "this looks like a self-transfer" from a name would require
private, user-specific data this module must never encode (no account
numbers, no user name, no saved e-Transfer contact labels). A generic
e-Transfer therefore always stays spending-eligible under this module --
false negatives (an actual self-transfer counted as spending) are the
accepted, disclosed trade-off over any false positive that would hide a
genuine payment to another person.

ATM/ABM cash withdrawals, "MISC DEBIT", and pre-authorized payments are also
never matched here -- those keep their existing, unchanged
structural-ambiguity handling (routed to category "Other", but correctly
still counted as spending: withdrawing cash or paying a recurring bill is
still money leaving the accounts this product tracks, not a transfer
between two of them).

Pure deterministic string matching: no fuzzy matching, no semantic
similarity, no embeddings, no ML call, no private identifiers, no account
numbers, no user name.
"""
from __future__ import annotations

import re

from backend.services.ambiguity import residual_identity_tokens

# Bank account-to-account transfer MECHANISM vocabulary only. Deliberately
# does NOT include any E-TRANSFER/INTERAC E-TRANSFER pattern -- that is a
# separate, person-reachable payment rail this module must never treat as
# proof of self-ownership (see module docstring).
_INTERNAL_TRANSFER_MARKER_PATTERNS = [
    r"\bONLINE BANKING TRANSFER\b",
    r"\bTRANSFER (?:TO|FROM)\b",
]
_INTERNAL_TRANSFER_MARKER_RE = re.compile("|".join(_INTERNAL_TRANSFER_MARKER_PATTERNS))


def is_internal_account_transfer(merchant: str) -> bool:
    """True only when the text carries bank account-transfer MECHANISM
    vocabulary (never an e-Transfer marker) AND no merchant/recipient
    identity token survives normalization -- i.e. the text names an account
    transfer and nothing else. Safe on empty input (returns False). See the
    module docstring for the full policy and why false negatives are the
    accepted trade-off over false positives."""
    text = (merchant or "").upper()
    if not text.strip():
        return False
    if not _INTERNAL_TRANSFER_MARKER_RE.search(text):
        return False
    return not residual_identity_tokens(text)
