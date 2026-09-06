"""
Conservative structural policy for evidence-free E-Transfers.

WHY THIS EXISTS
---------------
The real-bank baseline (private_data/reports/baseline_metrics.json) found
that most E-Transfers in real data DO carry a residual identity token (a
recipient's name) -- so backend.services.ambiguity.is_structurally_ambiguous
correctly leaves them ML-eligible (that check exists precisely to NOT
over-route boilerplate-wrapped real merchant identities to "Other" -- see
its own module docstring). But a recipient's NAME is not spending-purpose
evidence: "E-TRANSFER SENT JANE SMITH" tells you who received money, not
what it was for. Handed to the classifier anyway, that row gets a confident-
looking but fabricated category guess -- measured on the real baseline, 57
of 96 ground-truth E-Transfer rows were served "Food & Dining" specifically,
far more than plausible.

THIS IS A DELIBERATE, NARROWER, ADDITIONAL RULE
------------------------------------------------
This module does NOT change or weaken is_structurally_ambiguous (a bare
"E-TRANSFER SENT" with no residual identity at all is still caught by that
existing rule, unchanged). This module adds a second, narrower check for
rows that already have residual identity but where that identity carries
no spending-purpose signal: after removing E-Transfer boilerplate, does the
text contain either (a) a small closed vocabulary of GENERIC, PUBLIC
purpose-indicating words (rent/utility/bill-like obligations -- ordinary
English, never a private merchant/recipient name), or (b) a recognized
public gazetteer brand (backend.services.gazetteer)? If neither, the row is
routed to "Other" as a SYSTEM decision, same as general structural
ambiguity -- not because the text names nothing, but because what it DOES
name (a person) carries no purpose evidence a classifier could honestly
reason from.

KNOWN, DISCLOSED TRADE-OFF
---------------------------
A residual identity that happens to be a recognizable BUSINESS name with no
purpose-vocabulary word and no gazetteer entry (e.g. a small, non-chain
restaurant reached via e-transfer) will also be routed to "Other" by this
rule, where before it would have reached the classifier. This is an
accepted, conservative trade-off: distinguishing "this residual text is a
business name" from "this residual text is a person's name" would require
either a private business-name list (out of scope -- would itself risk
encoding private data) or a semantic/NLP judgment this pass deliberately
does not make. See tests/backend/services/test_e_transfer_policy.py and the
updated tests/backend/services/test_category_decision.py for exactly which
existing fixture case this changes and why.

Pure deterministic string matching throughout: no fuzzy matching, no
semantic similarity, no embeddings, no ML call, no private data.
"""
from __future__ import annotations

import re

from backend.services.gazetteer import match_gazetteer
from ml.categorization.text_normalize_v2 import normalize_deployment_text_v2

_E_TRANSFER_MARKER_RE = re.compile(r"\bE-?TRANSFER\b|\bINTERAC E-?TRANSFER\b")

# Small, closed, GENERIC purpose-evidence vocabulary: ordinary English words
# for housing/utility/bill-like financial obligations. Never a merchant
# name, never a recipient name, never learned from this project's private
# data -- a deliberately boring, public list of obligation-shaped words.
_PURPOSE_EVIDENCE_RE = re.compile(
    r"\bRENT\b|\bLANDLORD\b|\bMORTGAGE\b|\bHYDRO\b|\bUTILIT(?:Y|IES)\b|"
    r"\bINTERNET\b|\bWI[\s-]?FI\b|\bELECTRIC(?:ITY)?\b|\bWATER\s+BILL\b|"
    r"\bGAS\s+BILL\b|\bINSURANCE\b|\bTUITION\b|\bDAYCARE\b|\bCHILDCARE\b|"
    r"\bREPAIR\b|\bINVOICE\b|\bCONDO\s+FEE\b|\bMAINTENANCE\s+FEE\b"
)


def has_purpose_evidence(merchant: str) -> bool:
    """True when the (already boilerplate-normalized) text names either a
    generic obligation word or a recognized public gazetteer brand -- either
    is enough evidence that a category guess would be grounded in the text,
    not fabricated."""
    text = normalize_deployment_text_v2(merchant or "")
    if _PURPOSE_EVIDENCE_RE.search(text):
        return True
    return match_gazetteer(text) is not None


def is_purposeless_e_transfer(merchant: str) -> bool:
    """True when `merchant` is an E-Transfer (by marker) with no purpose
    evidence at all -- i.e. it should be routed to "Other" as a SYSTEM
    decision, never handed to the classifier. False for anything that isn't
    an E-Transfer (this function is a no-op on ATM/withdrawal/generic-
    transfer text -- those are backend.services.ambiguity's concern) and for
    an E-Transfer that DOES carry purpose evidence."""
    text = (merchant or "").upper()
    if not text.strip():
        return False
    if not _E_TRANSFER_MARKER_RE.search(text):
        return False
    return not has_purpose_evidence(text)
