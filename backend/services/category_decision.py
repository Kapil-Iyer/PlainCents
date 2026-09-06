"""
The single categorization decision path.

WHY THIS MODULE EXISTS
----------------------
Preview and Confirm used to compute categories differently, and the drift was
not cosmetic. Preview called the model and staged its raw output. Confirm
then, independently, (a) overrode structurally-ambiguous rows to "Other" and
(b) looked up remembered user corrections. So the category a user saw in the
Preview table was NOT the category that got stored when they clicked Confirm
Import -- for exactly the rows where the difference mattered most.

Everything that decides a category now lives here, and both paths call it.
There is one ordering, in one place:

    1. structural ambiguity  -- text that names nothing gets "Other"
                                (backend.services.ambiguity, unchanged)
    1b. E-Transfer, no purpose evidence -- a recipient's NAME survives, but
                                the text gives no evidence of what the money
                                was actually for (backend.services.
                                e_transfer_policy). Narrower than step 1:
                                does not touch anything step 1 already
                                handles, only E-Transfers step 1 correctly
                                leaves ML-eligible because a residual
                                identity token exists. Uses its OWN source,
                                SOURCE_AMBIGUOUS_E_TRANSFER -- distinct from
                                step 1's SOURCE_STRUCTURAL_OTHER, so stored
                                data can later tell "this text names nothing"
                                apart from "a name survives but names no
                                purpose" (see backend/schemas/transaction.py).
    1c. public-brand gazetteer -- a small, deterministic, PUBLIC merchant/
                                service knowledge layer (backend.services.
                                gazetteer) for brands recognizable on sight
                                that a small sparse-text classifier often
                                isn't.
    2. model + abstention    -- the classifier, and the policy for when its
                                answer is not worth serving
    3. correction memory     -- a prior GENUINE user correction for this
                                merchant identity on this bank. Applied
                                uniformly after EVERY step above except step
                                1 (unchanged, pre-existing behavior -- see
                                that branch's own comment) so a human
                                correction stays authoritative over the
                                gazetteer and the new E-Transfer policy, not
                                just over the model.

REAL-BANK BASELINE MOTIVATING 1b/1c
------------------------------------
Steps 1b and 1c were added after a private, local real-world baseline
evaluation of the frozen model (never used to retrain or tune anything --
see scripts/private_eval/) surfaced two distinct, non-ML failure modes: (a)
recognizable public brands/services the classifier's small vocabulary
simply never learned, and (b) person-name-only E-Transfers being served a
fabricated-looking category guess because a residual identity token existed
even though it carried no purpose signal. Neither step retrains, tunes, or
touches models/categorizer_v3.pkl, min_margin, or the training corpus --
both are pre-model, deterministic policy layers, exactly like step 1.

HITL SEMANTICS, PRESERVED EXACTLY
---------------------------------
    predicted_category  the current SYSTEM decision (model output, or the
                        system's own "Other" for an ambiguous / abstained
                        row). Never a human's choice.
    confirmed_category  a GENUINE user correction, or the reuse of one. NULL
                        otherwise.
    effective_category  COALESCE(confirmed_category, predicted_category),
                        computed by the v_transactions_effective view.

Steps 1 and 2 write only `predicted_category`. Auto-"Other" therefore can
never seed correction memory, and a row nobody looked at can never read as a
manual override. Step 3 is the only thing that may set
`confirmed_category`, and it only ever propagates a value that a real user
action produced in the first place.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.services.ambiguity import is_structurally_ambiguous
from backend.services.categorization_service import CategorizationService
from backend.services.e_transfer_policy import is_purposeless_e_transfer
from backend.services.gazetteer import match_gazetteer
from backend.services.merchant_identity import stable_merchant_key

SYSTEM_OTHER = "Other"

# How the system arrived at `predicted_category`. Surfaced through Preview so
# the UI can explain a decision instead of just showing a label.
SOURCE_MODEL = "model"
SOURCE_STRUCTURAL_OTHER = "structural_other"
SOURCE_LOW_CONFIDENCE_OTHER = "low_confidence_other"
# A public-brand gazetteer hit (backend.services.gazetteer) -- a
# deterministic, non-ML decision, same family as SOURCE_STRUCTURAL_OTHER but
# distinguishable because it names a real category, not "Other".
SOURCE_GAZETTEER = "gazetteer"
# A purposeless-E-Transfer decision (backend.services.e_transfer_policy):
# `predicted_category` is "Other", same as SOURCE_STRUCTURAL_OTHER, but the
# REASON is different and worth telling apart -- SOURCE_STRUCTURAL_OTHER
# means "this text names nothing at all" (an ATM withdrawal, a bare
# transfer), while this means "a recipient identity survives, but a
# person's name is not spending-purpose evidence". A customer-facing
# distinction ("genuine miscellaneous Other" vs "purposeless E-Transfer
# served as Other") needs this told apart in stored data, not just derived
# in memory at decide-time -- see backend/schemas/transaction.py and
# frontend/src/pages/transactions/CategoryBadge.tsx.
SOURCE_AMBIGUOUS_E_TRANSFER = "ambiguous_e_transfer"


@dataclass(frozen=True)
class CategoryDecision:
    """One row's complete categorization decision.

    `predicted_category` and `confirmed_category` map 1:1 onto the columns of
    the same name. Everything else is explanatory metadata: it is returned to
    Preview so the UI can show why, and it is never persisted.
    """

    predicted_category: str
    confirmed_category: str | None
    source: str
    merchant_key: str | None
    model_category: str | None = None
    n_active_features: int | None = None
    margin: float | None = None

    @property
    def effective_category(self) -> str:
        return self.confirmed_category or self.predicted_category

    @property
    def is_remembered_correction(self) -> bool:
        return self.confirmed_category is not None


class CorrectionMemory:
    """Lookup of prior genuine user corrections, keyed by stable merchant
    identity + bank.

    Deliberately a thin protocol around TransactionRepository rather than a
    new table: a correction already lives in `transactions.confirmed_category`,
    and that column is only ever written by a real user action (a PATCH via
    TransactionService.update) or by this exact propagation of one.
    """

    def __init__(self, txn_repo):
        self._repo = txn_repo

    def lookup(self, merchant_key: str | None) -> str | None:
        if not merchant_key:
            return None
        return self._repo.find_latest_confirmed_category_by_key(merchant_key)


def decide(
    merchant: str,
    bank_source: str | None,
    categorization: CategorizationService,
    memory: CorrectionMemory | None = None,
) -> CategoryDecision:
    """Decide one row. See the module docstring for the ordering and why.

    `memory=None` means "decide without consulting correction memory", which
    is what a caller wants when no bank context exists yet. It is NOT how
    Preview calls this -- Preview passes memory precisely so that what it
    shows matches what Confirm will store.
    """
    key = stable_merchant_key(merchant, bank_source)

    # 1. Structural ambiguity. Text that names nothing cannot be classified,
    #    and this is checked BEFORE the model so the classifier is never
    #    asked to guess on an input with no recoverable purpose signal.
    if is_structurally_ambiguous(merchant):
        return CategoryDecision(
            predicted_category=SYSTEM_OTHER,
            # A structurally-ambiguous row yields no merchant_key at all, so
            # there is nothing to look memory up by -- but this is also a
            # deliberate guarantee, not an accident of the key function:
            # generic transfers must never share a memory entry.
            confirmed_category=None,
            source=SOURCE_STRUCTURAL_OTHER,
            merchant_key=key,
        )

    # 1b. E-Transfer, no purpose evidence (backend.services.e_transfer_policy).
    #     Unlike step 1 above, a residual identity DOES exist here (a
    #     recipient's name), so `key` is very often non-None -- correction
    #     memory is deliberately consulted, so a human's prior explicit
    #     correction for this exact recipient identity still wins over this
    #     system default.
    if is_purposeless_e_transfer(merchant):
        return CategoryDecision(
            predicted_category=SYSTEM_OTHER,
            confirmed_category=memory.lookup(key) if memory is not None else None,
            source=SOURCE_AMBIGUOUS_E_TRANSFER,
            merchant_key=key,
        )

    # 1c. Public-brand gazetteer (backend.services.gazetteer). Deterministic,
    #     pre-model, never a fabricated guess -- and, same as 1b, correction
    #     memory is still consulted so a human correction remains
    #     authoritative over the gazetteer too.
    gazetteer_match = match_gazetteer(merchant)
    if gazetteer_match is not None:
        return CategoryDecision(
            predicted_category=gazetteer_match.category,
            confirmed_category=memory.lookup(key) if memory is not None else None,
            source=SOURCE_GAZETTEER,
            merchant_key=key,
        )

    # 2. Model, plus the abstention policy frozen into the artifact.
    result = categorization.classify(merchant)
    predicted = result["category"]
    source = SOURCE_LOW_CONFIDENCE_OTHER if result["abstained"] else SOURCE_MODEL

    # 3. Correction memory. `predicted_category` above is preserved untouched
    #    whether or not a remembered correction applies -- that is what keeps
    #    "what the system thinks" and "what the user decided" separately
    #    auditable for the lifetime of the row.
    confirmed = memory.lookup(key) if memory is not None else None

    return CategoryDecision(
        predicted_category=predicted,
        confirmed_category=confirmed,
        source=source,
        merchant_key=key,
        model_category=result["model_category"],
        n_active_features=result["n_active_features"],
        margin=result["margin"],
    )


def decide_batch(
    rows: list[tuple[str, str | None]],
    categorization: CategorizationService,
    memory: CorrectionMemory | None = None,
) -> list[CategoryDecision]:
    """Batch form of decide(), for import paths.

    Uses one vectorize+predict call for every non-ambiguous row rather than
    one per row -- the whole reason CategorizationService exposes a batch
    API. The per-row ordering and outcomes are identical to calling decide()
    in a loop; there is a test that asserts exactly that, because two
    implementations of one decision is the bug this module exists to remove.
    """
    decisions: list[CategoryDecision | None] = [None] * len(rows)
    to_classify: list[int] = []

    for i, (merchant, bank_source) in enumerate(rows):
        key = stable_merchant_key(merchant, bank_source)
        if is_structurally_ambiguous(merchant):
            decisions[i] = CategoryDecision(
                predicted_category=SYSTEM_OTHER,
                confirmed_category=None,
                source=SOURCE_STRUCTURAL_OTHER,
                merchant_key=key,
            )
            continue

        if is_purposeless_e_transfer(merchant):
            decisions[i] = CategoryDecision(
                predicted_category=SYSTEM_OTHER,
                confirmed_category=memory.lookup(key) if memory is not None else None,
                source=SOURCE_AMBIGUOUS_E_TRANSFER,
                merchant_key=key,
            )
            continue

        gazetteer_match = match_gazetteer(merchant)
        if gazetteer_match is not None:
            decisions[i] = CategoryDecision(
                predicted_category=gazetteer_match.category,
                confirmed_category=memory.lookup(key) if memory is not None else None,
                source=SOURCE_GAZETTEER,
                merchant_key=key,
            )
            continue

        to_classify.append(i)

    if to_classify:
        results = categorization.classify_batch([rows[i][0] for i in to_classify])
        for i, result in zip(to_classify, results):
            key = stable_merchant_key(rows[i][0], rows[i][1])
            decisions[i] = CategoryDecision(
                predicted_category=result["category"],
                confirmed_category=memory.lookup(key) if memory is not None else None,
                source=SOURCE_LOW_CONFIDENCE_OTHER if result["abstained"] else SOURCE_MODEL,
                merchant_key=key,
                model_category=result["model_category"],
                n_active_features=result["n_active_features"],
                margin=result["margin"],
            )

    return [d for d in decisions if d is not None]
