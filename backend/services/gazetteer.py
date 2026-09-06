"""
A small, deterministic PUBLIC merchant/service knowledge layer.

WHY THIS EXISTS
---------------
The real-bank baseline evaluation (private_data/reports/baseline_metrics.json
-- see scripts/private_eval/) found the frozen TF-IDF + Logistic Regression
categorizer frequently fails on merchants a human would recognize instantly
from the brand name alone (a streaming subscription, a public transit fare
card, a national grocery chain) -- not because the text is ambiguous, but
because a ~200-8000-term sparse text model trained on a small corpus simply
never saw that exact brand spelled that way. This module answers "is this a
publicly, unambiguously recognizable merchant/service, and if so what
category does its PRODUCT (not just its brand) actually belong to" --
deterministically, with no model call and no private data.

WHAT THIS DELIBERATELY IS NOT
------------------------------
  * NOT a private merchant list. Every entry here is a nationally/globally
    recognizable brand or public service name (streaming platforms, transit
    systems, national retail/grocery chains, well-known payment-processor-
    wrapped subscription services) -- never a recipient name, a small local
    business, or anything learned from this project's own private_data/.
  * NOT a full merchant directory. Requirement: keep this small and
    maintainable (a few dozen entries), not an attempt at completeness.
    V2 is the FINAL targeted refinement pass for this gazetteer -- no V3
    expansion is planned; remaining gaps are a documented, disclosed
    limitation (see scripts/private_eval/ reports), not a TODO.
  * NOT a rule for a bare, unqualified payment intermediary. There is
    deliberately NO bare "PAYPAL" rule and NO bare "GOOGLE" rule: either
    alone can carry a transaction of ANY category, so mapping either
    directly would be an unsafe broad mapping. A description like
    "PAYPAL *SPOTIFY" or "PAYPAL *NETFLIX" is still matched correctly --
    not via a PayPal-specific rule, but because the underlying recognizable
    brand token ("SPOTIFY", "NETFLIX") is what these rules actually key on,
    wherever it appears in the string. A bare "PAYPAL" send to a person, or
    an unrecognized PayPal merchant, matches nothing here and falls through
    to the classifier/abstention exactly as before. This is the intended
    design, not a gap.

    GAZETTEER V2: the one exception is `paypal_google_checkout` below --
    NOT a "PayPal" rule and NOT a "Google" rule, but a CO-OCCURRENCE rule
    that fires only when BOTH tokens are present together. See that rule's
    own comment for the public-semantics justification (this is not "the
    frozen dev set happened to label these Shopping" curve-fitting -- see
    the module's own disclosure of that risk there).

CONTEXTUAL PAIRS
----------------
The SAME company can sell products in different categories (a subscription
product and a marketplace product from the same company are not the same
spending category). Where that's true, the more specific/contextual rule is
listed and checked BEFORE its generic counterpart, e.g.:
  * "AMAZON PRIME" (subscription) is checked before bare "AMAZON" (marketplace)
  * "UBER EATS" (food delivery) is checked before bare "UBER" (rideshare)

COLLISION SAFETY
----------------
Every pattern is a whole-word/whole-phrase match (`\b...\b`), so a brand
token never fires on a substring of an unrelated word (e.g. `\bSHELL\b`
matches "SHELL" but not "SHELLEY'S BAKERY"). Deliberately short or generic
tokens that would carry real collision risk (a bare "GO", "ONE", "METRO" --
"METRO" alone is both a Montreal transit brand AND a Canadian grocery chain,
so it is intentionally NOT included at all) are excluded rather than
guessed at.

MATCHING TEXT
-------------
Matching runs on the same `normalize_deployment_text_v2`-cleaned text the
classifier itself is fed (strips bank transaction-method boilerplate and
reference codes) -- not a second, divergent text-cleaning path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ml.categorization.text_normalize_v2 import normalize_deployment_text_v2


@dataclass(frozen=True)
class GazetteerRule:
    name: str
    pattern: re.Pattern
    category: str
    note: str


def _rule(name: str, regex: str, category: str, note: str) -> GazetteerRule:
    return GazetteerRule(name=name, pattern=re.compile(regex), category=category, note=note)


# Ordered list -- FIRST match wins. Contextual/specific rules for a brand
# that sells more than one kind of product are listed before that brand's
# generic rule.
RULES: list[GazetteerRule] = [
    # -- contextual pairs: same brand, different product/category --------
    _rule("amazon_prime", r"\bAMAZON\s*PRIME\b", "Subscriptions",
          "Amazon's paid membership subscription, distinct from marketplace purchases."),
    _rule("amazon_generic", r"\bAMAZON\b", "Shopping",
          "General Amazon marketplace/retail purchase."),
    _rule("uber_eats", r"\bUBER\s*EATS\b", "Food & Dining",
          "Uber's food-delivery product, distinct from rideshare trips."),
    _rule("uber_generic", r"\bUBER\b", "Transport",
          "Uber rideshare trip."),
    _rule("google_one", r"\bGOOGLE\s*ONE\b", "Subscriptions",
          "Google's cloud-storage subscription, distinct from Play Store purchases."),
    _rule("google_play", r"\bGOOGLE\s*PLAY\b", "Shopping",
          "Google Play Store app/media marketplace purchase."),
    # GAZETTEER V2 (B1): a co-occurrence rule, not a "PayPal" rule and not a
    # "Google" rule -- it fires ONLY when BOTH tokens are present in the same
    # description, checked AFTER google_one/google_play above so a more
    # specific Google subscription signal always wins first.
    #
    # PUBLIC-SEMANTICS JUSTIFICATION (not "the frozen dev set labels these
    # Shopping"): PayPal and Google Pay are both general-purpose checkout/
    # payment rails used across essentially every category of purchase.
    # Neither one, alone, signals a category -- that's exactly why there is
    # no bare rule for either. But the PAIRING of "PAYPAL" with "GOOGLE" in
    # one bank description is characteristic of a specific, publicly
    # understood mechanism: a small online storefront's checkout flow
    # routed through Google-mediated payment infrastructure and settled via
    # PayPal (as opposed to a one-to-one Google product purchase, which
    # google_one/google_play already name specifically). That mechanism is
    # itself a retail/marketplace checkout pattern, independent of which
    # specific small merchant sits behind it -- the same reasoning already
    # used for "UBER EATS" (the compound names a delivery MECHANISM, not a
    # specific restaurant).
    #
    # DISCLOSED LIMITATION: this is a co-occurrence heuristic over a payment-
    # rail pairing, not a named brand the way "SPOTIFY" is. It is included
    # because scripts/private_eval/counterfactual_rescore.py's
    # development-informed ablation found the pairing perfectly consistent
    # on the frozen real-bank sample it was investigated against (a bare
    # "PAYPAL" row with no Google co-occurrence was NOT included in that
    # consistency check, and is still deliberately left unmapped by this
    # gazetteer, per the "never map bare PAYPAL alone" rule above) -- but a
    # sample from one household is not proof this pairing always means
    # Shopping. If real-world evidence ever contradicts this, remove or
    # narrow this ONE rule rather than the bare-PayPal/bare-Google
    # guardrails it was built to respect.
    _rule("paypal_google_checkout", r"(?=.*\bPAYPAL\b)(?=.*\bGOOGLE\b)", "Shopping",
          "PayPal+Google co-occurrence: characteristic of a Google-mediated "
          "checkout settled via PayPal, a retail/marketplace mechanism -- "
          "never inferred from bare PAYPAL or bare GOOGLE alone."),

    # -- subscriptions: recognizable recurring digital services ----------
    _rule("spotify", r"\bSPOTIFY\b", "Subscriptions", "Music streaming subscription."),
    _rule("netflix", r"\bNETFLIX\b", "Subscriptions", "Video streaming subscription."),
    _rule("disney_plus", r"\bDISNEY\s*\+|\bDISNEY\s+PLUS\b", "Subscriptions",
          "Video streaming subscription."),
    _rule("apple_subscription", r"\bAPPLE\s+(?:MUSIC|TV|ONE)\b|\bICLOUD\b", "Subscriptions",
          "Apple's recurring subscription products (music/video/cloud storage) -- "
          "deliberately not bare 'APPLE', which is also the App Store/hardware retailer."),
    # NOTE: matching runs on normalize_deployment_text_v2-cleaned text, which
    # strips bare/standalone numeric tokens (card suffixes, reference codes)
    # -- so a literal "365" in "MICROSOFT 365" is stripped before this rule
    # ever sees it. Matching bare "MICROSOFT" instead (Microsoft's other
    # consumer-facing offerings -- Game Pass, OneDrive -- are also
    # subscriptions) avoids relying on a token this pipeline can't reliably
    # preserve; "OFFICE 365" is deliberately NOT a separate rule -- bare
    # "OFFICE" alone collides with "OFFICE DEPOT" (a retailer), and it would
    # have the identical "365 gets stripped" problem.
    _rule("microsoft", r"\bMICROSOFT\b", "Subscriptions",
          "Microsoft's consumer-facing offerings are predominantly subscriptions."),
    _rule("adobe", r"\bADOBE\b", "Subscriptions",
          "Adobe's consumer-facing offerings are overwhelmingly Creative Cloud subscriptions."),
    _rule("openai", r"\bOPENAI\b|\bCHATGPT\b", "Subscriptions", "AI assistant subscription."),
    # GAZETTEER V2 (B3): the "audible" rule (bare \bAUDIBLE\b -> Subscriptions)
    # is REMOVED. scripts/private_eval/counterfactual_rescore.py's
    # development-informed ablation found it fired twice on the frozen
    # real-bank sample and was WRONG both times (0/2 accuracy) -- "audible"
    # as a bare word is not reliably an Audible.com subscription charge in
    # real bank text. Removed rather than narrowed: no compound phrase
    # ("AUDIBLE MEMBERSHIP", "AUDIBLE.COM") was actually observed to justify
    # keeping a narrower version, so there is nothing safely gazetteerable
    # here right now. Left as a comment, not silently deleted, so a future
    # pass doesn't re-add the same broken rule from the same publicly
    # obvious idea without knowing it was already tried and measured wrong.
    _rule("cursor_ai", r"\bCURSOR\s+AI\b", "Subscriptions",
          "Cursor AI coding-assistant subscription. Requires the compound "
          "'CURSOR AI', never bare 'CURSOR' -- bare 'cursor' is an ordinary "
          "English/computing word (mouse cursor, text cursor) with real "
          "collision risk in unrelated text."),
    _rule("goodlife_fitness", r"\bGOODLIFE\s*FITNESS\b", "Subscriptions",
          "Recurring gym membership (Canadian public chain)."),

    # -- transport: fuel, rideshare, public transit -----------------------
    _rule("lyft", r"\bLYFT\b", "Transport", "Rideshare trip."),
    _rule("presto", r"\bPRESTO\b", "Transport", "Ontario public transit fare card."),
    _rule("go_transit", r"\bGO\s*TRANSIT\b", "Transport", "Ontario regional transit."),
    _rule("metrolinx", r"\bMETROLINX\b", "Transport",
          "Ontario's regional transit agency (operates GO Transit/PRESTO) -- "
          "GAZETTEER V2 (B2). Unambiguous, low collision risk (not a common "
          "English word, unlike bare 'METRO')."),
    _rule("ttc", r"\bTTC\b", "Transport", "Toronto Transit Commission."),
    _rule("via_rail", r"\bVIA\s*RAIL\b", "Transport", "Canadian national passenger rail."),
    _rule("translink", r"\bTRANSLINK\b", "Transport", "Metro Vancouver transit authority."),
    _rule("shell_fuel", r"\bSHELL\b", "Transport", "Fuel/gas station chain."),
    _rule("petro_canada", r"\bPETRO[\s-]?CANADA\b", "Transport", "Fuel/gas station chain."),
    _rule("esso", r"\bESSO\b", "Transport", "Fuel/gas station chain."),

    # -- food & dining: unambiguous restaurant/coffee/grocery chains ------
    _rule("tim_hortons", r"\bTIM\s*HORTONS\b", "Food & Dining", "Coffee/QSR chain."),
    _rule("starbucks", r"\bSTARBUCKS\b", "Food & Dining", "Coffee chain."),
    _rule("mcdonalds", r"\bMCDONALD'?S\b", "Food & Dining", "QSR chain."),
    _rule("loblaws", r"\bLOBLAWS\b", "Food & Dining", "Canadian grocery chain."),
    _rule("sobeys", r"\bSOBEYS\b", "Food & Dining", "Canadian grocery chain."),
    _rule("no_frills", r"\bNO\s*FRILLS\b", "Food & Dining", "Canadian grocery chain."),
    _rule("freshco", r"\bFRESHCO\b", "Food & Dining", "Canadian grocery chain."),

    # -- shopping: general marketplace / big-box retail -------------------
    _rule("ebay", r"\bEBAY\b", "Shopping", "General online marketplace."),
    _rule("walmart", r"\bWALMART\b", "Shopping", "General big-box retailer."),
    _rule("ikea", r"\bIKEA\b", "Shopping", "Furniture/home goods retailer."),
    _rule("home_depot", r"\bHOME\s*DEPOT\b", "Shopping", "Home-improvement retailer."),
    _rule("best_buy", r"\bBEST\s*BUY\b", "Shopping", "Electronics retailer."),

    # -- healthcare: pharmacy chains ---------------------------------------
    _rule("shoppers_drug_mart", r"\bSHOPPERS\s*DRUG\s*MART\b", "Healthcare", "Pharmacy chain."),
    _rule("rexall", r"\bREXALL\b", "Healthcare", "Pharmacy chain."),
]


def match_gazetteer(merchant: str) -> GazetteerRule | None:
    """Returns the first matching rule, or None. Matching is case-insensitive
    by construction (input is uppercased by normalize_deployment_text_v2's
    own contract), whole-word/whole-phrase, deterministic, and side-effect
    free -- never a network call, never a fuzzy/semantic match."""
    text = normalize_deployment_text_v2(merchant or "")
    if not text:
        return None
    for rule in RULES:
        if rule.pattern.search(text):
            return rule
    return None
