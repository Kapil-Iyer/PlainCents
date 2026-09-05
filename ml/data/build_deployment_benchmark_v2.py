"""
ML-G: deployment-oriented categorization benchmark, v2.

WHY A v2 CORPUS
---------------
ML-F's deployment_benchmark_v1 (ml/data/build_deployment_benchmark.py, 190
rows / 73 merchant groups) had a structural flaw that no amount of model
tuning could fix, and which the ML-G audit measured directly:

  * Exactly ONE merchant identity per merchant_group, and every group's name
    was unique across the whole corpus. Because the split is merchant-GROUP
    isolated (correctly), a held-out group's words had *never* been seen in
    TRAIN. A bag-of-words / char-n-gram model therefore had literally no
    feature in common with the row it was being asked to classify.
  * Measured consequence on the shipped v1 artifact (models/tfidf_logreg_v2.pkl,
    vocabulary = 200 word tokens): 11 of 18 realistic deployment-shaped probe
    strings vectorized to an ALL-ZERO feature row. sklearn's
    LogisticRegression on an all-zero row returns argmax(intercept_) -- one
    fixed class for every such input. That class was "Food & Dining". This
    is the exact "everything becomes Food & Dining" production symptom, and
    it is a representation-coverage bug, not a hyperparameter problem.
  * v1 sealed FINAL_TEST macro-F1 was 0.174 with 4 of 8 classes at F1 = 0.0.

WHAT CHANGED, AND WHY IT IS LEGITIMATE
--------------------------------------
Real merchant descriptors generalize because businesses in the same category
share a *head noun*: PHARMACY / DENTAL / CLINIC / OPTICAL; TRANSIT / TAXI /
PARKING / FUEL; HYDRO / INTERNET / MOBILE / WATER; DINER / PIZZA / CAFE /
GROCERY. A human categorizes an unseen "X PHARMACY" from that head noun, not
from having memorized X. v1 gave the model no way to learn that, because each
head noun appeared in at most one merchant group, and that group was in
exactly one partition.

v2 therefore builds MANY distinct merchant groups per category that share
category-typical head nouns, with distinct fabricated brand words. Under the
same merchant-group-isolated split, a held-out "CEDARVALE PHARMACY" now has
never-seen brand words ("CEDARVALE") but a seen head noun ("PHARMACY"). That
is precisely the generalization the deployed system needs, and it is how the
domain actually works -- it is NOT label leakage: the head noun is a genuine
property of the merchant's name, not a copy of the label. No category name
("Healthcare", "Rent & Utilities", ...) or any string derived from the label
taxonomy ever appears in a description.

HONEST DIFFICULTY: ~2 groups per category are deliberately BRAND-ONLY, with
no category-typical head noun at all ("ZENOVARA", "KESSLIN & CO"). These are
realistic (plenty of real merchants are named this way) and are the rows a
text classifier genuinely cannot get right from the description alone. They
exist so the corpus does not overstate achievable accuracy, and so the
abstention/margin mechanism evaluated in the ML-G bake-off has real cases to
be measured on.

PRIVACY
-------
Every merchant name below is fabricated for this file. No literal or
paraphrased private RBC/Scotiabank transaction description, account number,
balance, or personal name appears anywhere. The *structural* templates
(VISA DEBIT PURCHASE -, POS PURCHASE / Opos, CONTACTLESS INTERAC PURCHASE -,
Scotiabank-style mid-word truncation + numeric suffix, PREAUTH PYMT +
reference, ONLINE PURCHASE ....COM, store-number suffix) are carried over
unchanged from v1, which derived them from the ML-F-A audit's structural
buckets -- never from literal private text.

TERMINOLOGY: this is a "sanitized deployment-oriented bank-description
benchmark." It is not real-world data and must never be described as such.

AMBIGUOUS ROWS: rows with no spending-purpose signal by construction
(generic Interac e-transfer, ABM/ATM withdrawal, bare reference-code-only
debit, internal account transfer) keep v1's convention: is_ambiguous=True,
true_category blank. They are excluded from macro-F1 and scored separately
as a deterministic routing/coverage question.

REPRODUCIBILITY: SEED=42. Only date placement, amount jitter, and which
boilerplate template decorates a given occurrence are randomized. No
merchant, category, or ambiguity flag is ever chosen at random.

Run:  python -m ml.data.build_deployment_benchmark_v2
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SEED = 42
OUT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "evaluation" / "deployment_benchmark_v2.csv"
)

DATE_START = pd.Timestamp("2024-01-01")
DATE_END = pd.Timestamp("2025-12-31")

AMBIGUOUS_CATEGORY = ""

# ---------------------------------------------------------------------------
# Deployment-shaped boilerplate templates (identical structural set to v1).
# ---------------------------------------------------------------------------


def _visa_debit(rng: random.Random, name: str) -> str:
    return f"VISA DEBIT PURCHASE - {rng.randint(1000, 9999)} {name}"


def _contactless(rng: random.Random, name: str) -> str:
    return f"CONTACTLESS INTERAC PURCHASE - {rng.randint(1000, 9999)} {name}"


def _pos_opos(rng: random.Random, name: str) -> str:
    return f"pos purchase Opos {name} "


def _scotia_truncated(rng: random.Random, name: str) -> str:
    """Scotiabank-style mid-word truncation with an appended numeric suffix.
    Character n-grams are the representation that survives this shape; whole
    word tokens mostly do not, which is exactly why the ML-G bake-off tests a
    word+char union."""
    return f"Opos {name[:14].rstrip()}+{rng.randint(1000, 9999)} "


def _plain_store(rng: random.Random, name: str) -> str:
    return f"{name} #{rng.randint(1, 999):04d}"


def _online(rng: random.Random, name: str) -> str:
    return f"ONLINE PURCHASE {name.replace(' ', '').replace('&', '')}.COM"


def _preauth(rng: random.Random, name: str) -> str:
    return f"{name} PREAUTH PYMT {rng.randint(100000, 999999)}"


def _etransfer_named(rng: random.Random, name: str) -> str:
    """Payment-method boilerplate that STILL carries a usable merchant
    identity: must remain ML-eligible, never routed to Other by the
    structural-ambiguity rule."""
    return f"E-TRANSFER SENT {name} REF{rng.randint(10000, 99999)}"


TEMPLATES = [
    _visa_debit, _contactless, _pos_opos, _scotia_truncated,
    _plain_store, _online, _preauth, _etransfer_named,
]

# ONE SHARED TEMPLATE POOL FOR EVERY MERCHANT -- a deliberate methodological
# choice, and the second thing v1 got wrong.
#
# The obvious way to build this corpus is to give each merchant archetype its
# own template mix (card rails for restaurants, PREAUTH for utilities, ONLINE
# for digital subscriptions). A first pass of v2 did exactly that, and the
# measurement was unambiguous: the *boilerplate* became a category shortcut.
# Stripping it with the v2 normalizer then made VALIDATION macro-F1 FALL
# (0.375 -> 0.299), which is the signature of a model leaning on transaction-
# method words instead of merchant identity.
#
# That shortcut is worthless in deployment -- a real export routes the same
# merchant through whichever rail the user happened to pay on -- and it
# inflates the benchmark. So every merchant here draws from the SAME pool of
# all eight structural templates, on a deterministic rotation. The benchmark
# therefore measures exactly one thing: can the model recover merchant
# identity through arbitrary bank boilerplate.

# Occurrences per merchant group. Five is enough for each group's template
# rotation to actually vary while keeping the corpus hand-auditable.
OCCURRENCES_PER_GROUP = 5

FD = "Food & Dining"
TR = "Transport"
RU = "Rent & Utilities"
EN = "Entertainment"
HC = "Healthcare"
SH = "Shopping"
SU = "Subscriptions"
OT = "Other"

# ---------------------------------------------------------------------------
# Merchant groups: (display_name, category, amount_lo, amount_hi).
#
# The display name IS the merchant_group -- one fabricated brand per group.
#
# HEAD-NOUN REDUNDANCY is the whole design. Within each category, every
# category-typical head noun (PHARMACY, DINER, TRANSIT, HYDRO, CINEMA, ...)
# is carried by 2-4 DIFFERENT brand groups. Under the merchant-group-isolated
# split that means a head noun a held-out group uses has very likely been
# seen in TRAIN attached to a different brand -- which is exactly the
# generalization a deployed categorizer needs, and exactly what v1's
# one-group-per-head-noun corpus made impossible.
#
# Roughly two groups per category are deliberately BRAND-ONLY (no head noun
# at all). Those are honest failures: no text classifier can place them from
# the description, and they are what correction memory and abstention exist
# for. They are left in so the benchmark does not overstate what is
# achievable.
# ---------------------------------------------------------------------------

MERCHANTS: list[tuple[str, str, float, float]] = [
    # ---------------- Food & Dining ----------------
    ("MAPLEWOOD DINER", FD, 9.0, 32.0),
    ("CEDARVALE DINER", FD, 10.0, 34.0),
    ("KINGSTON ROW DINER", FD, 11.0, 36.0),
    ("NORTHSIDE PIZZA CO", FD, 14.0, 40.0),
    ("STONEFIRE PIZZERIA", FD, 15.0, 44.0),
    ("BRIARWOOD PIZZA HOUSE", FD, 13.0, 42.0),
    ("SUNRISE CAFE", FD, 3.5, 12.0),
    ("GOLDLEAF CAFE", FD, 4.0, 14.0),
    ("HARBOURVIEW CAFE", FD, 3.5, 13.0),
    ("BEANWORKS COFFEE", FD, 3.0, 11.0),
    ("EMBERLINE COFFEE ROASTERS", FD, 4.0, 18.0),
    ("IRONGATE GRILL", FD, 18.0, 62.0),
    ("RIDGEPORT GRILL HOUSE", FD, 20.0, 70.0),
    ("SILVERBIRCH KITCHEN", FD, 16.0, 55.0),
    ("TALLOWAY KITCHEN", FD, 17.0, 58.0),
    ("RIVERSIDE BAKERY", FD, 4.0, 18.0),
    ("FERNHILL BAKERY", FD, 4.5, 20.0),
    ("BLUEWAVE SUSHI BAR", FD, 16.0, 58.0),
    ("KOMORI SUSHI HOUSE", FD, 18.0, 64.0),
    ("BRICKYARD BURGERS", FD, 8.0, 24.0),
    ("LANTERN BURGERS", FD, 9.0, 26.0),
    ("CANYON TACO SHOP", FD, 9.0, 27.0),
    ("EASTGATE NOODLE HOUSE", FD, 11.0, 30.0),
    ("CORNERSTONE DELI", FD, 7.0, 26.0),
    ("WESTPORT BISTRO", FD, 22.0, 78.0),
    ("LAKEVIEW GROCERY MARKET", FD, 35.0, 175.0),
    ("WESTFIELD GROCERY", FD, 32.0, 168.0),
    ("CEDAR GROCERS", FD, 30.0, 160.0),
    ("PRAIRIE FARMS MARKET", FD, 20.0, 92.0),
    ("GREENLEAF SUPERMARKET", FD, 40.0, 210.0),
    ("PINEHURST SUPERMARKET", FD, 38.0, 195.0),
    ("TILLMARK & SONS", FD, 12.0, 40.0),
    ("OKARA HOUSE", FD, 14.0, 46.0),

    # ---------------- Transport ----------------
    ("TRANSITLINK FARE", TR, 3.25, 130.0),
    ("METROWAY TRANSIT FARE", TR, 3.0, 145.0),
    ("HARBOURSIDE TRANSIT PASS", TR, 3.25, 152.0),
    ("CITYLINE BUS PASS", TR, 3.25, 156.0),
    ("REDPINE BUS LINES", TR, 12.0, 88.0),
    ("SKYWAY COMMUTER RAIL", TR, 6.0, 42.0),
    ("HARBOURLINE RAIL PASS", TR, 8.0, 168.0),
    ("NORTHSTAR TAXI", TR, 10.0, 55.0),
    ("BLUELINE TAXI CO", TR, 9.0, 48.0),
    ("SILVERKEY CAB SERVICE", TR, 11.0, 52.0),
    ("QUICKPARK GARAGE", TR, 4.0, 24.0),
    ("CENTRE ST PARKING", TR, 3.0, 30.0),
    ("WESTGATE PARKING GARAGE", TR, 5.0, 34.0),
    ("MAPLE FUEL STATION", TR, 35.0, 95.0),
    ("REDROCK FUEL STOP", TR, 30.0, 105.0),
    ("NORTHPOINT PETRO STATION", TR, 32.0, 98.0),
    ("BRIGHTPATH GAS BAR", TR, 28.0, 92.0),
    ("RAPIDRENT CAR RENTAL", TR, 60.0, 340.0),
    ("SUMMIT AUTO RENTAL", TR, 65.0, 380.0),
    ("HARBOUR FERRY SERVICE", TR, 8.0, 45.0),
    ("GREENWAY BIKE SHARE", TR, 4.0, 22.0),
    ("EASTLINK TOLL ROUTE", TR, 5.0, 38.0),
    ("STONEBRIDGE TOLL HIGHWAY", TR, 6.0, 44.0),
    ("SUMMITPOINT AUTO SERVICE CENTRE", TR, 45.0, 420.0),
    ("VELLORIN GROUP", TR, 12.0, 60.0),
    ("KESSLIN & CO", TR, 15.0, 70.0),

    # ---------------- Rent & Utilities ----------------
    ("BRIGHTWAVE INTERNET", RU, 60.0, 110.0),
    ("FIBRELINE INTERNET SERVICES", RU, 55.0, 125.0),
    ("CLEARPATH BROADBAND", RU, 58.0, 118.0),
    ("NORTHGRID POWER CO", RU, 35.0, 220.0),
    ("LAKELAND POWER CORP", RU, 38.0, 205.0),
    ("STONERIDGE HYDRO", RU, 40.0, 240.0),
    ("VALEPORT HYDRO ELECTRIC", RU, 42.0, 235.0),
    ("CITYLINE WATER UTILITY", RU, 30.0, 105.0),
    ("BRIDGEPORT WATER SERVICES", RU, 28.0, 112.0),
    ("VALLEY GAS UTILITY", RU, 25.0, 210.0),
    ("NORTHFIELD NATURAL GAS", RU, 27.0, 198.0),
    ("CLEARTEL MOBILE", RU, 45.0, 95.0),
    ("SIGNALPOINT MOBILE", RU, 42.0, 118.0),
    ("PRAIRIE TELECOM WIRELESS", RU, 50.0, 130.0),
    ("SUMMIT PROPERTY MGMT RENT", RU, 1150.0, 2300.0),
    ("OAKFIELD RESIDENCES RENT", RU, 1200.0, 2450.0),
    ("HARBOURGATE APARTMENTS RENT", RU, 1100.0, 2280.0),
    ("HEARTHSTONE CONDO FEES", RU, 280.0, 620.0),
    ("BAYCREST CONDO CORP FEES", RU, 310.0, 680.0),
    ("HOMEGRID HOME INSURANCE", RU, 40.0, 160.0),
    ("SHIELDPOINT TENANT INSURANCE", RU, 45.0, 175.0),
    ("MERIDIAN PROPERTY TAX INSTALMENT", RU, 180.0, 640.0),
    ("ARVELLO SERVICES", RU, 60.0, 190.0),
    ("NUMARIS HOLDINGS", RU, 700.0, 1900.0),

    # ---------------- Entertainment ----------------
    ("SILVERSCREEN CINEMAS", EN, 10.0, 45.0),
    ("NORTHGATE CINEMA", EN, 11.0, 48.0),
    ("LAKESHORE CINEPLEX CINEMAS", EN, 12.0, 52.0),
    ("ROYALCREST THEATRE", EN, 25.0, 130.0),
    ("BRIARWOOD THEATRE COMPANY", EN, 28.0, 145.0),
    ("ARCADE ZONE", EN, 8.0, 35.0),
    ("PIXELPORT ARCADE BAR", EN, 12.0, 52.0),
    ("TICKETVAULT EVENTS", EN, 35.0, 220.0),
    ("HARBOURFEST EVENTS", EN, 28.0, 165.0),
    ("STAGEDOOR TICKETS", EN, 30.0, 195.0),
    ("FRONTROW TICKETS OFFICE", EN, 32.0, 205.0),
    ("STARLIGHT BOWL LANES", EN, 18.0, 65.0),
    ("KINGPIN BOWLING LANES", EN, 20.0, 72.0),
    ("PUZZLEROOM ESCAPE GAMES", EN, 28.0, 46.0),
    ("VAULTBREAK ESCAPE ROOMS", EN, 30.0, 52.0),
    ("PIXELPLAY GAME STORE", EN, 9.99, 79.99),
    ("QUARRY HILL MUSEUM", EN, 8.0, 38.0),
    ("WESTLIGHT GALLERY ADMISSION", EN, 10.0, 32.0),
    ("MOONRIDGE CONCERT HALL", EN, 40.0, 210.0),
    ("AURORA CONCERT SERIES", EN, 45.0, 190.0),
    ("HALVREN & PARTNERS", EN, 15.0, 70.0),
    ("ODESSIA GROUP", EN, 22.0, 95.0),

    # ---------------- Healthcare ----------------
    ("CAREWELL PHARMACY", HC, 6.0, 65.0),
    ("CEDARVALE PHARMACY", HC, 7.0, 72.0),
    ("BROOKFIELD PHARMACY", HC, 8.0, 68.0),
    ("NORTHFIELD DRUG MART", HC, 8.0, 80.0),
    ("VALEPORT DRUG MART", HC, 9.0, 78.0),
    ("BRIGHT SMILE DENTAL", HC, 85.0, 420.0),
    ("LAKESHORE DENTAL CENTRE", HC, 95.0, 480.0),
    ("STONEGATE DENTISTRY", HC, 90.0, 440.0),
    ("QUICKCARE WALK-IN CLINIC", HC, 20.0, 95.0),
    ("GREENLEAF NATUROPATHIC CLINIC", HC, 55.0, 140.0),
    ("HARBOURVIEW MEDICAL CLINIC", HC, 30.0, 160.0),
    ("VITAL PHYSIO CLINIC", HC, 60.0, 150.0),
    ("MOTIONWORKS PHYSIOTHERAPY", HC, 65.0, 160.0),
    ("REDPINE PHYSIOTHERAPY CENTRE", HC, 62.0, 155.0),
    ("CLEARVIEW OPTICAL", HC, 40.0, 250.0),
    ("BRIGHTLENS OPTICAL CENTRE", HC, 45.0, 285.0),
    ("NORTHSIGHT OPTOMETRY", HC, 50.0, 265.0),
    ("HANDS-ON MASSAGE THERAPY", HC, 75.0, 130.0),
    ("STILLWATER MASSAGE THERAPY", HC, 78.0, 138.0),
    ("SPINEWELL CHIROPRACTIC", HC, 55.0, 120.0),
    ("MERIDIAN MEDICAL LAB", HC, 25.0, 180.0),
    ("TELEHEALTH VIRTUAL CLINIC", HC, 20.0, 90.0),
    ("VOSKEN & ASSOCIATES", HC, 70.0, 220.0),
    ("ALDERWICK PARTNERS", HC, 60.0, 190.0),

    # ---------------- Shopping ----------------
    ("VALUEMART DEPT STORE", SH, 15.0, 210.0),
    ("NORTHGATE DEPT STORE", SH, 18.0, 240.0),
    ("BRIGHTFIELD DEPARTMENT STORE", SH, 20.0, 260.0),
    ("HOMEBASE HARDWARE", SH, 12.0, 260.0),
    ("IRONWORKS HARDWARE SUPPLY", SH, 14.0, 290.0),
    ("STONEGATE HARDWARE DEPOT", SH, 16.0, 310.0),
    ("BUILDRIGHT HOME IMPROVEMENT", SH, 18.0, 420.0),
    ("TRENDLINE APPAREL", SH, 14.0, 90.0),
    ("URBANWEAVE APPAREL CO", SH, 16.0, 105.0),
    ("FERNGATE CLOTHING", SH, 15.0, 98.0),
    ("RIDGELINE OUTFITTERS", SH, 22.0, 180.0),
    ("PAGEBOUND BOOKS", SH, 9.0, 65.0),
    ("INKWELL BOOKSHOP", SH, 8.0, 58.0),
    ("MARGINALIA BOOKS", SH, 10.0, 70.0),
    ("ACTIVEGEAR SPORTS", SH, 25.0, 180.0),
    ("PEAKLINE SPORTS OUTLET", SH, 28.0, 195.0),
    ("SUMMITPEAK OUTDOORS", SH, 28.0, 240.0),
    ("OFFICEPLUS OFFICE SUPPLY", SH, 8.0, 140.0),
    ("CLERKWELL OFFICE SUPPLY", SH, 9.0, 155.0),
    ("VOLTEDGE ELECTRONICS", SH, 20.0, 520.0),
    ("CIRCUITWAY ELECTRONICS", SH, 22.0, 560.0),
    ("OAKFRAME FURNITURE", SH, 60.0, 890.0),
    ("HEARTHWOOD FURNITURE CO", SH, 70.0, 940.0),
    ("BARGAIN BIN DISCOUNT", SH, 3.0, 28.0),
    ("PARRENTO", SH, 18.0, 130.0),
    ("MIRABEL & CO", SH, 24.0, 165.0),

    # ---------------- Subscriptions ----------------
    ("CLOUDDESK WORKSPACE PLAN", SU, 10.0, 10.0),
    ("FOCUSFLOW PRODUCTIVITY PLAN", SU, 8.99, 8.99),
    ("TUNEDRIFT MUSIC PLAN", SU, 9.99, 13.99),
    ("AUDIOWAVE PODCAST PLAN", SU, 14.95, 14.95),
    ("BOOKSTACK AUDIOBOOK PLAN", SU, 14.95, 14.95),
    ("VAULTKEEP CLOUD STORAGE PLAN", SU, 2.99, 11.99),
    ("PIXELCRAFT DESIGN SUBSCRIPTION", SU, 14.99, 14.99),
    ("CODEFORGE DEV TOOLS SUBSCRIPTION", SU, 12.0, 25.0),
    ("TEAMSYNC CHAT SUBSCRIPTION", SU, 10.0, 15.0),
    ("SAFEGUARD ANTIVIRUS SUBSCRIPTION", SU, 4.99, 9.99),
    ("NEWSPRINT DIGITAL SUBSCRIPTION", SU, 5.99, 19.99),
    ("STREAMBOX PLUS MONTHLY", SU, 11.99, 15.99),
    ("CINEFLOW STREAM PREMIUM", SU, 12.99, 19.99),
    ("NOVAPLAY STREAMING PREMIUM", SU, 10.99, 17.99),
    ("FITZONE GYM MEMBERSHIP", SU, 35.0, 75.0),
    ("PEAKFORM FITNESS MEMBERSHIP", SU, 32.0, 82.0),
    ("CLIMBWORKS CLUB MEMBERSHIP", SU, 40.0, 95.0),
    ("MEALPREP WEEKLY PLAN", SU, 55.0, 120.0),
    ("PETCARE MONTHLY PLAN", SU, 18.0, 45.0),
    ("ZENOVARA", SU, 9.99, 9.99),
    ("QUILVENT", SU, 12.5, 12.5),

    # ---------------- Other (identifiable, genuinely not a spending category) --
    ("MONTHLY ACCOUNT SERVICE FEE", OT, 4.0, 16.95),
    ("PAPER STATEMENT FEE", OT, 1.5, 4.0),
    ("CHEQUE ORDER FEE", OT, 8.0, 20.0),
    ("ACCOUNT REACTIVATION FEE", OT, 10.0, 25.0),
    ("SAFETY DEPOSIT BOX FEE", OT, 40.0, 120.0),
    ("INTERAC ACCESS FEE", OT, 1.0, 5.0),
    ("WIRE TRANSFER SERVICE FEE", OT, 15.0, 45.0),
    ("FOREIGN EXCHANGE SERVICE FEE", OT, 1.0, 12.0),
    ("NSF RETURNED ITEM FEE", OT, 45.0, 48.0),
    ("OVERLIMIT PENALTY FEE", OT, 20.0, 30.0),
    ("GOVERNMENT LICENCE RENEWAL FEE", OT, 20.0, 130.0),
    ("OVERDRAFT INTEREST CHARGE", OT, 0.25, 5.0),
    ("STOP PAYMENT SERVICE CHARGE", OT, 12.5, 20.0),
    ("CREDIT BALANCE INTEREST CHARGE", OT, 0.5, 9.0),
    ("LATE PAYMENT INTEREST CHARGE", OT, 1.0, 22.0),
    ("DONATION TO COMMUNITY FUND", OT, 10.0, 150.0),
    ("CHARITABLE GIVING PROGRAM", OT, 15.0, 90.0),
    ("ESTATE ADMINISTRATION CHARGE", OT, 35.0, 180.0),
]

# Structurally-ambiguous groups: no spending-purpose signal by construction.
AMBIGUOUS_GROUPS: list[tuple[str, list[str], float, float, str]] = [
    (
        "GENERIC ETRANSFER SENT",
        ["E-TRANSFER SENT", "Free Interac E-Transfer", "INTERAC E-TRANSFER SENT",
         "E-TRANSFER SENT REF 449021"],
        10.0, 800.0,
        "generic_transfer_description (could be rent, a gift, a friend repayment -- no purpose signal)",
    ),
    (
        "GENERIC ABM WITHDRAWAL",
        ["ABM WITHDRAWAL", "abm withdrawal", "ABM WITHDRAWAL 00412"],
        20.0, 300.0,
        "generic_description (cash withdrawal carries no merchant/purpose signal)",
    ),
    (
        "GENERIC ATM WITHDRAWAL",
        ["ATM WITHDRAWAL", "CASH WITHDRAWAL", "ATM WITHDRAWAL 8821"],
        20.0, 300.0,
        None,
    ),
    (
        "GENERIC PREAUTH REFERENCE ONLY",
        ["PREAUTH PYMT", "MISC DEBIT TRANSACTION", "PREAUTH PYMT 774120"],
        5.0, 120.0,
        "malformed_low_information description (no recognizable merchant name at all)",
    ),
    (
        "GENERIC ONLINE BANKING TRANSFER",
        ["ONLINE BANKING TRANSFER", "ONLINE TRANSFER TO DEPOSIT ACCOUNT",
         "TRANSFER TO SAVINGS ACCOUNT"],
        30.0, 900.0,
        "internal_transfer_description (moving money between the user's own accounts)",
    ),
]



def build() -> pd.DataFrame:
    rng = random.Random(SEED)
    rows: list[dict] = []
    span_days = (DATE_END - DATE_START).days
    n_templates = len(TEMPLATES)

    for group_index, (name, category, amt_lo, amt_hi) in enumerate(MERCHANTS):
        for i in range(OCCURRENCES_PER_GROUP):
            # Deterministic rotation through the SHARED template pool, with a
            # per-group starting offset. Every group sees several different
            # boilerplate shapes, and across the corpus each template is
            # spread evenly over all eight categories -- so transaction-method
            # boilerplate carries no category signal for a model to lean on.
            template = TEMPLATES[(group_index * 3 + i) % n_templates]
            date = DATE_START + pd.Timedelta(days=rng.randint(0, span_days))
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "description": template(rng, name),
                "amount": round(rng.uniform(amt_lo, amt_hi), 2),
                "merchant_group": name,
                "true_category": category,
                "is_ambiguous": False,
                "error_analysis_tag": "",
            })

    for group, variants, amt_lo, amt_hi, note in AMBIGUOUS_GROUPS:
        for variant in variants:
            for _ in range(2):
                date = DATE_START + pd.Timedelta(days=rng.randint(0, span_days))
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "description": variant,
                    "amount": round(rng.uniform(amt_lo, amt_hi), 2),
                    "merchant_group": group,
                    "true_category": AMBIGUOUS_CATEGORY,
                    "is_ambiguous": True,
                    "error_analysis_tag": note or "",
                })

    return (
        pd.DataFrame(rows)
        .sort_values(["merchant_group", "date", "description"])
        .reset_index(drop=True)
    )


def main() -> None:
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    n_amb = int(df["is_ambiguous"].sum())
    print(f"Wrote {len(df)} rows / {df['merchant_group'].nunique()} merchant groups -> {OUT_PATH}")
    print(f"  ambiguous: {n_amb} rows ({n_amb / len(df):.1%})")
    print(df.loc[~df["is_ambiguous"], "true_category"].value_counts().to_string())
    groups = df.loc[~df["is_ambiguous"]].drop_duplicates("merchant_group")
    print("\nmerchant groups per category:")
    print(groups["true_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
