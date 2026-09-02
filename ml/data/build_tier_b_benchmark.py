"""
ML-B / ML Spec Section 3.2: Tier B independently curated categorization
evaluation benchmark.

WHAT THIS IS:
  A hand-authored set of ~90 merchant identities (grouped) x 2-4 transaction
  variants each, with human-confirmed category labels, covering the frozen
  8-category MVP taxonomy (config.CATEGORIES). Every merchant name, category
  label, and description wording below was chosen by hand in this file --
  nothing is derived from pipeline/cluster.py's MERCHANT_KEYWORDS dict or
  scripts/generate_synthetic_24mo.py's vocabulary. That independence is the
  entire scientific point of this dataset (ML Spec Section 2/Section 3.2):
  it breaks the V1 coupling where the same person wrote both the synthetic
  merchant strings and the keyword rules used to "grade" them.

WHAT THIS IS NOT (ML Spec Section 3.2/Section 21):
  - NOT Tier A. This is not real bank data. It must never be described as
    real-world TD accuracy or naturally-occurring transaction performance.
  - NOT a bigger version of the same coupling problem: labels here are
    assigned once, by hand, per row at authoring time -- there is no
    runtime keyword-matching function that assigns a label from the
    description text (contrast pipeline/cluster.py's _get_true_labels,
    which computes a label from merchant text at call time). The label
    column below is data, fixed at authoring time, exactly like a human
    annotator would confirm a label on a real transaction.

KNOWN LIMITATION (documented, not hidden): the same person (the builder)
both wrote the descriptions and assigned the labels. This is weaker than
an independent third-party annotation process. It still meaningfully
reduces the generator/heuristic-vocabulary circularity described in ML
Spec Section 2, because the vocabulary and formatting patterns are new and
because several genuinely ambiguous/hard cases are included deliberately
(Section 8's required error-analysis categories), rather than optimized to
be easy to classify.

MERCHANT GROUPING FOR LEAKAGE-SAFE SPLITTING (ML Spec Section 6):
  Real bank exports repeat the "same" merchant with varying suffixes (store
  numbers, city names, reference codes). `merchant_group` is the normalized
  identity that must not be split across TRAIN/VALIDATION/FINAL TEST -- all
  rows sharing a merchant_group go to exactly one partition. Description
  variants within a group intentionally vary formatting (case, truncation,
  trailing digits) the way real exports do, while `merchant_group` stays
  fixed so the split code has an unambiguous grouping key to work from.

REPRODUCIBILITY: the only randomness here is DATE placement and small AMOUNT
jitter per transaction instance (seeded, deterministic). No label, no
merchant, no description wording is chosen at random.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SEED = 42
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "tier_b_benchmark.csv"

# Calendar window the benchmark descriptions are dated across -- deliberately
# disjoint-looking from the synthetic generator's 2023-01..2024-12 window is
# not required (categorization is not temporally split, ML Spec Section 6),
# but a real, plausible, multi-year span is used for realism.
DATE_START = pd.Timestamp("2024-01-01")
DATE_END = pd.Timestamp("2025-12-31")

# ---------------------------------------------------------------------------
# Hand-authored merchant groups.
#
# Each entry: merchant_group -> (category, [description variants], (amount_lo, amount_hi), notes)
#
# `notes` records which Section 8 error-analysis phenomenon (if any) this
# group was deliberately included to exercise. Groups with notes=None are
# ordinary/clear-cut cases -- most of the benchmark should be ordinary,
# matching Section 8's instruction that hard cases are examined, not that
# the whole dataset is adversarial.
# ---------------------------------------------------------------------------

MERCHANT_GROUPS: dict[str, tuple[str, list[str], tuple[float, float], str | None]] = {
    # ---------------- Food & Dining ----------------
    "STARBUCKS COFFEE": ("Food & Dining", [
        "STARBUCKS COFFEE #0442 TORONTO ON", "STARBUCKS COFFEE #1187 OTTAWA ON",
        "STARBUCKS CANADA #0442", "STARBUCKS #0921 MISSISSAUGA",
    ], (4.25, 12.75), None),
    "A&W RESTAURANT": ("Food & Dining", [
        "A&W RESTAURANT #315", "A & W DRIVE THRU #315 BARRIE ON", "A&W CANADA 0315",
    ], (8.50, 22.00), None),
    "PIZZA PIZZA": ("Food & Dining", [
        "PIZZA PIZZA #4471", "PIZZA PIZZA ORDER ONLINE", "PIZZA PIZZA LTD #4471 LONDON ON",
    ], (14.00, 38.00), None),
    "SWISS CHALET": ("Food & Dining", [
        "SWISS CHALET ROTIS #204", "SWISS CHALET RESTAURANT KITCHENER",
    ], (18.00, 46.00), None),
    "HARVEYS": ("Food & Dining", [
        "HARVEYS RESTAURANT #88", "HARVEYS BURGERS OSHAWA ON",
    ], (9.00, 24.00), None),
    "WENDYS": ("Food & Dining", [
        "WENDYS #6621 OLD YONGE ST", "WENDY'S RESTAURANT 6621",
    ], (7.50, 19.00), None),
    "POPEYES": ("Food & Dining", [
        "POPEYES LOUISIANA KITCHEN #12", "POPEYES CHICKEN WATERLOO ON",
    ], (10.00, 26.00), None),
    "KFC CANADA": ("Food & Dining", [
        "KFC CANADA #5510", "KFC RESTAURANTS OF CANADA 5510",
    ], (9.50, 27.00), None),
    "DAIRY QUEEN": ("Food & Dining", [
        "DAIRY QUEEN #217 BRAMPTON", "DQ GRILL AND CHILL 217",
    ], (5.00, 16.00), None),
    "SECOND CUP": ("Food & Dining", [
        "SECOND CUP CAFE #91", "SECOND CUP COFFEE CO 0091",
    ], (4.00, 11.50), None),
    "FRESHCO": ("Food & Dining", [
        "FRESHCO SUPERMARKET #2201", "FRESHCO #2201 SCARBOROUGH ON", "FRESHCO WEEKLY SHOP",
    ], (35.00, 165.00), None),
    "SOBEYS": ("Food & Dining", [
        "SOBEYS SUPERMARKET #0817", "SOBEYS INC 0817 HALIFAX NS", "SOBEYS EXTRA #0817",
    ], (40.00, 190.00), None),
    "NO FRILLS": ("Food & Dining", [
        "NO FRILLS #3390", "NOFRILLS ONTARIO 3390 WHITBY",
    ], (28.00, 140.00), None),
    "FARM BOY": ("Food & Dining", [
        "FARM BOY MARKET #12 OTTAWA", "FARM BOY PREPARED FOODS #12",
    ], (22.00, 95.00), "grocery_vs_prepared_food ambiguity (grocery market that also sells ready-made meals)"),

    # ---------------- Transport ----------------
    "OC TRANSPO": ("Transport", [
        "OC TRANSPO FARE PAYMENT", "OC TRANSPO FARE CARD LOAD", "OCTRANSPO OTTAWA ON",
    ], (3.50, 130.00), None),
    "TTC": ("Transport", [
        "TTC METROPASS MONTHLY", "TORONTO TRANSIT COMMISSION", "TTC FARE VENDING MACHINE",
    ], (3.35, 156.00), None),
    "VIA RAIL": ("Transport", [
        "VIA RAIL CANADA INC", "VIA RAIL TICKET TORONTO-OTTAWA",
    ], (45.00, 210.00), None),
    "BIXI BIKE SHARE": ("Transport", [
        "BIXI MONTREAL BIKE SHARE", "BIXI SUBSCRIPTION MONTHLY",
    ], (5.00, 25.00), None),
    "LYFT": ("Transport", [
        "LYFT *RIDE THU 4PM", "LYFT TRIP TORONTO ON",
    ], (9.00, 42.00), None),
    "PARK N FLY": ("Transport", [
        "PARK N FLY PEARSON AIRPORT", "PARK'N FLY YYZ LOT 3",
    ], (18.00, 120.00), None),
    "IMPARK": ("Transport", [
        "IMPARK PARKING #0093", "IMPARK MOBILE PAY TORONTO",
    ], (4.00, 22.00), None),
    "CHEVRON GAS": ("Transport", [
        "CHEVRON STATION #4421", "CHEVRON CANADA LTD 4421",
    ], (35.00, 95.00), None),
    "PIONEER GAS BAR": ("Transport", [
        "PIONEER GAS BAR #118", "PIONEER ENERGY 0118 BARRIE",
    ], (30.00, 88.00), None),
    "CITY TAXI": ("Transport", [
        "CITY TAXI CO TORONTO", "CITY TAXI DISPATCH #40",
    ], (12.00, 55.00), None),
    "HERTZ RENTAL": ("Transport", [
        "HERTZ CAR RENTAL YYZ", "HERTZ RENT A CAR TORONTO PEARSON",
    ], (65.00, 340.00), None),

    # ---------------- Rent & Utilities ----------------
    "FIDO MOBILE": ("Rent & Utilities", [
        "FIDO MOBILE MONTHLY BILL", "FIDO SOLUTIONS INC PREAUTH",
    ], (45.00, 95.00), None),
    "TELUS": ("Rent & Utilities", [
        "TELUS COMMUNICATIONS", "TELUS MOBILITY PREAUTH PYMT",
    ], (55.00, 130.00), None),
    "COGECO CABLE": ("Rent & Utilities", [
        "COGECO CABLE INTERNET", "COGECO CONNEXION MONTHLY",
    ], (60.00, 140.00), None),
    "VIRGIN PLUS": ("Rent & Utilities", [
        "VIRGIN PLUS MOBILE BILL", "VIRGIN PLUS PREAUTH PAYMENT",
    ], (40.00, 90.00), None),
    "CITY WATER UTILITY": ("Rent & Utilities", [
        "CITY WATER UTILITY BILL", "MUNICIPAL WATER SERVICES PYMT",
    ], (35.00, 110.00), None),
    "NATURAL GAS UTILITY": ("Rent & Utilities", [
        "NATURAL GAS UTILITY CO", "RESIDENTIAL GAS SERVICE BILL",
    ], (25.00, 220.00), None),
    "CONDO MAINTENANCE FEE": ("Rent & Utilities", [
        "CONDO MAINTENANCE FEE", "CONDOMINIUM CORP MONTHLY FEE",
    ], (280.00, 620.00), None),
    "PROPERTY MANAGEMENT CORP": ("Rent & Utilities", [
        "PROPERTY MANAGEMENT CORP RENT", "PMC RESIDENTIAL RENT PYMT",
    ], (1200.00, 2400.00), None),
    "RENTAL PAYMENT LANDLORD": ("Rent & Utilities", [
        "RENTAL PAYMENT E-TRFR LANDLORD", "MONTHLY RENT ETRANSFER",
    ], (1100.00, 2200.00), "generic_etransfer_description (looks like a transfer, is actually rent)"),
    "HOME INTERNET SERVICE": ("Rent & Utilities", [
        "HOME INTERNET SERVICE PROVIDER", "RESIDENTIAL INTERNET MONTHLY BILL",
    ], (65.00, 110.00), None),

    # ---------------- Entertainment ----------------
    "DISNEY PLUS": ("Entertainment", [
        "DISNEY PLUS SUBSCRIPTION", "DISNEY+ MONTHLY BILLING",
    ], (11.99, 15.99), None),
    "CRAVE TV": ("Entertainment", [
        "CRAVE TV ONLINE SERVICE", "CRAVE MONTHLY MEMBERSHIP",
    ], (9.99, 19.99), None),
    "XBOX GAME PASS": ("Entertainment", [
        "XBOX GAME PASS ULTIMATE", "XBOX GAME PASS MONTHLY RENEWAL",
    ], (10.99, 16.99), None),
    "PLAYSTATION NETWORK": ("Entertainment", [
        "PLAYSTATION NETWORK STORE", "PSN DIGITAL PURCHASE",
    ], (9.99, 79.99), None),
    "TICKETMASTER": ("Entertainment", [
        "TICKETMASTER EVENT TICKETS", "TICKETMASTER CANADA ORDER",
    ], (55.00, 220.00), None),
    "GOOGLE PLAY MOVIES": ("Entertainment", [
        "GOOGLE *PLAY MOVIES", "GOOGLE PLAY DIGITAL RENTAL",
    ], (4.99, 19.99), None),
    "APPLE TV PLUS": ("Entertainment", [
        "APPLE TV PLUS SUBSCRIPTION", "APPLE.COM/BILL TV PLUS",
    ], (8.99, 8.99), "subscriptions_vs_entertainment ambiguity (a streaming *subscription* billed like Entertainment content)"),
    "TWITCH": ("Entertainment", [
        "TWITCH SUBSCRIPTION CHANNEL", "TWITCH INTERACTIVE MONTHLY",
    ], (4.99, 24.99), None),
    "BOWLERO": ("Entertainment", [
        "BOWLERO BOWLING ALLEY", "BOWLERO LANES #08 MARKHAM",
    ], (18.00, 65.00), None),
    "ESCAPE ROOM": ("Entertainment", [
        "ESCAPE ROOM EXPERIENCE", "PUZZLE ESCAPE GAMES TORONTO",
    ], (28.00, 45.00), None),

    # ---------------- Healthcare ----------------
    "LONDON DRUGS": ("Healthcare", [
        "LONDON DRUGS STORE #0071", "LONDON DRUGS #71 VANCOUVER BC",
    ], (8.00, 65.00), None),
    "GUARDIAN DRUG MART": ("Healthcare", [
        "GUARDIAN DRUG MART #14", "GUARDIAN DRUGS DISPENSARY 14",
    ], (10.00, 80.00), None),
    "DENTIST OFFICE": ("Healthcare", [
        "DENTIST OFFICE VISIT DR PATEL", "DENTAL CARE CLINIC INVOICE",
    ], (85.00, 420.00), None),
    "PHYSIOTHERAPY CLINIC": ("Healthcare", [
        "PHYSIOTHERAPY CLINIC SESSION", "PHYSIO REHAB CENTRE INVOICE",
    ], (60.00, 150.00), None),
    "WALK IN MEDICAL CLINIC": ("Healthcare", [
        "WALK IN MEDICAL CLINIC", "URGENT CARE CLINIC VISIT FEE",
    ], (0.00, 45.00), None),
    "OPTOMETRIST EYE EXAM": ("Healthcare", [
        "OPTOMETRIST EYE EXAM FEE", "VISION CARE CENTRE EXAM",
    ], (40.00, 250.00), None),
    "MASSAGE THERAPY CLINIC": ("Healthcare", [
        "MASSAGE THERAPY CLINIC", "RMT MASSAGE SESSION INVOICE",
    ], (75.00, 130.00), None),
    "VIRTUAL DOCTOR VISIT": ("Healthcare", [
        "VIRTUAL DOCTOR VISIT PLATFORM", "ONLINE MD CONSULT FEE",
    ], (0.00, 60.00), None),

    # ---------------- Shopping ----------------
    "WALMART SUPERCENTRE": ("Shopping", [
        "WALMART SUPERCENTRE #3092", "WAL-MART #3092 KITCHENER ON", "WALMART CANADA 3092",
    ], (15.00, 210.00), "multi_purpose_merchant (big-box store selling both groceries and general merchandise)"),
    "COSTCO WHOLESALE": ("Shopping", [
        "COSTCO WHOLESALE #547", "COSTCO CANADA 0547 VAUGHAN",
    ], (45.00, 340.00), "multi_purpose_merchant (bulk groceries and general merchandise in one store)"),
    "CANADIAN TIRE": ("Shopping", [
        "CANADIAN TIRE STORE #291", "CDN TIRE #0291 GUELPH ON",
    ], (12.00, 260.00), None),
    "WINNERS RETAIL": ("Shopping", [
        "WINNERS #4415", "WINNERS MERCHANTS INTL 4415",
    ], (14.00, 90.00), None),
    "INDIGO BOOKS": ("Shopping", [
        "INDIGO BOOKS AND MUSIC", "INDIGO #212 YORKDALE",
    ], (9.00, 65.00), None),
    "SPORT CHEK": ("Shopping", [
        "SPORT CHEK #603", "SPORTCHEK RETAIL 0603",
    ], (25.00, 180.00), None),
    "HOME DEPOT": ("Shopping", [
        "HOME DEPOT #7166", "THE HOME DEPOT 7166 AJAX ON",
    ], (18.00, 420.00), None),
    "STAPLES OFFICE SUPPLY": ("Shopping", [
        "STAPLES OFFICE SUPPLY #55", "STAPLES CANADA 0055",
    ], (8.00, 140.00), None),
    "DOLLARAMA": ("Shopping", [
        "DOLLARAMA #612", "DOLLARAMA LP 0612 BRAMPTON",
    ], (3.00, 28.00), None),
    "MARSHALLS RETAIL": ("Shopping", [
        "MARSHALLS #331", "MARSHALLS DEPT STORE 0331",
    ], (12.00, 85.00), None),

    # ---------------- Subscriptions ----------------
    "NOTION APP": ("Subscriptions", [
        "NOTION LABS INC MONTHLY", "NOTION.SO SUBSCRIPTION",
    ], (10.00, 10.00), None),
    "DROPBOX": ("Subscriptions", [
        "DROPBOX CLOUD STORAGE PLAN", "DROPBOX INC MONTHLY BILL",
    ], (13.99, 13.99), None),
    "SLACK": ("Subscriptions", [
        "SLACK WORKSPACE BILLING", "SLACK TECHNOLOGIES INC",
    ], (10.00, 15.00), None),
    "ZOOM": ("Subscriptions", [
        "ZOOM PRO SUBSCRIPTION", "ZOOM VIDEO COMMUNICATIONS",
    ], (16.99, 16.99), None),
    "LINKEDIN PREMIUM": ("Subscriptions", [
        "LINKEDIN PREMIUM CAREER", "LINKEDIN CORP MONTHLY",
    ], (39.99, 39.99), None),
    "AUDIBLE MEMBERSHIP": ("Subscriptions", [
        "AUDIBLE MEMBERSHIP MONTHLY", "AUDIBLE.CA CREDIT PLAN",
    ], (14.95, 14.95), None),
    "PATREON": ("Subscriptions", [
        "PATREON CREATOR SUPPORT", "PATREON MEMBERSHIP MONTHLY",
    ], (5.00, 25.00), None),
    "GYM MEMBERSHIP FEE": ("Subscriptions", [
        "GYM MEMBERSHIP FEE MONTHLY", "FITNESS CLUB PREAUTH PYMT",
    ], (35.00, 75.00), None),
    "CANVA PRO": ("Subscriptions", [
        "CANVA PRO SUBSCRIPTION", "CANVA.COM MONTHLY BILLING",
    ], (14.99, 14.99), "subscriptions_vs_shopping ambiguity (a software subscription that could read like a design/retail purchase)"),

    # ---------------- Other ----------------
    "INTERAC ETRANSFER FEE": ("Other", [
        "INTERAC E-TRANSFER FEE", "INTERAC ETRFR SERVICE CHARGE",
    ], (1.00, 1.50), None),
    "NSF FEE": ("Other", [
        "NSF FEE CHARGE", "NON-SUFFICIENT FUNDS FEE",
    ], (45.00, 48.00), None),
    "OVERDRAFT SERVICE CHARGE": ("Other", [
        "OVERDRAFT SERVICE CHARGE", "OD PROTECTION FEE MONTHLY",
    ], (5.00, 5.00), None),
    "CASH WITHDRAWAL": ("Other", [
        "CASH WITHDRAWAL DEBIT", "CASH WITHDRAWAL BRANCH #221",
    ], (20.00, 300.00), "generic_description (cash withdrawal carries no merchant signal at all)"),
    "WIRE TRANSFER FEE": ("Other", [
        "WIRE TRANSFER FEE OUTGOING", "INTL WIRE TRANSFER SERVICE CHARGE",
    ], (15.00, 45.00), None),
    "UNKNOWN MERCHANT POS": ("Other", [
        "POS PURCHASE REF 88213", "MISC DEBIT TRANSACTION 44120", "PREAUTH PYMT 445123",
    ], (5.00, 120.00), "malformed_low_information description (no recognizable merchant name at all)"),
    "MONTHLY ACCOUNT FEE": ("Other", [
        "MONTHLY ACCOUNT FEE", "ACCOUNT MAINTENANCE FEE",
    ], (4.00, 16.95), None),
    "REFUND RETURNED ITEM": ("Other", [
        "REFUND - RETURNED ITEM", "MERCHANT REFUND CREDIT ADJ",
    ], (-85.00, -12.00), "refund_credit (negative amount; V1's ingest.py does not net debits/credits)"),
    "ETRFR SENT GENERIC": ("Other", [
        "E TRFR SENT", "SENT E-TRANSFER",
    ], (20.00, 400.00), "generic_transfer_description (could be rent, could be a gift, could be anything)"),
}


def build() -> pd.DataFrame:
    rng = random.Random(SEED)
    rows = []
    total_span_days = (DATE_END - DATE_START).days

    for merchant_group, (category, variants, (amt_lo, amt_hi), notes) in MERCHANT_GROUPS.items():
        # Each variant description appears once-to-twice, so groups have
        # between 2 and ~6 rows -- deliberately uneven, like real recurring
        # vs. occasional merchants.
        for variant in variants:
            n_occurrences = 1 if amt_lo == amt_hi else rng.choice([1, 1, 2])
            for _ in range(n_occurrences):
                offset_days = rng.randint(0, total_span_days)
                date = DATE_START + pd.Timedelta(days=offset_days)
                amount = round(rng.uniform(amt_lo, amt_hi), 2)
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "description": variant,
                    "amount": amount,
                    "merchant_group": merchant_group,
                    "true_category": category,
                    "error_analysis_tag": notes or "",
                })

    out = pd.DataFrame(rows).sort_values(["merchant_group", "date"]).reset_index(drop=True)
    return out


def main() -> None:
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows across {df['merchant_group'].nunique()} merchant groups to {OUT_PATH}")
    print(df["true_category"].value_counts())


if __name__ == "__main__":
    main()
