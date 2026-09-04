"""
ML-F: deployment-oriented categorization benchmark.

WHAT THIS IS:
  A hand-authored, PRIVACY-SAFE corpus that models the *structural* patterns
  ML-F-A's private real-export audit found in actual RBC and Scotiabank
  exports -- transaction-method boilerplate prefixes (VISA DEBIT PURCHASE -,
  POS PURCHASE / Opos-prefixed, CONTACTLESS INTERAC PURCHASE -), card-suffix
  and reference-code noise, Scotiabank-style mid-word truncation with an
  appended numeric suffix, and genuinely purpose-less row types (generic
  Interac e-transfer sends, ABM/ATM withdrawals) -- but populated entirely
  with FABRICATED merchant names invented for this file. No literal private
  transaction description, account number, balance, or private name from the
  audit's real-export inputs appears anywhere below.

WHY A SECOND CORPUS (ML-F-A audit, Section D/§20): Tier B
(data/evaluation/tier_b_benchmark.csv) is a real, useful benchmark, but its
133-row TRAIN partition is what the current production vocabulary (50 words)
was fit on, and the audit found real-export merchant identities barely
overlap it. This corpus exists to give the categorizer TRAINING coverage of
the boilerplate/truncation shapes real bank exports actually use, not just a
held-out evaluation set. It is the PRIMARY dataset for ML-F candidate
selection; Tier B remains a SECONDARY continuity benchmark (still evaluated,
never used to pick the winner).

Precise terminology: this is a "sanitized deployment-oriented bank-description
benchmark." It is not real-world data and must never be described as such.

AMBIGUOUS ROWS (ML-F-A audit §14/§16): a small set of rows carry NO
spending-purpose signal by construction -- a generic Interac e-transfer send,
an ABM/ATM cash withdrawal, a bare pre-authorized-payment reference code with
no merchant name. These are marked `is_ambiguous=True` with `true_category`
left blank (never a fabricated spending-purpose label) so the bake-off can
evaluate them as a separate abstention/routing question rather than folding a
made-up label into 8-class macro-F1.

MERCHANT GROUPING FOR LEAKAGE-SAFE SPLITTING: identical discipline to Tier B
(ml/data/build_tier_b_benchmark.py) -- `merchant_group` is the normalized
identity that must not be split across TRAIN/VALIDATION/FINAL_TEST.

REPRODUCIBILITY: only DATE placement, small AMOUNT jitter, and which
boilerplate template/card-suffix/reference-code decorates a given row are
seeded/deterministic (SEED=42). No merchant, category, or ambiguity flag is
chosen at random.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SEED = 42
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "deployment_benchmark.csv"

DATE_START = pd.Timestamp("2024-01-01")
DATE_END = pd.Timestamp("2025-12-31")

AMBIGUOUS_CATEGORY = ""  # true_category left blank for is_ambiguous=True rows

# ---------------------------------------------------------------------------
# Deployment-shaped boilerplate templates (generalized from the ML-F-A real-
# export audit's structural buckets -- never the literal private text).
# Each returns a full `description` string given a bare merchant/name string.
# ---------------------------------------------------------------------------

def _visa_debit(rng: random.Random, name: str) -> str:
    suffix = rng.randint(1000, 9999)
    return f"VISA DEBIT PURCHASE - {suffix} {name}"


def _contactless(rng: random.Random, name: str) -> str:
    suffix = rng.randint(1000, 9999)
    return f"CONTACTLESS INTERAC PURCHASE - {suffix} {name}"


def _pos_opos(rng: random.Random, name: str) -> str:
    return f"pos purchase Opos {name} "


def _scotia_truncated(rng: random.Random, name: str) -> str:
    """Mimics the real Scotiabank export's observed behavior: a longer
    service/plan name cut short mid-word, then a numeric suffix appended --
    never the literal private merchant this pattern was originally noticed
    on, just the structural shape (ML-F-A audit, structural signature
    buckets)."""
    truncated = name[:14].rstrip()
    phone = rng.randint(1000, 9999)
    return f"Opos {truncated}+{phone} "


def _plain_store(rng: random.Random, name: str) -> str:
    store_no = rng.randint(1, 999)
    return f"{name} #{store_no:04d}"


def _online(rng: random.Random, name: str) -> str:
    return f"ONLINE PURCHASE {name.replace(' ', '')}.COM"


def _preauth(rng: random.Random, name: str) -> str:
    ref = rng.randint(100000, 999999)
    return f"{name} PREAUTH PYMT {ref}"


TEMPLATES = [_visa_debit, _contactless, _pos_opos, _scotia_truncated, _plain_store, _online, _preauth]

# ---------------------------------------------------------------------------
# Fabricated merchant groups. category="" + is_ambiguous=True for rows with
# no recoverable spending-purpose signal.
#
# Each entry: merchant_group -> (category, [bare names], template_indices,
#                                 (amount_lo, amount_hi), is_ambiguous, notes)
# ---------------------------------------------------------------------------

GROUPS: dict[str, tuple[str, list[str], list[int], tuple[float, float], bool, str | None]] = {
    # ---------------- Food & Dining ----------------
    "MAPLEWOOD DINER": ("Food & Dining", ["MAPLEWOOD DINER"], [0, 2, 5], (9.0, 32.0), False, None),
    "NORTHSIDE PIZZA CO": ("Food & Dining", ["NORTHSIDE PIZZA CO"], [0, 4, 5], (14.0, 40.0), False, None),
    "GOLDEN WOK EXPRESS": ("Food & Dining", ["GOLDEN WOK EXPRESS"], [1, 2], (11.0, 34.0), False, None),
    "RIVERSIDE BAKERY": ("Food & Dining", ["RIVERSIDE BAKERY"], [0, 4], (4.0, 18.0), False, None),
    "HARBOUR FISH AND CHIPS": ("Food & Dining", ["HARBOUR FISH AND CHIPS"], [2, 4], (12.0, 29.0), False, None),
    "BRICKYARD BURGERS": ("Food & Dining", ["BRICKYARD BURGERS"], [0, 1, 4], (8.0, 24.0), False, None),
    "LAKEVIEW GROCERY MARKET": ("Food & Dining", ["LAKEVIEW GROCERY MARKET"], [4, 5], (35.0, 175.0), False, None),
    "CEDAR GROCERS": ("Food & Dining", ["CEDAR GROCERS"], [0, 4], (30.0, 160.0), False, None),
    "SUNRISE CAFE": ("Food & Dining", ["SUNRISE CAFE"], [0, 1, 2], (3.5, 11.0), False, None),
    "PRAIRIE FARMS MARKET": (
        "Food & Dining", ["PRAIRIE FARMS MARKET"], [4, 6], (20.0, 90.0), False,
        "grocery_vs_prepared_food ambiguity (grocery market that also sells ready-made meals), mirrors Tier B's FARM BOY case",
    ),

    # ---------------- Transport ----------------
    "TRANSITLINK FARE": ("Transport", ["TRANSITLINK FARE"], [4, 6], (3.25, 130.0), False, None),
    "CITYLINE BUS PASS": ("Transport", ["CITYLINE BUS PASS"], [4], (3.25, 156.0), False, None),
    "SKYWAY COMMUTER RAIL": ("Transport", ["SKYWAY COMMUTER RAIL"], [0, 4], (6.0, 42.0), False, None),
    "QUICKPARK GARAGE": ("Transport", ["QUICKPARK GARAGE"], [0, 4], (4.0, 24.0), False, None),
    "NORTHSTAR TAXI": ("Transport", ["NORTHSTAR TAXI"], [0, 2], (10.0, 55.0), False, None),
    "RAPIDRENT CAR RENTAL": ("Transport", ["RAPIDRENT CAR RENTAL"], [0, 4], (60.0, 340.0), False, None),
    "MAPLE FUEL STATION": ("Transport", ["MAPLE FUEL STATION"], [0, 4], (35.0, 95.0), False, None),
    "HARBOUR FERRY SERVICE": ("Transport", ["HARBOUR FERRY SERVICE"], [4, 6], (8.0, 45.0), False, None),
    "GREENWAY BIKE SHARE": ("Transport", ["GREENWAY BIKE SHARE"], [0, 6], (4.0, 22.0), False, None),

    # ---------------- Rent & Utilities ----------------
    "BRIGHTWAVE INTERNET": ("Rent & Utilities", ["BRIGHTWAVE INTERNET"], [6, 0], (60.0, 110.0), False, None),
    "NORTHGRID POWER CO": ("Rent & Utilities", ["NORTHGRID POWER CO"], [6, 0], (35.0, 220.0), False, None),
    "CITYLINE WATER UTILITY": ("Rent & Utilities", ["CITYLINE WATER UTILITY"], [6, 0], (30.0, 105.0), False, None),
    "SUMMIT PROPERTY MGMT RENT": (
        "Rent & Utilities", ["SUMMIT PROPERTY MGMT RENT"], [6, 0], (1150.0, 2300.0), False,
        "generic_etransfer_description would apply if paid via e-transfer, but this template is the pre-authorized-payment path, which does carry a merchant/landlord name",
    ),
    "CLEARTEL MOBILE": ("Rent & Utilities", ["CLEARTEL MOBILE"], [6, 0], (45.0, 95.0), False, None),
    "HEARTHSTONE CONDO FEES": ("Rent & Utilities", ["HEARTHSTONE CONDO FEES"], [6, 0], (280.0, 620.0), False, None),
    "VALLEY GAS UTILITY": ("Rent & Utilities", ["VALLEY GAS UTILITY"], [6, 0], (25.0, 210.0), False, None),
    "PRAIRIE TELECOM": ("Rent & Utilities", ["PRAIRIE TELECOM"], [6, 0], (50.0, 130.0), False, None),
    "HOMEGRID INSURANCE PYMT": ("Rent & Utilities", ["HOMEGRID INSURANCE PYMT"], [6, 0], (40.0, 160.0), False, None),

    # ---------------- Entertainment ----------------
    "SILVERSCREEN CINEMAS": ("Entertainment", ["SILVERSCREEN CINEMAS"], [0, 4], (10.0, 45.0), False, None),
    "ARCADE ZONE": ("Entertainment", ["ARCADE ZONE"], [0, 2], (8.0, 35.0), False, None),
    "TICKETVAULT EVENTS": ("Entertainment", ["TICKETVAULT EVENTS"], [5], (35.0, 220.0), False, None),
    "STARLIGHT BOWL LANES": ("Entertainment", ["STARLIGHT BOWL LANES"], [0, 4], (18.0, 65.0), False, None),
    "PUZZLEROOM ESCAPE GAMES": ("Entertainment", ["PUZZLEROOM ESCAPE GAMES"], [0, 5], (28.0, 46.0), False, None),
    "COMEDY LOFT TICKETS": ("Entertainment", ["COMEDY LOFT TICKETS"], [5], (25.0, 75.0), False, None),
    "PIXELPLAY GAME STORE": ("Entertainment", ["PIXELPLAY GAME STORE"], [0, 4], (9.99, 79.99), False, None),
    "MOVIENIGHT DIGITAL RENTAL": ("Entertainment", ["MOVIENIGHT DIGITAL RENTAL"], [5], (4.99, 19.99), False, None),

    # ---------------- Healthcare ----------------
    "CAREWELL PHARMACY": ("Healthcare", ["CAREWELL PHARMACY"], [0, 4], (6.0, 65.0), False, None),
    "BRIGHT SMILE DENTAL": ("Healthcare", ["BRIGHT SMILE DENTAL"], [6], (85.0, 420.0), False, None),
    "VITAL PHYSIO CLINIC": ("Healthcare", ["VITAL PHYSIO CLINIC"], [6], (60.0, 150.0), False, None),
    "QUICKCARE WALK-IN CLINIC": ("Healthcare", ["QUICKCARE WALK-IN CLINIC"], [6], (0.0, 45.0), False, None),
    "CLEARVIEW OPTICAL": ("Healthcare", ["CLEARVIEW OPTICAL"], [0, 4], (40.0, 250.0), False, None),
    "HANDS-ON MASSAGE THERAPY": ("Healthcare", ["HANDS-ON MASSAGE THERAPY"], [6, 0], (75.0, 130.0), False, None),
    "TELEHEALTH VIRTUAL VISIT": ("Healthcare", ["TELEHEALTH VIRTUAL VISIT"], [5, 3], (0.0, 60.0), False, None),
    "GREENLEAF NATUROPATHIC CLINIC": ("Healthcare", ["GREENLEAF NATUROPATHIC CLINIC"], [6, 0], (55.0, 140.0), False, None),

    # ---------------- Shopping ----------------
    "VALUEMART DEPT STORE": (
        "Shopping", ["VALUEMART DEPT STORE"], [0, 4], (15.0, 210.0), False,
        "multi_purpose_merchant (big-box store selling both groceries and general merchandise), mirrors Tier B's WALMART case",
    ),
    "HOMEBASE HARDWARE": ("Shopping", ["HOMEBASE HARDWARE"], [0, 4], (12.0, 260.0), False, None),
    "TRENDLINE APPAREL": ("Shopping", ["TRENDLINE APPAREL"], [0, 2], (14.0, 90.0), False, None),
    "BARGAIN BIN DISCOUNT": ("Shopping", ["BARGAIN BIN DISCOUNT"], [0, 4], (3.0, 28.0), False, None),
    "PAGEBOUND BOOKS": ("Shopping", ["PAGEBOUND BOOKS"], [0, 5], (9.0, 65.0), False, None),
    "ACTIVEGEAR SPORTS": ("Shopping", ["ACTIVEGEAR SPORTS"], [0, 4], (25.0, 180.0), False, None),
    "BUILDRIGHT HOME IMPROVEMENT": ("Shopping", ["BUILDRIGHT HOME IMPROVEMENT"], [0, 4], (18.0, 420.0), False, None),
    "OFFICEPLUS SUPPLY": ("Shopping", ["OFFICEPLUS SUPPLY"], [0, 4], (8.0, 140.0), False, None),

    # ---------------- Subscriptions ----------------
    "CLOUDDESK WORKSPACE": ("Subscriptions", ["CLOUDDESK WORKSPACE"], [5, 3], (10.0, 10.0), False, None),
    "FOCUSFLOW PRODUCTIVITY APP": ("Subscriptions", ["FOCUSFLOW PRODUCTIVITY APP"], [5, 3], (8.99, 8.99), False, None),
    "PIXELCRAFT DESIGN SUB": (
        "Subscriptions", ["PIXELCRAFT DESIGN SUB"], [5, 3], (14.99, 14.99), False,
        "subscriptions_vs_shopping ambiguity, mirrors Tier B's CANVA PRO case",
    ),
    "AUDIOWAVE PODCAST PLAN": ("Subscriptions", ["AUDIOWAVE PODCAST PLAN"], [5, 3], (14.95, 14.95), False, None),
    "FITZONE GYM MEMBERSHIP": ("Subscriptions", ["FITZONE GYM MEMBERSHIP"], [6], (35.0, 75.0), False, None),
    "CODEFORGE DEV TOOLS": ("Subscriptions", ["CODEFORGE DEV TOOLS"], [5, 3], (12.0, 25.0), False, None),
    "BOOKSTACK AUDIOBOOK PLAN": ("Subscriptions", ["BOOKSTACK AUDIOBOOK PLAN"], [5, 3], (14.95, 14.95), False, None),
    "TEAMSYNC CHAT SUB": ("Subscriptions", ["TEAMSYNC CHAT SUB"], [5, 3], (10.0, 15.0), False, None),
    "STREAMBOX PLUS": (
        "Subscriptions", ["STREAMBOX PLUS"], [5, 3], (11.99, 15.99), False,
        "subscriptions_vs_entertainment ambiguity, mirrors Tier B's DISNEY PLUS case",
    ),

    # ---------------- Other (identifiable, but genuinely "Other") ----------------
    "MONTHLY ACCOUNT SERVICE FEE": ("Other", ["MONTHLY ACCOUNT SERVICE FEE"], [6, 0], (4.0, 16.95), False, None),
    "OVERDRAFT INTEREST CHARGE": ("Other", ["OVERDRAFT INTEREST CHARGE"], [6, 0], (0.25, 5.0), False, None),
    "WIRE TRANSFER SERVICE FEE": ("Other", ["WIRE TRANSFER SERVICE FEE"], [6, 0], (15.0, 45.0), False, None),
    "NSF RETURNED ITEM FEE": ("Other", ["NSF RETURNED ITEM FEE"], [6, 0], (45.0, 48.0), False, None),
    "FOREIGN EXCHANGE FEE": ("Other", ["FOREIGN EXCHANGE FEE"], [6, 0], (1.0, 12.0), False, None),
    "ACCOUNT REACTIVATION FEE": ("Other", ["ACCOUNT REACTIVATION FEE"], [6, 0], (10.0, 25.0), False, None),
    "CHEQUE ORDER FEE": ("Other", ["CHEQUE ORDER FEE"], [6, 0], (8.0, 20.0), False, None),

    # ---------------- Ambiguous (no spending-purpose signal at all) ----------------
    "GENERIC ETRANSFER SENT": (
        AMBIGUOUS_CATEGORY, ["E-TRANSFER SENT", "Free Interac E-Transfer", "INTERAC E-TRANSFER SENT"],
        [0], (10.0, 800.0), True,
        "generic_transfer_description (ML-F-A audit §14: could be rent, a gift, a friend repayment -- no purpose signal in the text)",
    ),
    "GENERIC ABM WITHDRAWAL": (
        AMBIGUOUS_CATEGORY, ["ABM WITHDRAWAL", "abm withdrawal"], [0], (20.0, 300.0), True,
        "generic_description (ML-F-A audit §14: cash withdrawal carries no merchant/purpose signal at all)",
    ),
    "GENERIC ATM WITHDRAWAL": (
        AMBIGUOUS_CATEGORY, ["ATM WITHDRAWAL", "CASH WITHDRAWAL"], [0], (20.0, 300.0), True, None,
    ),
    "GENERIC PREAUTH REFERENCE ONLY": (
        AMBIGUOUS_CATEGORY, ["PREAUTH PYMT", "MISC DEBIT TRANSACTION"], [0], (5.0, 120.0), True,
        "malformed_low_information description (no recognizable merchant name at all), mirrors Tier B's UNKNOWN MERCHANT POS case",
    ),
    "GENERIC ONLINE BANKING TRANSFER": (
        AMBIGUOUS_CATEGORY, ["ONLINE BANKING TRANSFER", "ONLINE TRANSFER TO DEPOSIT ACCOUNT"], [0], (30.0, 900.0), True,
        "internal_transfer_description (moving money between the user's own accounts -- not a spending-purpose transaction at all)",
    ),
}


def build() -> pd.DataFrame:
    rng = random.Random(SEED)
    rows = []
    total_span_days = (DATE_END - DATE_START).days

    for merchant_group, (category, names, template_idxs, (amt_lo, amt_hi), is_ambiguous, notes) in GROUPS.items():
        for name in names:
            for t_idx in template_idxs:
                template = TEMPLATES[t_idx]
                n_occurrences = 1 if amt_lo == amt_hi else rng.choice([1, 1, 2])
                for _ in range(n_occurrences):
                    offset_days = rng.randint(0, total_span_days)
                    date = DATE_START + pd.Timedelta(days=offset_days)
                    amount = round(rng.uniform(amt_lo, amt_hi), 2)
                    description = template(rng, name)
                    rows.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "description": description,
                        "amount": amount,
                        "merchant_group": merchant_group,
                        "true_category": category,
                        "is_ambiguous": is_ambiguous,
                        "error_analysis_tag": notes or "",
                    })

    out = pd.DataFrame(rows).sort_values(["merchant_group", "date"]).reset_index(drop=True)
    return out


def main() -> None:
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    n_ambiguous = int(df["is_ambiguous"].sum())
    print(f"Wrote {len(df)} rows across {df['merchant_group'].nunique()} merchant groups to {OUT_PATH}")
    print(f"  of which {n_ambiguous} rows ({n_ambiguous / len(df):.1%}) are is_ambiguous=True")
    print(df.loc[~df["is_ambiguous"], "true_category"].value_counts())


if __name__ == "__main__":
    main()
