"""
Phase 3: Generate data/raw/synthetic_24mo.csv.
24 months (Jan 2023 – Dec 2024) with temporal structure for forecasting.
Top-down: generate monthly totals first, then distribute into transactions.
Same merchants as 12mo for Phase 2 clustering compatibility. Seed 42.
"""
import random
from datetime import datetime
from pathlib import Path

import pandas as pd

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "raw" / "synthetic_24mo.csv"

MERCHANTS = {
    "Food & Dining": [
        "TIM HORTONS COFFEE DRIVE THRU",
        "MCDONALDS BURGER FAST FOOD",
        "SUBWAY SANDWICH FRESH",
        "LOBLAWS GROCER WEEKLY",
        "METRO GROCER FRESH PRODUCE",
    ],
    "Transport": [
        "UBER RIDESHARE TRIP",
        "PRESTO TRANSIT RELOAD",
        "SHELL GASOLINE FUEL PUMP",
        "ESSO PETRO FUEL PUMP",
        "GO TRANSIT RAIL PASS",
    ],
    "Rent & Utilities": [
        "ROGERS WIRELESS PHONE",
        "BELL INTERNET BROADBAND",
        "HYDRO ONE ELECTRICITY GRID",
        "ENBRIDGE GAS HEAT",
        "TORONTO HYDRO ELECTRICITY",
    ],
    "Entertainment": [
        "NETFLIX STREAMING CINEMA",
        "SPOTIFY MUSIC STREAMING",
        "STEAM GAMING DOWNLOAD",
        "CINEPLEX CINEMA TICKET",
        "AMAZON PRIME STREAMING",
    ],
    "Healthcare": [
        "SHOPPERS PHARMACY PRESCRIPTION",
        "REXALL PHARMACY PRESCRIPTION",
        "MAPLE TELEHEALTH VIRTUAL",
        "TELEHEALTH VIRTUAL PRESCRIPTION",
    ],
    "Shopping": [
        "AMAZON RETAIL APPAREL",
        "ZARA APPAREL FASHION",
        "HM APPAREL CLOTHING",
        "IKEA FURNITURE RETAIL",
        "BESTBUY ELECTRONICS RETAIL",
    ],
    "Subscriptions": [
        "ADOBE CREATIVE SUITE",
        "MICROSOFT OFFICE SUITE",
        "ICLOUD BACKUP SUITE",
        "YOUTUBE PREMIUM SUITE",
    ],
    "Other": [
        "ATM CASH WITHDRAWAL",
        "MISCELLANEOUS EXPENSE",
        "BANK FEE PENALTY",
    ],
}

SUBSCRIPTION_AMOUNTS = {
    "ADOBE CREATIVE SUITE": 14.99,
    "MICROSOFT OFFICE SUITE": 12.99,
    "ICLOUD BACKUP SUITE": 3.99,
    "YOUTUBE PREMIUM SUITE": 13.99,
}

UTILITY_BASE_AMOUNTS = {
    "ROGERS WIRELESS PHONE": 85.00,
    "BELL INTERNET BROADBAND": 79.99,
    "HYDRO ONE ELECTRICITY GRID": 120.00,
    "ENBRIDGE GAS HEAT": 95.00,
    "TORONTO HYDRO ELECTRICITY": 110.00,
}

AMOUNT_RANGES = {
    "Food & Dining": (8, 120),
    "Transport": (5, 80),
    "Entertainment": (10, 60),
    "Healthcare": (15, 90),
    "Shopping": (20, 200),
    "Other": (20, 100),
}

TXN_COUNTS = {
    "Food & Dining": (8, 12),
    "Transport": (3, 6),
    "Entertainment": (2, 4),
    "Healthcare": (1, 3),
    "Shopping": (2, 5),
    "Other": (1, 3),
}


def random_date_in_month(year: int, month: int) -> str:
    if month in (4, 6, 9, 11):
        day = random.randint(1, 30)
    elif month == 2:
        day = random.randint(1, 28)
    else:
        day = random.randint(1, 31)
    return datetime(year, month, day).strftime("%m/%d/%Y")


def get_year_month(month_idx: int):
    """Convert 0-based month index to (year, month). Starts Jan 2023."""
    year = 2023 + month_idx // 12
    month = (month_idx % 12) + 1
    return year, month


def generate_monthly_totals():
    """Generate 24 monthly totals per variable category with temporal structure."""
    totals = {}

    # Food & Dining: autocorrelated ±15%, December ×1.25
    prev = 450.0
    food = []
    for m_idx in range(24):
        _, month = get_year_month(m_idx)
        dec_mult = 1.25 if month == 12 else 1.0
        change = random.uniform(-0.15, 0.15)
        val = prev * (1 + change) * dec_mult
        val = max(250, min(val, 700))
        food.append(round(val, 2))
        prev = val / dec_mult
    totals["Food & Dining"] = food

    # Transport: summer ×1.2, base ~150
    transport = []
    for m_idx in range(24):
        _, month = get_year_month(m_idx)
        summer_mult = 1.2 if month in [6, 7, 8] else 1.0
        val = 150 * summer_mult * random.uniform(0.85, 1.15)
        transport.append(round(val, 2))
    totals["Transport"] = transport

    # Entertainment: December ×1.3, base ~100
    ent = []
    for m_idx in range(24):
        _, month = get_year_month(m_idx)
        dec_mult = 1.3 if month == 12 else 1.0
        val = 100 * dec_mult * random.uniform(0.85, 1.15)
        ent.append(round(val, 2))
    totals["Entertainment"] = ent

    # Healthcare: moderate ±20%, base ~120
    health = []
    for m_idx in range(24):
        val = 120 * random.uniform(0.80, 1.20)
        health.append(round(val, 2))
    totals["Healthcare"] = health

    # Shopping: December ×1.3, base ~300, ±20%
    shop = []
    for m_idx in range(24):
        _, month = get_year_month(m_idx)
        dec_mult = 1.3 if month == 12 else 1.0
        val = 300 * dec_mult * random.uniform(0.80, 1.20)
        shop.append(round(val, 2))
    totals["Shopping"] = shop

    # Other: moderate ±20%, base ~120
    other = []
    for m_idx in range(24):
        val = 120 * random.uniform(0.80, 1.20)
        other.append(round(val, 2))
    totals["Other"] = other

    return totals


def distribute_total(target, n_txn, lo, hi, merchants):
    """Split a monthly category total into individual transactions."""
    avg = target / n_txn
    amounts = []
    for _ in range(n_txn - 1):
        jitter = random.uniform(0.7, 1.3)
        amt = max(lo, min(hi, avg * jitter))
        amounts.append(round(amt, 2))
    remainder = target - sum(amounts)
    amounts.append(round(max(lo, min(hi, remainder)), 2))

    txns = []
    for amt in amounts:
        txns.append({"Description": random.choice(merchants), "Amount": amt})
    return txns


rows = []
monthly_totals = generate_monthly_totals()

for m_idx in range(24):
    year, month = get_year_month(m_idx)

    # Subscriptions: fixed merchants, near-constant amounts (±2% noise)
    for merchant, base_amt in SUBSCRIPTION_AMOUNTS.items():
        amt = round(base_amt * random.uniform(0.98, 1.02), 2)
        rows.append({
            "Date": random_date_in_month(year, month),
            "Description": merchant,
            "Amount": amt,
        })

    # Rent & Utilities: fixed merchants, near-constant with winter heating uplift
    winter_mult = 1.08 if month in [11, 12, 1, 2] else 1.0
    util_merchants = random.sample(list(UTILITY_BASE_AMOUNTS.keys()),
                                   random.choice([3, 4]))
    for merch in util_merchants:
        base = UTILITY_BASE_AMOUNTS[merch]
        amt = round(base * winter_mult * random.uniform(0.92, 1.08), 2)
        rows.append({
            "Date": random_date_in_month(year, month),
            "Description": merch,
            "Amount": amt,
        })

    # Variable categories: distribute monthly total into transactions
    for cat in ["Food & Dining", "Transport", "Entertainment",
                "Healthcare", "Shopping", "Other"]:
        target = monthly_totals[cat][m_idx]
        lo, hi = AMOUNT_RANGES[cat]
        n_txn = random.randint(*TXN_COUNTS[cat])
        txns = distribute_total(target, n_txn, lo, hi, MERCHANTS[cat])
        for txn in txns:
            txn["Date"] = random_date_in_month(year, month)
            rows.append(txn)

random.shuffle(rows)
df = pd.DataFrame(rows)[["Date", "Description", "Amount"]]
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print(f"Wrote {len(df)} rows to {OUT_PATH}")
print(f"Date range: {sorted(df['Date'].unique())[0]} to {sorted(df['Date'].unique())[-1]}")
print(f"\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

print(f"\nRow count by month (should span 24 months):")
df["_month"] = pd.to_datetime(df["Date"], format="%m/%d/%Y").dt.to_period("M")
print(df.groupby("_month").size().to_string())
