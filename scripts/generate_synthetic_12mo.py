"""
Phase 1: Generate data/raw/synthetic_12mo.csv.
One-off generator; run once. Seed 42 for reproducibility.
"""
import random
from datetime import datetime
from pathlib import Path

import pandas as pd

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "raw" / "synthetic_12mo.csv"

MERCHANTS = {
    "Food & Dining": ["Tim Hortons", "McDonald's", "Subway", "Loblaws", "Metro"],
    "Transport": ["Uber", "Presto", "Shell", "Esso", "GO Transit"],
    "Rent & Utilities": ["Rogers", "Bell", "Hydro One", "Enbridge", "Toronto Hydro"],
    "Entertainment": ["Netflix", "Spotify", "Steam", "Cineplex", "Amazon Prime"],
    "Healthcare": ["Shoppers Drug Mart", "Rexall", "Maple", "Telehealth"],
    "Shopping": ["Amazon", "Zara", "H&M", "IKEA", "Best Buy"],
    "Subscriptions": ["Adobe", "Microsoft 365", "iCloud", "YouTube Premium"],
    "Other": ["ATM Withdrawal", "Miscellaneous", "Bank Fee"],
}

AMOUNT_RANGES = {
    "Food & Dining": (8, 120),
    "Transport": (5, 80),
    "Rent & Utilities": (80, 180),
    "Entertainment": (10, 60),
    "Healthcare": (15, 90),
    "Shopping": (20, 200),
    "Subscriptions": (10, 20),
    "Other": (20, 100),
}

MONTH_MULTIPLIER = {
    1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
    6: 0.85, 7: 0.85, 8: 0.85,
    9: 1.0, 10: 1.0, 11: 1.0,
    12: 1.4,
}


def random_date_in_month(year: int, month: int) -> str:
    """Return MM/DD/YYYY for a random day in the month."""
    if month in (4, 6, 9, 11):
        day = random.randint(1, 30)
    elif month == 2:
        day = random.randint(1, 29)
    else:
        day = random.randint(1, 31)
    d = datetime(year, month, day)
    return d.strftime("%m/%d/%Y")


def amount_for_category(cat: str, month: int, is_recurring: bool = False) -> float:
    lo, hi = AMOUNT_RANGES[cat]
    base = random.uniform(lo, hi) if not is_recurring else random.uniform(lo, hi)
    if is_recurring and cat == "Subscriptions":
        base = round(base, 2)
    mult = MONTH_MULTIPLIER.get(month, 1.0)
    if cat in ("Rent & Utilities", "Subscriptions"):
        mult = 1.0
    return round(base * mult, 2)


rows = []
year = 2024

# Recurring: Rent & Utilities 3-4 per month, Subscriptions same each month
rent_merchants = MERCHANTS["Rent & Utilities"]
sub_merchants = MERCHANTS["Subscriptions"]
rent_amounts = [round(random.uniform(80, 180), 2) for _ in range(4)]
sub_amounts = [round(random.uniform(10, 20), 2) for _ in range(4)]
for month in range(1, 13):
    for _ in range(random.randint(3, 4)):
        m = random.choice(rent_merchants)
        amt = rent_amounts[len(rows) % 4]
        rows.append({"Date": random_date_in_month(year, month), "Description": m, "Amount": amt})
    for i, m in enumerate(sub_merchants):
        rows.append({"Date": random_date_in_month(year, month), "Description": m, "Amount": sub_amounts[i]})

# Fill to 300-400 with other categories
categories_for_random = ["Food & Dining", "Transport", "Entertainment", "Healthcare", "Shopping", "Other"]
target_total = random.randint(300, 400)
while len(rows) < target_total:
    cat = random.choice(categories_for_random)
    merchant = random.choice(MERCHANTS[cat])
    month = random.randint(1, 12)
    amt = amount_for_category(cat, month, is_recurring=False)
    rows.append({"Date": random_date_in_month(year, month), "Description": merchant, "Amount": amt})

random.shuffle(rows)
df = pd.DataFrame(rows)
df = df[["Date", "Description", "Amount"]]
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(df)} rows to {OUT_PATH}")
print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))
