"""
Phase 4: Seed all 6 SQLite tables with demo data.
Standalone script — never imported by main.py or any other module.
Idempotent: DELETE FROM all tables before inserting.
Run: python db/seed_synthetic_data.py
"""
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CATEGORIES
from db.database import (
    get_connection,
    insert_forecast_vs_actual,
    insert_portfolio,
    insert_predictions,
    insert_transactions,
    upsert_monthly_summary,
    upsert_price_cache,
)

import pandas as pd

random.seed(42)

SESSION_ID = "SEED_001"

MERCHANTS = {
    "Food & Dining": ["TIM HORTONS", "MCDONALDS", "SUBWAY", "LOBLAWS", "METRO"],
    "Transport": ["UBER", "PRESTO", "SHELL", "ESSO", "GO TRANSIT"],
    "Rent & Utilities": ["ROGERS", "BELL", "HYDRO ONE", "ENBRIDGE", "TORONTO HYDRO"],
    "Entertainment": ["NETFLIX", "SPOTIFY", "STEAM", "CINEPLEX", "AMAZON PRIME"],
    "Healthcare": ["SHOPPERS", "REXALL", "MAPLE CLINIC", "TELEHEALTH"],
    "Shopping": ["AMAZON", "ZARA", "H&M", "IKEA", "BEST BUY"],
    "Subscriptions": ["ADOBE", "MICROSOFT 365", "ICLOUD", "YOUTUBE PREMIUM"],
    "Other": ["ATM WITHDRAWAL", "MISCELLANEOUS", "BANK FEE"],
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


def generate_seed_transactions() -> pd.DataFrame:
    """Generate 12 months of demo transactions for all 8 categories."""
    rows = []
    for month in range(1, 13):
        month_str = f"2024-{month:02d}"
        mult = MONTH_MULTIPLIER[month]

        for cat in CATEGORIES:
            lo, hi = AMOUNT_RANGES[cat]
            cat_mult = 1.0 if cat in ("Rent & Utilities", "Subscriptions") else mult

            if cat == "Subscriptions":
                for merch in MERCHANTS[cat]:
                    amt = round(random.uniform(lo, hi), 2)
                    day = random.randint(1, 28)
                    rows.append({
                        "date": f"{month_str}-{day:02d}",
                        "merchant": merch,
                        "amount": amt,
                        "category": cat,
                        "cluster_id": None,
                    })
            elif cat == "Rent & Utilities":
                for merch in random.sample(MERCHANTS[cat], random.randint(3, 4)):
                    amt = round(random.uniform(lo, hi) * cat_mult, 2)
                    day = random.randint(1, 28)
                    rows.append({
                        "date": f"{month_str}-{day:02d}",
                        "merchant": merch,
                        "amount": amt,
                        "category": cat,
                        "cluster_id": None,
                    })
            else:
                n_txn = random.randint(2, 5)
                for _ in range(n_txn):
                    merch = random.choice(MERCHANTS[cat])
                    amt = round(random.uniform(lo, hi) * cat_mult, 2)
                    day = random.randint(1, 28)
                    rows.append({
                        "date": f"{month_str}-{day:02d}",
                        "merchant": merch,
                        "amount": amt,
                        "category": cat,
                        "cluster_id": None,
                    })

    return pd.DataFrame(rows)


def generate_monthly_summaries(txn_df: pd.DataFrame) -> list[dict]:
    """Compute monthly summaries from transaction data."""
    txn_df = txn_df.copy()
    txn_df["month"] = txn_df["date"].str[:7]
    summaries = []
    for month, group in txn_df.groupby("month"):
        total = round(group["amount"].sum(), 2)
        cat_spend = group.groupby("category")["amount"].sum().round(2).to_dict()
        forecast_next = round(total * random.uniform(0.90, 1.10), 2)
        summaries.append({
            "month": month,
            "total_spend": total,
            "category_spend_json": json.dumps(cat_spend),
            "forecast_next_month": forecast_next,
            "portfolio_value": round(random.uniform(8000, 12000), 2),
        })
    return summaries


def generate_predictions() -> list[dict]:
    """Generate 3-month forecast predictions for all 8 categories."""
    rows = []
    base_month = datetime(2025, 1, 1)
    for offset in [1, 2, 3]:
        forecast_month = (base_month + timedelta(days=30 * offset)).strftime("%Y-%m")
        for cat in CATEGORIES:
            lo, hi = AMOUNT_RANGES[cat]
            predicted = round(random.uniform(lo * 2, hi * 1.5), 2)
            rows.append({
                "category": cat,
                "month_offset": offset,
                "forecast_month": forecast_month,
                "predicted_amount": predicted,
            })
    return rows


def generate_forecast_vs_actual() -> list[dict]:
    """Generate 3 months of forecast-vs-actual monitoring data with 8-14% error."""
    rows = []
    months = ["2024-10", "2024-11", "2024-12"]
    for month in months:
        for cat in CATEGORIES:
            lo, hi = AMOUNT_RANGES[cat]
            actual = round(random.uniform(lo * 2, hi * 1.5), 2)
            pct_err = round(random.uniform(8, 14), 1)
            direction = random.choice([-1, 1])
            predicted = round(actual * (1 + direction * pct_err / 100), 2)
            abs_err = round(abs(predicted - actual), 2)
            rows.append({
                "category": cat,
                "forecast_month": month,
                "predicted_value": predicted,
                "actual_value": actual,
                "absolute_error": abs_err,
                "pct_error": pct_err,
            })
    return rows


def generate_portfolio() -> list[dict]:
    """4 holdings with realistic prices."""
    return [
        {"ticker": "AAPL", "shares": 15, "avg_cost": 145.00, "current_price": 178.50, "pnl": round(15 * (178.50 - 145.00), 2)},
        {"ticker": "MSFT", "shares": 10, "avg_cost": 280.00, "current_price": 415.20, "pnl": round(10 * (415.20 - 280.00), 2)},
        {"ticker": "SPY", "shares": 20, "avg_cost": 420.00, "current_price": 512.30, "pnl": round(20 * (512.30 - 420.00), 2)},
        {"ticker": "BNS.TO", "shares": 50, "avg_cost": 65.00, "current_price": 72.80, "pnl": round(50 * (72.80 - 65.00), 2)},
    ]


def main():
    conn = get_connection()

    print("Clearing all tables...")
    for table in ["transactions", "predictions", "portfolio", "price_cache",
                   "monthly_summary", "forecast_vs_actual"]:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    print("All tables cleared.\n")

    txn_df = generate_seed_transactions()
    n = insert_transactions(conn, txn_df, SESSION_ID)
    print(f"[1/6] transactions: {n} rows seeded")

    summaries = generate_monthly_summaries(txn_df)
    n = upsert_monthly_summary(conn, summaries)
    print(f"[2/6] monthly_summary: {n} rows seeded")

    pred_rows = generate_predictions()
    pred_df = pd.DataFrame(pred_rows)
    n = insert_predictions(conn, pred_df, SESSION_ID)
    print(f"[3/6] predictions: {n} rows seeded")

    fva_rows = generate_forecast_vs_actual()
    n = insert_forecast_vs_actual(conn, fva_rows)
    print(f"[4/6] forecast_vs_actual: {n} rows seeded")

    portfolio = generate_portfolio()
    n = insert_portfolio(conn, portfolio, SESSION_ID)
    print(f"[5/6] portfolio: {n} rows seeded")

    now = datetime.now().isoformat()
    for holding in portfolio:
        upsert_price_cache(conn, holding["ticker"], holding["current_price"], now)
    print(f"[6/6] price_cache: {len(portfolio)} rows seeded")

    print("\n=== Final row counts ===")
    for table in ["transactions", "predictions", "portfolio", "price_cache",
                   "monthly_summary", "forecast_vs_actual"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count}")

    conn.close()
    print("\nSeed complete.")


if __name__ == "__main__":
    main()
