"""
PlainCents — ML-driven spending intelligence and reporting.
Phase 6: main.py orchestrates ingest → cluster → forecast → portfolio → DB writes.
"""
import json
import logging
from datetime import datetime

from db.database import (
    get_connection,
    get_predictions,
    get_transactions,
    insert_forecast_vs_actual,
    insert_predictions,
    insert_transactions,
    upsert_monthly_summary,
)
from pipeline.cluster import predict_categories
from pipeline.forecast import fit_and_forecast
from pipeline.ingest import load_and_clean
from pipeline.portfolio import build_portfolio

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    CSV_FILE = "synthetic_24mo.csv"
    BANK = "TD"
    SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    HOLDINGS = [
        {"ticker": "AAPL", "shares": 15, "avg_cost": 145.00},
        {"ticker": "MSFT", "shares": 10, "avg_cost": 280.00},
        {"ticker": "SPY", "shares": 20, "avg_cost": 420.00},
        {"ticker": "BNS.TO", "shares": 50, "avg_cost": 65.00},
    ]

    conn = get_connection()
    try:
        # Step 1 — Ingest
        df = load_and_clean(CSV_FILE, bank=BANK)
        logger.info("Loaded %d transactions", len(df))
        if df is None or df.empty:
            logger.error("Ingest returned empty DataFrame")
            raise ValueError("No transactions loaded")

        # Step 2 — Cluster
        df = predict_categories(df)
        logger.info(
            "K-Means model loaded — run python -m pipeline.cluster for held-out accuracy"
        )
        logger.info("Categories assigned. Unique: %d", df["category"].nunique())

        # Step 3 — Forecast
        forecast_df, overall_mape, cat_mape = fit_and_forecast(df)
        logger.info("Overall MAPE: %.1f%%", overall_mape)
        for cat, mape in sorted(cat_mape.items()):
            logger.info("  %s MAPE: %.1f%%", cat, mape)

        # Step 4 — Portfolio
        portfolio_df = build_portfolio(conn, HOLDINGS, session_id=SESSION_ID)
        logger.info("Portfolio rows: %d", len(portfolio_df))

        # Step 5 — DB writes
        insert_transactions(conn, df, SESSION_ID)

        insert_predictions(conn, forecast_df, SESSION_ID)

        df["month"] = df["date"].str[:7]
        months = sorted(df["month"].unique())
        forecast_next_month = round(
            forecast_df[forecast_df["month_offset"] == 1]["predicted_amount"].sum(), 2
        )
        if portfolio_df is not None and not portfolio_df.empty:
            portfolio_value = round(
                (portfolio_df["current_price"] * portfolio_df["shares"]).sum(), 2
            )
        else:
            portfolio_value = None

        monthly_rows = []
        for m in months:
            month_df = df[df["month"] == m]
            total_spend = round(month_df["amount"].sum(), 2)
            category_spend_json = json.dumps(
                month_df.groupby("category")["amount"].sum().round(2).to_dict()
            )
            monthly_rows.append({
                "month": m,
                "total_spend": total_spend,
                "category_spend_json": category_spend_json,
                "forecast_next_month": forecast_next_month,
                "portfolio_value": portfolio_value,
            })
        upsert_monthly_summary(conn, monthly_rows)

        txn_all = get_transactions(conn)
        txn_all["month"] = txn_all["date"].str[:7]
        actuals = (
            txn_all.groupby(["month", "category"])["amount"]
            .sum()
            .reset_index()
            .rename(columns={"amount": "actual_value", "month": "forecast_month"})
        )

        prior_preds = get_predictions(conn)
        prior_preds = prior_preds[prior_preds["session_id"] != SESSION_ID]

        if prior_preds.empty:
            logger.info(
                "Initial run: no historical predictions found for accuracy tracking — skipping"
            )
        else:
            matched = prior_preds.merge(
                actuals, on=["category", "forecast_month"], how="inner"
            )
            if matched.empty:
                logger.info(
                    "No matching actuals for prior predictions — skipping monitoring"
                )
            else:
                fva_rows = []
                for _, row in matched.iterrows():
                    abs_err = round(
                        abs(row["predicted_amount"] - row["actual_value"]), 2
                    )
                    pct_err = round(
                        abs_err / max(abs(row["actual_value"]), 1e-9) * 100, 2
                    )
                    fva_rows.append({
                        "category": row["category"],
                        "forecast_month": row["forecast_month"],
                        "predicted_value": row["predicted_amount"],
                        "actual_value": row["actual_value"],
                        "absolute_error": abs_err,
                        "pct_error": pct_err,
                    })
                insert_forecast_vs_actual(conn, fva_rows)
                logger.info("Wrote %d monitoring rows", len(fva_rows))

        # Step 6 — Exports stub
        print("PowerBI exports: Phase 8")
        logger.info("Pipeline complete. Session: %s", SESSION_ID)

        print("\n=== PlainCents Pipeline Complete ===")
        print(f"Session:        {SESSION_ID}")
        print(f"Transactions:   {len(df)}")
        print(f"Categories:     {df['category'].nunique()}")
        print(f"Forecast MAPE:  {overall_mape:.1f}%")
        print(f"Portfolio rows: {len(portfolio_df)}")
        print("Tables written: transactions, predictions,")
        print("  monthly_summary, forecast_vs_actual")

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        raise
    finally:
        conn.close()
