# -*- coding: utf-8 -*-
"""
Phase 7: Matplotlib PDF reporting.
Generates a 5-chart report from SQLite tables with empty guards.
"""
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from config import CHART_COLORS, EXPORTS_DIR
from db.database import (
    get_connection,
    get_forecast_accuracy,
    get_monthly_summary,
    get_portfolio,
    get_predictions,
    get_transactions,
)

logger = logging.getLogger(__name__)


def get_report_path() -> Path:
    month = datetime.now().strftime("%Y-%m")
    return Path(EXPORTS_DIR) / f"PlainCents_Report_{month}.pdf"


def _no_data_chart(ax, message: str = "No Data Available"):
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=14,
        color="grey",
        transform=ax.transAxes,
    )
    ax.axis("off")


def chart_spending_trend(ax, df: pd.DataFrame):
    """
    Source: monthly_summary
    Needs: month, total_spend
    Guard: if fewer than 2 rows, show placeholder
    """
    if df is None or len(df) < 2:
        _no_data_chart(ax, "No Data Available")
        return

    chart_df = df.copy().sort_values("month")
    if "total_spend" not in chart_df.columns:
        _no_data_chart(ax, "No Data Available")
        return

    ax.plot(
        chart_df["month"],
        chart_df["total_spend"],
        color=CHART_COLORS["accent_line"],
        marker="o",
        label="Total Spend",
    )

    rolling = chart_df["total_spend"].rolling(3).mean()
    ax.plot(
        chart_df["month"],
        rolling,
        color=CHART_COLORS["accent_rolling"],
        linestyle="--",
        label="3-Month Avg",
    )

    ax.set_title("Monthly Spending Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount (CAD)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()


def chart_category_distribution(ax, df: pd.DataFrame):
    """
    Source: transactions
    Needs: category, amount, date
    Guard: if empty, show placeholder
    """
    if df is None or df.empty:
        _no_data_chart(ax, "No Transactions")
        return

    chart_df = df.copy()
    if not {"date", "category", "amount"}.issubset(chart_df.columns):
        _no_data_chart(ax, "No Transactions")
        return

    chart_df["month"] = chart_df["date"].astype(str).str[:7]
    latest = chart_df["month"].max()
    month_df = chart_df[chart_df["month"] == latest]

    if month_df.empty:
        _no_data_chart(ax, "No Transactions")
        return

    cat_totals = month_df.groupby("category")["amount"].sum().sort_values()
    colors = [CHART_COLORS.get(c, CHART_COLORS["Other"]) for c in cat_totals.index]

    ax.barh(cat_totals.index, cat_totals.values, color=colors)
    ax.set_title(f"Category Spend - {latest}")
    ax.set_xlabel("Amount (CAD)")


def chart_forecast_projection(ax, df: pd.DataFrame):
    """
    Source: predictions
    Needs: category, month_offset, predicted_amount
    Guard: if empty, show placeholder
    """
    if df is None or df.empty:
        _no_data_chart(ax, "No Forecast Available")
        return

    if not {"category", "month_offset", "predicted_amount"}.issubset(df.columns):
        _no_data_chart(ax, "No Forecast Available")
        return

    chart_df = df.copy()
    offsets = sorted(chart_df["month_offset"].unique())
    categories = sorted(chart_df["category"].dropna().unique())

    if len(categories) == 0 or len(offsets) == 0:
        _no_data_chart(ax, "No Forecast Available")
        return

    x = list(range(len(categories)))
    width = 0.25
    bar_colors = [
        CHART_COLORS["accent_bar"],
        CHART_COLORS["accent_line"],
        CHART_COLORS["accent_rolling"],
    ]

    for i, offset in enumerate(offsets):
        subset = chart_df[chart_df["month_offset"] == offset]
        vals = []
        for cat in categories:
            cat_vals = subset[subset["category"] == cat]["predicted_amount"].values
            vals.append(float(cat_vals[0]) if len(cat_vals) > 0 else 0.0)

        positions = [xi + i * width for xi in x]
        ax.bar(
            positions,
            vals,
            width=width,
            color=bar_colors[i % len(bar_colors)],
            label=f"+{int(offset)} month",
        )

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_title("3-Month Forecast Projection")
    ax.set_ylabel("Predicted Amount (CAD)")
    ax.legend()


def chart_portfolio_performance(ax, df: pd.DataFrame):
    """
    Source: monthly_summary
    Needs: month, portfolio_value
    Guard: all-null portfolio_value -> placeholder
    """
    if df is None or df.empty:
        _no_data_chart(ax, "No Portfolio Data Available")
        return

    chart_df = df.copy().sort_values("month")
    if "portfolio_value" not in chart_df.columns:
        _no_data_chart(ax, "No Portfolio Data Available")
        return

    pv = chart_df["portfolio_value"].dropna()
    if pv.empty:
        _no_data_chart(ax, "No Portfolio Data Available")
        return

    months = chart_df.loc[pv.index, "month"]
    ax.plot(months, pv, color=CHART_COLORS["accent_portfolio"], marker="o")
    ax.set_title("Portfolio Value Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Portfolio Value (CAD)")
    ax.tick_params(axis="x", rotation=45)


def chart_forecast_accuracy(ax, df: pd.DataFrame):
    """
    Source: forecast_vs_actual
    Needs: category, pct_error
    Guard: <2 rows -> placeholder
    """
    if df is None or len(df) < 2:
        _no_data_chart(ax, "Insufficient History")
        return

    if not {"category", "pct_error"}.issubset(df.columns):
        _no_data_chart(ax, "Insufficient History")
        return

    avg_err = df.groupby("category")["pct_error"].mean().sort_values()
    colors = [
        CHART_COLORS["accent_good"] if v <= 15 else CHART_COLORS["accent_bad"]
        for v in avg_err.values
    ]

    ax.bar(avg_err.index, avg_err.values, color=colors)
    ax.axhline(15, color="grey", linestyle="--", label="15% target")
    ax.set_title("Forecast Accuracy by Category")
    ax.text(
        0.5,
        1.02,
        "(Seed data - 8-14% synthetic error range)",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="grey",
    )
    ax.set_ylabel("Avg % Error")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()


def generate_report(conn) -> Path:
    """
    Pull all data from SQLite and generate a 5-chart PDF via PdfPages.
    Returns output path.
    """
    monthly_df = get_monthly_summary(conn)
    txn_df = get_transactions(conn)
    pred_df = get_predictions(conn)
    portfolio_df = get_portfolio(conn)
    fva_df = get_forecast_accuracy(conn)

    _ = portfolio_df  # retained for Phase 8 extensions/consistency

    report_path = get_report_path()
    Path(EXPORTS_DIR).mkdir(parents=True, exist_ok=True)

    with PdfPages(str(report_path)) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))
        chart_spending_trend(ax, monthly_df)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        chart_category_distribution(ax, txn_df)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        chart_forecast_projection(ax, pred_df)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        chart_portfolio_performance(ax, monthly_df)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        chart_forecast_accuracy(ax, fva_df)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    logger.info("Report saved: %s", report_path)
    return report_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conn = get_connection()
    try:
        path = generate_report(conn)
        print(f"Report generated: {path}")
    finally:
        conn.close()
