"""
Phase 8 Part A: Export SQLite data for PowerBI.
Generates 4 CSV files under data/exports.
"""
import logging
from pathlib import Path

import pandas as pd

from config import EXPORTS_DIR
from db.database import (
    get_connection,
    get_forecast_accuracy,
    get_portfolio,
    get_predictions,
    get_transactions,
)

logger = logging.getLogger(__name__)


def _latest_runtime_session(df: pd.DataFrame) -> str | None:
    if df is None or df.empty or "session_id" not in df.columns:
        return None
    session_series = df["session_id"].astype(str)
    runtime_ids = session_series[session_series.str.match(r"^\d{8}_\d{6}$", na=False)]
    if not runtime_ids.empty:
        return runtime_ids.max()
    return session_series.max()


def _export_selected_columns(df: pd.DataFrame, columns: list[str], out_path: Path) -> int:
    if df is None or df.empty:
        export_df = pd.DataFrame(columns=columns)
    else:
        export_df = df.reindex(columns=columns)
        if "date" in export_df.columns:
            export_df["date"] = export_df["date"].astype(str)
            export_df = export_df.sort_values("date")
        elif "forecast_month" in export_df.columns:
            export_df = export_df.sort_values("forecast_month")
        elif "ticker" in export_df.columns:
            export_df = export_df.sort_values("ticker")
        elif "category" in export_df.columns:
            export_df = export_df.sort_values("category")
    export_df.to_csv(out_path, index=False)
    return len(export_df)


def export_powerbi_csvs(conn) -> dict[str, Path]:
    """
    Export 4 PowerBI CSV files to EXPORTS_DIR.
    Returns {filename: path}.
    """
    Path(EXPORTS_DIR).mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    # CSV 1 — transactions_clean.csv
    tx_file = "transactions_clean.csv"
    tx_path = Path(EXPORTS_DIR) / tx_file
    tx_df = get_transactions(conn)
    if not tx_df.empty and "session_id" in tx_df.columns:
        latest = _latest_runtime_session(tx_df)
        tx_df = tx_df[tx_df["session_id"] == latest]
    tx_rows = _export_selected_columns(
        tx_df,
        ["date", "merchant", "amount", "category"],
        tx_path,
    )
    logger.info("%s: %d rows exported", tx_file, tx_rows)
    outputs[tx_file] = tx_path

    # CSV 2 — forecasts.csv
    fc_file = "forecasts.csv"
    fc_path = Path(EXPORTS_DIR) / fc_file
    fc_df = get_predictions(conn)
    if not fc_df.empty and "session_id" in fc_df.columns:
        latest = _latest_runtime_session(fc_df)
        fc_df = fc_df[fc_df["session_id"] == latest]
    fc_rows = _export_selected_columns(
        fc_df,
        ["category", "month_offset", "predicted_amount"],
        fc_path,
    )
    logger.info("%s: %d rows exported", fc_file, fc_rows)
    outputs[fc_file] = fc_path

    # CSV 3 — portfolio.csv
    pf_file = "portfolio.csv"
    pf_path = Path(EXPORTS_DIR) / pf_file
    pf_df = get_portfolio(conn)
    if not pf_df.empty and "session_id" in pf_df.columns:
        latest = _latest_runtime_session(pf_df)
        pf_df = pf_df[pf_df["session_id"] == latest]
    pf_rows = _export_selected_columns(
        pf_df,
        ["ticker", "shares", "avg_cost", "current_price", "pnl"],
        pf_path,
    )
    logger.info("%s: %d rows exported", pf_file, pf_rows)
    outputs[pf_file] = pf_path

    # CSV 4 — forecast_accuracy.csv
    fa_file = "forecast_accuracy.csv"
    fa_path = Path(EXPORTS_DIR) / fa_file
    fa_rows = _export_selected_columns(
        get_forecast_accuracy(conn),
        ["category", "forecast_month", "predicted_value", "actual_value", "pct_error"],
        fa_path,
    )
    logger.info("%s: %d rows exported", fa_file, fa_rows)
    outputs[fa_file] = fa_path

    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conn = get_connection()
    try:
        paths = export_powerbi_csvs(conn)
        for name, path in paths.items():
            print(f"{name}: {path}")
    finally:
        conn.close()
