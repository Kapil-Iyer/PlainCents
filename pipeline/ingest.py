"""
Phase 1: CSV ingestion and cleaning.
Loads bank CSV, normalizes columns and dates, cleans merchant names. No DB write.
"""
import logging
from pathlib import Path

import pandas as pd

from config import BANK_DATE_FORMATS, DATA_RAW

logger = logging.getLogger(__name__)

# Bank-specific column name variants (detection/mapping only; date parsing uses BANK_DATE_FORMATS).
BANK_COLUMNS = {
    "TD": {
        "date": ["Date", "Transaction Date", "Posting Date", "DATE"],
        "merchant": ["Description", "Transaction Description", "Merchant", "DESCRIPTION"],
        "amount": ["Amount", "Debit", "Credit", "AMOUNT"],
    },
    "RBC": {
        "date": ["Transaction Date", "Date", "Posting Date", "DATE"],
        "merchant": ["Description", "Merchant", "Transaction", "DESCRIPTION"],
        "amount": ["Amount", "Debit", "Credit", "AMOUNT"],
    },
    "Scotiabank": {
        "date": ["Date", "Transaction Date", "Posting Date", "DATE"],
        "merchant": ["Description", "Merchant", "Transaction", "DESCRIPTION"],
        "amount": ["Amount", "Debit", "Credit", "AMOUNT"],
    },
}


def _detect_bank(df: pd.DataFrame) -> str | None:
    """Detect bank from which set of column names matches the DataFrame. Column logic only."""
    cols_upper = {c.upper().strip(): c for c in df.columns}
    for bank, mapping in BANK_COLUMNS.items():
        found = {}
        for std_name, candidates in mapping.items():
            for cand in candidates:
                if cand.upper() in cols_upper:
                    found[std_name] = cols_upper[cand.upper()]
                    break
        if set(found) == {"date", "merchant", "amount"}:
            return bank
    return None


def _find_column_mapping(df: pd.DataFrame, bank: str) -> dict[str, str]:
    """For given bank, return dict mapping standard name -> actual column name."""
    result = {}
    cols_upper = {c.upper().strip(): c for c in df.columns}
    for std_name, candidates in BANK_COLUMNS[bank].items():
        for cand in candidates:
            if cand.upper().strip() in cols_upper:
                result[std_name] = cols_upper[cand.upper().strip()]
                break
        if std_name not in result and std_name == "amount":
            for col in df.columns:
                if "amount" in col.lower() or "debit" in col.lower() or "credit" in col.lower():
                    result[std_name] = col
                    break
    return result


def load_and_clean(
    csv_path: Path | str,
    bank: str | None = None,
) -> pd.DataFrame:
    """
    Load a bank CSV, standardize columns and dates, clean merchant names, dedupe.
    Does not write to DB or save files.

    Parameters
    ----------
    csv_path : Path or str
        Path to the CSV file (e.g. from data/raw/).
    bank : str or None
        One of "TD", "RBC", "Scotiabank". If None, bank is detected from column names.

    Returns
    -------
    pandas.DataFrame
        Columns: date (YYYY-MM-DD str), merchant, amount. Cleaned and deduplicated.
    """
    path = Path(csv_path)
    if not path.is_absolute():
        path = DATA_RAW / path
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        logger.warning("CSV is empty: %s", path)
        return pd.DataFrame(columns=["date", "merchant", "amount"])

    # 1) Column detection / mapping (separate from date parsing)
    detected_bank = _detect_bank(df)
    if bank is not None:
        if bank not in BANK_COLUMNS:
            raise ValueError(f"Unknown bank: {bank}. Use one of {list(BANK_COLUMNS)}")
        use_bank = bank
    else:
        use_bank = detected_bank
        if use_bank is None:
            raise ValueError(
                "Could not detect bank from column names. Pass bank='TD', 'RBC', or 'Scotiabank' explicitly."
            )

    col_map = _find_column_mapping(df, use_bank)
    if set(col_map) != {"date", "merchant", "amount"}:
        raise ValueError(f"Missing columns for {use_bank}. Need date, merchant, amount. Got: {list(df.columns)}")
    df = df.rename(columns={v: k for k, v in col_map.items()})[["date", "merchant", "amount"]].copy()

    # 2) Parse dates (use bank format from config to resolve DD/MM vs MM/DD)
    date_fmt = BANK_DATE_FORMATS.get(use_bank)
    if date_fmt:
        df["date"] = pd.to_datetime(df["date"], format=date_fmt, errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 3) Drop NaT and log
    before = len(df)
    df = df.dropna(subset=["date"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with unparseable dates", dropped)

    # 4) Normalize to YYYY-MM-DD string
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # 5) Merchant: strip whitespace, uppercase, remove/normalize special chars
    df["merchant"] = df["merchant"].astype(str).str.strip()
    df["merchant"] = df["merchant"].str.upper()
    df["merchant"] = df["merchant"].str.replace(r"[^\w\s\-&]", "", regex=True)
    df["merchant"] = df["merchant"].str.replace(r"\s+", " ", regex=True).str.strip()

    # 6) Amount: ensure numeric (handle debits/credits if needed later; here assume single amount column)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    # 7) Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    return df
