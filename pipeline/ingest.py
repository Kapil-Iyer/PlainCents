"""
Phase 1: CSV ingestion and cleaning.
Loads bank CSV, normalizes columns and dates, cleans merchant names. No DB write.
"""
import io
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


# ---------------------------------------------------------------------------
# Build Plan Phase 4: bytes-based entry point (TRD §3, §7.1). ADDITIVE ONLY.
#
# load_and_clean(csv_path, bank) above is untouched — same code, same
# behavior, same signature — so V1's main.py keeps working unmodified.
# load_and_clean_from_bytes() below is a fully independent implementation
# (deliberately not sharing helper code with load_and_clean's body) so there
# is zero risk of an extraction accidentally changing V1's verified path.
#
# It also supports one additive fallback shape, TD-only: a headerless,
# 5-positional-column layout (date, description, withdrawals, deposits,
# balance). This is third-party evidence about some real TD EasyWeb
# chequing/savings exports (see tests/fixtures/td_csv/README.md) — NOT a
# PlainCents-verified schema, and NOT the frozen Phase 0 fixtures' shape
# (those remain header-based Date/Description/Amount, %m/%d/%Y, and that
# header-based path remains the primary, tested path below). The fallback
# only ever activates when header-based column detection fails AND the
# caller asked for bank="TD" — RBC/Scotiabank get no such fallback, and
# Phase 4 does not add any new bank.
# ---------------------------------------------------------------------------

# TD headerless fallback: example real-world date rendering reported for
# this shape is "Aug 27, 2024" — try the explicit format first (deterministic
# for that exact style), then fall back to pandas' general inference for any
# stray row a strict format would otherwise still get right.
_TD_HEADERLESS_DATE_FORMATS = ["%b %d, %Y", "%B %d, %Y"]
_TD_HEADERLESS_COLUMNS = ["date", "description", "withdrawals", "deposits", "balance"]


def _clean_header_based(df: pd.DataFrame, col_map: dict[str, str], use_bank: str) -> tuple[pd.DataFrame, dict]:
    """Header-based cleaning steps, mirroring load_and_clean()'s steps 2-7
    exactly, but tracking counts instead of only logging them (Phase 4 needs
    rows_unparseable for ImportPreview, TRD §6) and returning rather than
    raising on the already-checked column mapping."""
    rows_total = len(df)
    df = df.rename(columns={v: k for k, v in col_map.items()})[["date", "merchant", "amount"]].copy()

    date_fmt = BANK_DATE_FORMATS.get(use_bank)
    if date_fmt:
        df["date"] = pd.to_datetime(df["date"], format=date_fmt, errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["date"])
    rows_unparseable = before - len(df)
    if rows_unparseable:
        logger.warning("Dropped %d rows with unparseable dates", rows_unparseable)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    df["merchant"] = df["merchant"].astype(str).str.strip().str.upper()
    df["merchant"] = df["merchant"].str.replace(r"[^\w\s\-&]", "", regex=True)
    df["merchant"] = df["merchant"].str.replace(r"\s+", " ", regex=True).str.strip()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    before_amount = len(df)
    df = df.dropna(subset=["amount"])
    rows_unparseable += before_amount - len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    return df, {"bank_detected": use_bank, "rows_total": rows_total, "rows_unparseable": rows_unparseable}


def _try_td_headerless(file_bytes: bytes) -> tuple[pd.DataFrame, dict] | None:
    """
    TD-only, additive fallback for a headerless, 5-positional-column export
    shape — see the module-level note above. Returns None (never raises) if
    the bytes don't actually look like this shape, so the caller falls
    through to the standard "unrecognized format" 400 rather than a
    confusing partial parse of a genuinely unrelated file.
    """
    try:
        raw = pd.read_csv(io.BytesIO(file_bytes), header=None)
    except Exception:
        return None

    # Deliberately 4-5 columns only, never 3: a 3-column layout is exactly
    # what every header-based bank format in this codebase already uses
    # (date/merchant/amount) — accepting 3 here would risk silently
    # reinterpreting a genuinely unrecognized 3-column header-based file
    # (e.g. tests/fixtures/td_csv/unrecognized_format.csv) as this shape
    # instead of correctly raising the "unrecognized format" 400.
    if raw.empty or raw.shape[1] < 4 or raw.shape[1] > 5:
        return None

    rows_total = len(raw)
    raw = raw.copy()
    raw.columns = _TD_HEADERLESS_COLUMNS[: raw.shape[1]]

    # Try the explicit reported format(s) first (deterministic, no warning);
    # only fall back to pandas' general inference for any stray row those
    # don't cover.
    parsed_date = pd.Series(pd.NaT, index=raw.index)
    for fmt in _TD_HEADERLESS_DATE_FORMATS:
        mask = parsed_date.isna()
        if not mask.any():
            break
        parsed_date.loc[mask] = pd.to_datetime(raw.loc[mask, "date"], format=fmt, errors="coerce")
    remaining = parsed_date.isna()
    if remaining.any():
        parsed_date.loc[remaining] = pd.to_datetime(raw.loc[remaining, "date"], errors="coerce")
    raw["date"] = parsed_date

    # Sanity gate: if most rows don't even look like dates, this genuinely
    # isn't the headerless TD shape (most likely a header row landed in
    # row 0 and got treated as data, or the file is unrelated) — bail out to
    # the standard unrecognized-format error rather than a near-empty import.
    if rows_total == 0 or raw["date"].notna().mean() < 0.5:
        return None

    before = len(raw)
    raw = raw.dropna(subset=["date"])
    rows_unparseable = before - len(raw)
    raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")

    withdrawals = (
        pd.to_numeric(raw["withdrawals"], errors="coerce")
        if "withdrawals" in raw.columns
        else pd.Series(float("nan"), index=raw.index)
    )
    deposits = (
        pd.to_numeric(raw["deposits"], errors="coerce")
        if "deposits" in raw.columns
        else pd.Series(float("nan"), index=raw.index)
    )

    # Spend-only MVP scope: every category/forecast concept in the frozen
    # PRD/TRD is about spending, and the categorization model (config.py's
    # 8 categories) was trained only on positive spend amounts. A withdrawal
    # is a spend transaction (positive amount, the same convention every
    # other bank/format in this codebase already uses). A deposit-only row
    # (income/transfer-in, no withdrawal) is not a spend transaction at all
    # and is out of scope here — it is dropped, and folded into
    # rows_unparseable for visibility (so it's counted somewhere in the
    # ImportPreview, not silently vanishing) rather than fabricated as a
    # zero-amount or negative-amount "spend."
    raw["amount"] = withdrawals
    deposit_only = raw["amount"].isna() & deposits.notna()
    rows_unparseable += int(deposit_only.sum())
    raw = raw[~deposit_only]

    before_amount = len(raw)
    raw = raw.dropna(subset=["amount"])
    rows_unparseable += before_amount - len(raw)

    raw["merchant"] = raw["description"].astype(str).str.strip().str.upper()
    raw["merchant"] = raw["merchant"].str.replace(r"[^\w\s\-&]", "", regex=True)
    raw["merchant"] = raw["merchant"].str.replace(r"\s+", " ", regex=True).str.strip()

    result = raw[["date", "merchant", "amount"]].drop_duplicates().reset_index(drop=True)
    return result, {"bank_detected": "TD", "rows_total": rows_total, "rows_unparseable": rows_unparseable}


def load_and_clean_from_bytes(file_bytes: bytes, bank: str | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Bytes-based counterpart to load_and_clean(), for an uploaded file rather
    than a filesystem path (Build Plan Phase 4, TRD §3/§7.1). Does not modify
    or call load_and_clean() — the two are independent implementations of
    the same cleaning steps so V1's verified path can never be affected by
    this addition.

    Parameters
    ----------
    file_bytes : bytes
        Raw uploaded file content.
    bank : str or None
        One of "TD", "RBC", "Scotiabank". If None, bank is detected from
        column names (header-based path only — the headerless TD fallback
        requires bank="TD" to be explicit, since it is not a generically
        detectable shape).

    Returns
    -------
    (df, meta)
        df : pandas.DataFrame
            Columns: date (YYYY-MM-DD str), merchant, amount. Cleaned and
            intra-file-deduplicated, same contract as load_and_clean()'s
            return value.
        meta : dict
            {"bank_detected": str, "rows_total": int, "rows_unparseable": int}
            rows_unparseable counts rows dropped during cleaning for any
            reason load_and_clean() would also drop (unparseable dates,
            non-numeric amounts), plus — for the TD headerless fallback only
            — deposit-only rows that aren't a spend transaction at all (see
            _try_td_headerless).

    Raises
    ------
    ValueError
        Whole-file failure: bank could not be detected/mapped from column
        names and (if bank="TD") the headerless fallback didn't match
        either. Mapped to HTTP 400 by the caller (TRD §10) — not a 200 with
        every row invalid.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    if df.empty:
        return pd.DataFrame(columns=["date", "merchant", "amount"]), {
            "bank_detected": bank or "",
            "rows_total": 0,
            "rows_unparseable": 0,
        }

    detected_bank = _detect_bank(df)
    if bank is not None:
        if bank not in BANK_COLUMNS:
            raise ValueError(f"Unknown bank: {bank}. Use one of {list(BANK_COLUMNS)}")
        use_bank = bank
    else:
        use_bank = detected_bank

    if use_bank is not None:
        col_map = _find_column_mapping(df, use_bank)
        if set(col_map) == {"date", "merchant", "amount"}:
            return _clean_header_based(df, col_map, use_bank)

    # Header-based detection/mapping failed. TD-only additive fallback (see
    # module-level note above) — never attempted for RBC/Scotiabank, and
    # never attempted unless the caller explicitly asked for bank="TD".
    if bank == "TD":
        headerless = _try_td_headerless(file_bytes)
        if headerless is not None:
            return headerless

    raise ValueError(
        f"Could not detect bank/columns from file. Got columns: {list(df.columns)}"
    )
