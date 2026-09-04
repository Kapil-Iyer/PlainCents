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
# NOTE: this dict (and _detect_bank/_find_column_mapping below) back V1's
# load_and_clean() ONLY. The V2 bytes-import path (load_and_clean_from_bytes,
# Phase 4+) has its own independent, stricter per-bank detection below it and
# never reads the "RBC"/"Scotiabank" entries here -- those were an early,
# never-fixture-tested placeholder (generic single date/merchant/amount
# columns) that does not reflect either bank's real export shape (Phase
# 12A/12A.5). Left in place, untouched, so V1's CLI behavior and signature
# cannot regress.
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
    """Detect bank from which set of column names matches the DataFrame. Column logic only.
    V1 (load_and_clean) only -- see the BANK_COLUMNS note above."""
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
    """For given bank, return dict mapping standard name -> actual column name.
    V1 (load_and_clean) only -- see the BANK_COLUMNS note above."""
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
# Build Plan Phase 4 / Phase 12A-12B: bytes-based entry point (TRD §3, §7.1).
# ADDITIVE ONLY.
#
# load_and_clean(csv_path, bank) above is untouched -- same code, same
# behavior, same signature -- so V1's main.py keeps working unmodified.
# load_and_clean_from_bytes() below is a fully independent implementation
# (deliberately not sharing helper code with load_and_clean's body) so there
# is zero risk of an extraction accidentally changing V1's verified path.
#
# Six-bank detection (Phase 12A.5 frozen evidence table):
#   RBC         -- ACTUAL EXPORT evidence (a real RBC CSV was supplied)
#   Scotiabank  -- ACTUAL EXPORT evidence (a real Scotiabank CSV was supplied)
#   TD          -- PROJECT-VERIFIED (header-based, existing since Phase 0)
#                  + an additive, TD-only headerless fallback (third-party
#                  evidence, disclosed unverified -- see
#                  tests/fixtures/td_csv/README.md)
#   CIBC        -- RESEARCH-BACKED, fail-closed (consistent but unofficial
#                  web evidence; strict header + strict date format, no
#                  headerless fallback)
#   BMO         -- BLOCKED (Phase 12A.5): independent sources describe three
#                  mutually incompatible column layouts and disagree on
#                  whether dates are even a delimited string. Not implemented.
#   National Bank -- BLOCKED (Phase 12A.5): no reliable column- or
#                  date-format evidence found. Not implemented.
#
# Detection order (strongest evidence first), per bank param:
#   bank=None ("Auto")   -> RBC fingerprint -> Scotiabank fingerprint ->
#                           CIBC fingerprint -> [blocked-format guard] ->
#                           TD header-based -> TD headerless -> unsupported
#   bank="RBC"/"Scotiabank"/"CIBC"/"TD" (explicit) -> validate ONLY that
#                           bank's own path(s); never silently reinterpret
#                           as a different bank.
#   bank="BMO"/"National Bank" (explicit) -> explicit "not yet supported"
#                           error (Phase 12A.5 BLOCKED), not a guess.
# ---------------------------------------------------------------------------

RBC_COLUMNS = [
    "Account Type",
    "Account Number",
    "Transaction Date",
    "Cheque Number",
    "Description 1",
    "Description 2",
    "CAD$",
    "USD$",
]

SCOTIA_COLUMNS = [
    "Filter",
    "Date",
    "Description",
    "Sub-description",
    "Type of Transaction",
    "Amount",
    "Balance",
]

CIBC_COLUMNS = ["Transaction Date", "Description", "Withdrawals", "Deposits", "Balance"]

_CANONICAL_BANKS = {
    "RBC": "RBC",
    "SCOTIABANK": "Scotiabank",
    "SCOTIA": "Scotiabank",
    "TD": "TD",
    "CIBC": "CIBC",
    "BMO": "BMO",
    "NATIONAL BANK": "National Bank",
    "NATIONAL": "National Bank",
}
# Phase 12A.5 evidence gate: these two remain evidence-BLOCKED, not
# implemented. They stay selectable (a user-safe explicit error, not a
# KeyError) rather than absent, so the frontend's six-bank selector and this
# module agree on the full named set.
_BLOCKED_BANKS = {"BMO", "National Bank"}
_ALL_KNOWN_BANKS = sorted(set(_CANONICAL_BANKS.values()))

# TD headerless fallback: example real-world date rendering reported for
# this shape is "Aug 27, 2024" -- try the explicit format first (deterministic
# for that exact style), then fall back to pandas' general inference for any
# stray row a strict format would otherwise still get right.
_TD_HEADERLESS_DATE_FORMATS = ["%b %d, %Y", "%B %d, %Y"]
_TD_HEADERLESS_COLUMNS = ["date", "description", "withdrawals", "deposits", "balance"]


def _normalize_bank_name(bank: str | None) -> str | None:
    """None/blank/"auto" -> None (auto-detect). A known bank name (any case,
    a couple of accepted spellings) -> its canonical form. Anything else ->
    ValueError, same as an unknown bank always has been."""
    if bank is None:
        return None
    stripped = bank.strip()
    if not stripped or stripped.upper() in ("AUTO", "AUTO-DETECT", "AUTODETECT"):
        return None
    key = stripped.upper()
    if key in _CANONICAL_BANKS:
        return _CANONICAL_BANKS[key]
    raise ValueError(f"Unknown bank: {bank}. Use one of {_ALL_KNOWN_BANKS} or omit for auto-detect.")


def _normalize_header_name(name: str) -> str:
    """Strip a UTF-8 BOM (pandas can leave one glued to the first header
    cell), trim whitespace, uppercase -- for case/whitespace-insensitive
    header comparison (Phase 12A.5 §14)."""
    return str(name).replace("﻿", "").strip().upper()


def _fingerprint_matches(df: pd.DataFrame, expected_columns: list[str]) -> bool:
    """Exact, order-independent, normalized column-name match. Deliberately
    stricter than TD's original candidate-list detector -- reordered/extra/
    missing columns never match (fail closed rather than fuzzy-match), per
    Phase 12A.5 §14."""
    actual = {_normalize_header_name(c) for c in df.columns}
    expected = {_normalize_header_name(c) for c in expected_columns}
    return actual == expected


def _column_lookup(df: pd.DataFrame, expected_columns: list[str]) -> dict[str, str]:
    """Map each expected canonical column name -> the DataFrame's actual
    column name, case/whitespace-insensitively. Only ever called after
    _fingerprint_matches has already confirmed every expected column has a
    match, so every lookup here is guaranteed to succeed."""
    norm_to_actual = {_normalize_header_name(c): c for c in df.columns}
    return {expected: norm_to_actual[_normalize_header_name(expected)] for expected in expected_columns}


def _looks_like_blocked_or_unknown_balance_format(df: pd.DataFrame) -> bool:
    """
    Phase 12A found the old loose TD header detector can accept
    Date,Description,Amount,Balance (one of BMO's reported shapes) as TD,
    since it only checks that date/merchant/amount candidates resolve and
    silently ignores extra columns. BMO and National Bank remain
    evidence-BLOCKED (Phase 12A.5) rather than implemented, but in AUTO
    mode a file carrying a Balance-like column that doesn't match one of
    the three implemented strict fingerprints (RBC/Scotiabank/CIBC) must
    never be silently absorbed into TD's own loose 3-column match -- a
    genuine TD header-based export never carries a Balance column (see
    tests/fixtures/td_csv/clean_valid.csv). Fail closed instead of guessing.
    """
    normalized = {_normalize_header_name(c) for c in df.columns}
    return "BALANCE" in normalized


def _clean_merchant_text(series: pd.Series) -> pd.Series:
    """Shared merchant-cleaning steps (trim, uppercase, strip punctuation
    except hyphen/ampersand, collapse whitespace) -- identical behavior to
    what _clean_header_based/_try_td_headerless always did inline, just
    factored out once the six-bank adapters need the same steps repeatedly.
    No advanced/NLP merchant extraction (Phase 12A.5 §12)."""
    cleaned = series.fillna("").astype(str).str.strip().str.upper()
    cleaned = cleaned.str.replace(r"[^\w\s\-&]", "", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned


def _join_description_fields(primary: pd.Series, secondary: pd.Series) -> pd.Series:
    """Deterministic raw_description join for banks with a primary + optional
    secondary description field (RBC's Description 1/2, Scotiabank's
    Description/Sub-description -- Phase 12A.5 §8/§4/§5). Trim each field
    first; append the secondary only when it is non-blank after trimming, so
    the same source row always normalizes identically regardless of which
    field a given bank happens to populate, and RBC/Scotiabank's two-field
    ordering can never change a transaction's identity for dedup purposes."""
    primary_clean = primary.fillna("").astype(str).str.strip()
    secondary_clean = secondary.fillna("").astype(str).str.strip()
    has_secondary = secondary_clean != ""
    return primary_clean.where(~has_secondary, primary_clean + " " + secondary_clean)


def _clean_header_based(df: pd.DataFrame, col_map: dict[str, str], use_bank: str) -> tuple[pd.DataFrame, dict]:
    """TD's project-verified header-based path (Date/Description/Amount,
    %m/%d/%Y). Mirrors load_and_clean()'s steps 2-7, but tracks counts
    instead of only logging them, returns raw_description alongside merchant
    (Phase 12A.5 §12 -- previously discarded), and -- Phase 12B dedup fix --
    no longer calls drop_duplicates(): two legitimate transactions sharing
    (date, amount, merchant) must both survive here so IngestionService's
    occurrence_index can tell them apart, not be silently collapsed to one
    row before it gets the chance (Phase 12A finding)."""
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

    df["raw_description"] = df["merchant"].fillna("").astype(str).str.strip()
    df["merchant"] = _clean_merchant_text(df["merchant"])

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    before_amount = len(df)
    df = df.dropna(subset=["amount"])
    rows_unparseable += before_amount - len(df)

    result = df[["date", "raw_description", "merchant", "amount"]].reset_index(drop=True)
    return result, {
        "bank_detected": use_bank,
        "rows_total": rows_total,
        "rows_unparseable": rows_unparseable,
        "rows_skipped_credit": 0,
        "rows_skipped_currency": 0,
    }


def _try_td_headerless(file_bytes: bytes) -> tuple[pd.DataFrame, dict] | None:
    """
    TD-only, additive fallback for a headerless, 5-positional-column export
    shape -- see the module-level note above. Returns None (never raises) if
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
    # (date/merchant/amount) -- accepting 3 here would risk silently
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
    # row 0 and got treated as data, or the file is unrelated) -- bail out to
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
    # and is out of scope here -- it is excluded and counted separately
    # (rows_skipped_credit, Phase 12A.5 §17/§24) rather than folded into
    # rows_unparseable (it isn't malformed -- it was correctly recognized
    # and intentionally excluded) or fabricated as a zero/negative "spend."
    raw["amount"] = withdrawals
    credit_only = raw["amount"].isna() & deposits.notna()
    rows_skipped_credit = int(credit_only.sum())
    raw = raw[~credit_only]

    before_amount = len(raw)
    raw = raw.dropna(subset=["amount"])
    rows_unparseable += before_amount - len(raw)

    raw["raw_description"] = raw["description"].fillna("").astype(str).str.strip()
    raw["merchant"] = _clean_merchant_text(raw["description"])

    result = raw[["date", "raw_description", "merchant", "amount"]].reset_index(drop=True)
    return result, {
        "bank_detected": "TD",
        "rows_total": rows_total,
        "rows_unparseable": rows_unparseable,
        "rows_skipped_credit": rows_skipped_credit,
        "rows_skipped_currency": 0,
    }


def _clean_rbc(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    RBC personal chequing export (Phase 12A.5 §4 -- ACTUAL EXPORT evidence:
    an actual RBC CSV was supplied to this project). Frozen: %m/%d/%Y dates;
    negative CAD$ = spend (canonical amount = abs(CAD$)); positive CAD$ =
    credit/inflow, excluded; a USD$-only row is excluded as an unsupported
    currency (no conversion, never silently treated as CAD -- PlainCents has
    no currency field at all); both CAD$ and USD$ populated, or CAD$ == 0,
    is ambiguous and rejected rather than guessed. Account Number, Account
    Type, and Cheque Number are read only to confirm the header shape and
    are never included in the returned columns -- they do not reach the
    caller, the DB, logs, or the API.
    """
    rows_total = len(df)
    cols = _column_lookup(df, RBC_COLUMNS)
    work = df.rename(columns={v: k for k, v in cols.items()})

    work["date"] = pd.to_datetime(work["Transaction Date"], format="%m/%d/%Y", errors="coerce")
    before = len(work)
    work = work.dropna(subset=["date"])
    rows_unparseable = before - len(work)
    work["date"] = work["date"].dt.strftime("%Y-%m-%d")

    cad = pd.to_numeric(work["CAD$"], errors="coerce")
    usd = pd.to_numeric(work["USD$"], errors="coerce")
    cad_present = cad.notna()
    usd_present = usd.notna()

    spend_mask = cad_present & ~usd_present & (cad < 0)
    credit_mask = cad_present & ~usd_present & (cad > 0)
    currency_mask = usd_present & ~cad_present
    # Both populated, neither populated, or CAD$ == 0: can't classify safely
    # -- fail closed rather than guess (Phase 12A.5 §4).
    reject_mask = ~(spend_mask | credit_mask | currency_mask)

    rows_unparseable += int(reject_mask.sum())
    rows_skipped_currency = int(currency_mask.sum())
    rows_skipped_credit = int(credit_mask.sum())

    work = work[spend_mask].copy()
    work["amount"] = cad[spend_mask].abs()

    work["raw_description"] = _join_description_fields(work["Description 1"], work["Description 2"])
    work["merchant"] = _clean_merchant_text(work["raw_description"])

    result = work[["date", "raw_description", "merchant", "amount"]].reset_index(drop=True)
    return result, {
        "bank_detected": "RBC",
        "rows_total": rows_total,
        "rows_unparseable": rows_unparseable,
        "rows_skipped_credit": rows_skipped_credit,
        "rows_skipped_currency": rows_skipped_currency,
    }


def _clean_scotiabank(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Scotiabank Preferred Package export (Phase 12A.5 §5 -- ACTUAL EXPORT
    evidence: an actual Scotiabank CSV was supplied to this project).
    Frozen: %Y-%m-%d dates; Amount's sign must AGREE with Type of Transaction
    (Debit -> negative -> spend, Credit -> positive -> excluded credit) or
    the row is rejected -- transaction type is never inferred from sign
    alone, and an unknown/missing Type of Transaction value is also
    rejected. Filter and Balance are read only to confirm the header shape
    and are never persisted.
    """
    rows_total = len(df)
    cols = _column_lookup(df, SCOTIA_COLUMNS)
    work = df.rename(columns={v: k for k, v in cols.items()})

    work["date"] = pd.to_datetime(work["Date"], format="%Y-%m-%d", errors="coerce")
    before = len(work)
    work = work.dropna(subset=["date"])
    rows_unparseable = before - len(work)
    work["date"] = work["date"].dt.strftime("%Y-%m-%d")

    amount = pd.to_numeric(work["Amount"], errors="coerce")
    txn_type = work["Type of Transaction"].fillna("").astype(str).str.strip().str.upper()

    spend_mask = (txn_type == "DEBIT") & amount.notna() & (amount < 0)
    credit_mask = (txn_type == "CREDIT") & amount.notna() & (amount > 0)
    # Debit+positive, Credit+negative, unknown/missing type, or NaN amount:
    # contradictory or unclassifiable -- reject rather than guess.
    reject_mask = ~(spend_mask | credit_mask)

    rows_unparseable += int(reject_mask.sum())
    rows_skipped_credit = int(credit_mask.sum())

    work = work[spend_mask].copy()
    work["amount"] = amount[spend_mask].abs()

    work["raw_description"] = _join_description_fields(work["Description"], work["Sub-description"])
    work["merchant"] = _clean_merchant_text(work["raw_description"])

    result = work[["date", "raw_description", "merchant", "amount"]].reset_index(drop=True)
    return result, {
        "bank_detected": "Scotiabank",
        "rows_total": rows_total,
        "rows_unparseable": rows_unparseable,
        "rows_skipped_credit": rows_skipped_credit,
        "rows_skipped_currency": 0,
    }


def _clean_cibc(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    CIBC personal chequing export (Phase 12A.5 §7 -- RESEARCH-BACKED,
    fail-closed: this exact header and %Y-%m-%d date format were the single
    most consistent claim across independent web searches, but all sources
    are unofficial bank-statement-conversion sites, not CIBC documentation
    or an actual supplied export -- see the Phase 12A.5 freeze table. No
    headerless fallback: a file without this exact header never matches
    CIBC, and never falls back to generic/ambiguous date inference.
    """
    rows_total = len(df)
    cols = _column_lookup(df, CIBC_COLUMNS)
    work = df.rename(columns={v: k for k, v in cols.items()})

    work["date"] = pd.to_datetime(work["Transaction Date"], format="%Y-%m-%d", errors="coerce")
    before = len(work)
    work = work.dropna(subset=["date"])
    rows_unparseable = before - len(work)
    work["date"] = work["date"].dt.strftime("%Y-%m-%d")

    withdrawals = pd.to_numeric(work["Withdrawals"], errors="coerce")
    deposits = pd.to_numeric(work["Deposits"], errors="coerce")
    w_present = withdrawals.notna()
    d_present = deposits.notna()

    spend_mask = w_present & ~d_present
    credit_mask = d_present & ~w_present
    # Both populated or neither populated: ambiguous -- reject rather than
    # guess which column wins.
    reject_mask = ~(spend_mask | credit_mask)

    rows_unparseable += int(reject_mask.sum())
    rows_skipped_credit = int(credit_mask.sum())

    work = work[spend_mask].copy()
    work["amount"] = withdrawals[spend_mask].abs()

    work["raw_description"] = work["Description"].fillna("").astype(str).str.strip()
    work["merchant"] = _clean_merchant_text(work["Description"])

    result = work[["date", "raw_description", "merchant", "amount"]].reset_index(drop=True)
    return result, {
        "bank_detected": "CIBC",
        "rows_total": rows_total,
        "rows_unparseable": rows_unparseable,
        "rows_skipped_credit": rows_skipped_credit,
        "rows_skipped_currency": 0,
    }


_EMPTY_META = {
    "rows_total": 0,
    "rows_unparseable": 0,
    "rows_skipped_credit": 0,
    "rows_skipped_currency": 0,
}


def load_and_clean_from_bytes(file_bytes: bytes, bank: str | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Bytes-based counterpart to load_and_clean(), for an uploaded file rather
    than a filesystem path (Build Plan Phase 4, TRD §3/§7.1; six-bank
    detection added Phase 12A-12B). Does not modify or call load_and_clean()
    -- the two are independent implementations of the same cleaning steps so
    V1's verified path can never be affected by this addition.

    Parameters
    ----------
    file_bytes : bytes
        Raw uploaded file content.
    bank : str or None
        One of "RBC", "Scotiabank", "TD", "CIBC" (implemented), or
        "BMO"/"National Bank" (explicitly BLOCKED -- Phase 12A.5 evidence
        gate, raises a clear "not yet supported" ValueError rather than
        guessing). None (or "Auto"/"" ) means auto-detect: try the strongest
        evidence first (RBC -> Scotiabank -> CIBC fingerprints), then guard
        against misclassifying a recognized-but-unsupported shape as TD,
        then TD's header-based path, then TD's headerless fallback.
        Explicit bank selection only ever validates that one bank's own
        path(s) -- it never silently reinterprets the file as a different
        bank (Phase 12A.5 §15).

    Returns
    -------
    (df, meta)
        df : pandas.DataFrame
            Columns: date (YYYY-MM-DD str), raw_description, merchant,
            amount. Cleaned; NOT intra-file-deduplicated (Phase 12B fix --
            two legitimate identical transactions in one file must both
            survive so IngestionService's occurrence_index can distinguish
            them; see the module note above and IngestionService).
        meta : dict
            {"bank_detected": str, "rows_total": int, "rows_unparseable": int,
             "rows_skipped_credit": int, "rows_skipped_currency": int}
            rows_unparseable counts rows dropped for being genuinely
            malformed/ambiguous (unparseable dates, non-numeric amounts,
            contradictory or unclassifiable debit/credit signals).
            rows_skipped_credit / rows_skipped_currency count rows that
            were correctly recognized but intentionally excluded (credits/
            deposits; RBC USD$-only rows) -- not malformed, just out of
            spend-tracking scope (Phase 12A.5 §17).

    Raises
    ------
    ValueError
        Whole-file failure: bank could not be detected/mapped from column
        names (or the explicitly requested bank's shape didn't match, or
        the requested bank is Phase-12A.5-BLOCKED). Mapped to HTTP 400 by
        the caller (TRD §10) -- not a 200 with every row invalid.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    if df.empty:
        return pd.DataFrame(columns=["date", "raw_description", "merchant", "amount"]), {
            "bank_detected": bank or "",
            **_EMPTY_META,
        }

    normalized_bank = _normalize_bank_name(bank)

    if normalized_bank in _BLOCKED_BANKS:
        raise ValueError(
            f"{normalized_bank} CSV import is not yet supported -- the export format evidence "
            "for this bank is unresolved (Phase 12A.5 evidence gate). Supported banks: RBC, "
            "Scotiabank, TD, CIBC."
        )

    # -- Explicit bank selected: validate ONLY that bank, never reinterpret. --
    if normalized_bank == "RBC":
        if _fingerprint_matches(df, RBC_COLUMNS):
            return _clean_rbc(df)
        raise ValueError(f"File does not match RBC's expected column format. Got columns: {list(df.columns)}")

    if normalized_bank == "Scotiabank":
        if _fingerprint_matches(df, SCOTIA_COLUMNS):
            return _clean_scotiabank(df)
        raise ValueError(
            f"File does not match Scotiabank's expected column format. Got columns: {list(df.columns)}"
        )

    if normalized_bank == "CIBC":
        if _fingerprint_matches(df, CIBC_COLUMNS):
            return _clean_cibc(df)
        raise ValueError(f"File does not match CIBC's expected column format. Got columns: {list(df.columns)}")

    if normalized_bank == "TD":
        col_map = _find_column_mapping(df, "TD")
        if set(col_map) == {"date", "merchant", "amount"}:
            # Phase 12B closure patch: the Balance guard must apply to
            # explicit bank="TD" too, not just Auto-detect. A genuine TD
            # headered export never carries a Balance column (see
            # tests/fixtures/td_csv/clean_valid.csv) -- a file that does
            # (e.g. one of the reported BMO shapes) must be rejected here
            # even when the caller explicitly asked for TD, rather than
            # loosely accepted because its date/merchant/amount candidates
            # happen to resolve. TD's headerless fallback is untouched and
            # still reachable below regardless of this rejection.
            if _looks_like_blocked_or_unknown_balance_format(df):
                raise ValueError(
                    "File carries a Balance column that TD's header-based export never has "
                    f"-- does not match TD's expected format. Got columns: {list(df.columns)}"
                )
            return _clean_header_based(df, col_map, "TD")
        headerless = _try_td_headerless(file_bytes)
        if headerless is not None:
            return headerless
        raise ValueError(f"Could not detect TD columns from file. Got columns: {list(df.columns)}")

    # -- Auto-detect: strongest evidence first. --
    if _fingerprint_matches(df, RBC_COLUMNS):
        return _clean_rbc(df)
    if _fingerprint_matches(df, SCOTIA_COLUMNS):
        return _clean_scotiabank(df)
    if _fingerprint_matches(df, CIBC_COLUMNS):
        return _clean_cibc(df)

    if _looks_like_blocked_or_unknown_balance_format(df):
        raise ValueError(f"Could not detect bank/columns from file. Got columns: {list(df.columns)}")

    col_map = _find_column_mapping(df, "TD")
    if set(col_map) == {"date", "merchant", "amount"}:
        return _clean_header_based(df, col_map, "TD")

    # Auto-detect must also reach TD's headerless fallback (Phase 12A.5 §15
    # fix -- previously this only ran when bank="TD" was explicit, so
    # Auto-detect could never classify a genuine headerless TD file).
    headerless = _try_td_headerless(file_bytes)
    if headerless is not None:
        return headerless

    raise ValueError(f"Could not detect bank/columns from file. Got columns: {list(df.columns)}")
