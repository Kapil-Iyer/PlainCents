"""
Six-bank parser tests (Phase 12A.5/12B): pure parsing correctness for the
four implemented adapters (RBC, Scotiabank, TD, CIBC) plus the auto-detect
order, the BMO/National BLOCKED explicit-bank behavior, and the
anti-misclassification guard for recognized-but-unsupported shapes. No DB,
no services -- see test_ingestion_service.py / test_imports.py for the
full-stack equivalents.
"""
from pathlib import Path

import pytest

from pipeline.ingest import load_and_clean_from_bytes

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"
RBC_DIR = FIXTURES / "rbc_csv"
SCOTIA_DIR = FIXTURES / "scotia_csv"
CIBC_DIR = FIXTURES / "cibc_csv"
TD_DIR = FIXTURES / "td_csv"
SHARED_DIR = FIXTURES / "shared_csv"


def _read(directory: Path, name: str) -> bytes:
    return (directory / name).read_bytes()


# -- RBC ----------------------------------------------------------------


def test_rbc_explicit_bank_clean_valid():
    df, meta = load_and_clean_from_bytes(_read(RBC_DIR, "clean_valid.csv"), bank="RBC")
    assert meta["bank_detected"] == "RBC"
    assert meta["rows_total"] == 9
    # 2 identical TIM HORTONS spends + 1 cheque payment spend = 3 valid rows
    assert len(df) == 3
    assert meta["rows_skipped_credit"] == 1  # PAYROLL DEPOSIT (positive CAD$)
    assert meta["rows_skipped_currency"] == 1  # USD$-only AMAZON row
    # both-populated, neither-populated, CAD$==0, and bad-date rows
    assert meta["rows_unparseable"] == 4
    assert list(df.columns) == ["date", "raw_description", "merchant", "amount"]
    assert df["date"].iloc[0] == "2026-08-03"
    assert df["amount"].iloc[0] == 6.75  # abs(-6.75), positive-for-spend convention
    assert "ACCOUNT" not in "".join(df.columns).upper()  # account number/type never propagate


def test_rbc_auto_detect():
    df, meta = load_and_clean_from_bytes(_read(RBC_DIR, "clean_valid.csv"), bank=None)
    assert meta["bank_detected"] == "RBC"
    assert len(df) == 3


def test_rbc_description_join_blank_description_2():
    df, meta = load_and_clean_from_bytes(_read(RBC_DIR, "clean_valid.csv"), bank="RBC")
    tim_hortons_row = df[df["raw_description"] == "TIM HORTONS #123"]
    assert len(tim_hortons_row) == 2  # blank Description 2 -> no join, no trailing space


def test_rbc_missing_header_rejected():
    with pytest.raises(ValueError):
        load_and_clean_from_bytes(_read(RBC_DIR, "missing_header.csv"), bank="RBC")


def test_rbc_explicit_bank_never_reinterpreted_as_scotia():
    with pytest.raises(ValueError):
        load_and_clean_from_bytes(_read(SCOTIA_DIR, "clean_valid.csv"), bank="RBC")


# -- Scotiabank -----------------------------------------------------------


def test_scotiabank_explicit_bank_clean_valid():
    df, meta = load_and_clean_from_bytes(_read(SCOTIA_DIR, "clean_valid.csv"), bank="Scotiabank")
    assert meta["bank_detected"] == "Scotiabank"
    assert meta["rows_total"] == 9
    # 2 identical TIM HORTONS + GROCERY MART + COFFEE SHOP = 4 valid spend rows
    assert len(df) == 4
    assert meta["rows_skipped_credit"] == 1  # PAYROLL (Credit + positive)
    # Debit+positive, Credit+negative, unknown type, bad date
    assert meta["rows_unparseable"] == 4
    assert meta["rows_skipped_currency"] == 0
    assert df["date"].iloc[0] == "2026-08-31"
    assert df["amount"].iloc[0] == 6.75


def test_scotiabank_auto_detect():
    df, meta = load_and_clean_from_bytes(_read(SCOTIA_DIR, "clean_valid.csv"), bank=None)
    assert meta["bank_detected"] == "Scotiabank"


def test_scotiabank_sub_description_join():
    df, meta = load_and_clean_from_bytes(_read(SCOTIA_DIR, "clean_valid.csv"), bank="Scotiabank")
    grocery_row = df[df["raw_description"] == "GROCERY MART LOYALTY POINTS EARNED"]
    assert len(grocery_row) == 1


def test_scotiabank_filter_and_balance_never_persisted():
    df, meta = load_and_clean_from_bytes(_read(SCOTIA_DIR, "clean_valid.csv"), bank="Scotiabank")
    assert "filter" not in [c.lower() for c in df.columns]
    assert "balance" not in [c.lower() for c in df.columns]


def test_scotiabank_missing_header_rejected():
    with pytest.raises(ValueError):
        load_and_clean_from_bytes(_read(SCOTIA_DIR, "missing_header.csv"), bank="Scotiabank")


# -- CIBC (research-backed, fail-closed) -----------------------------------


def test_cibc_explicit_bank_clean_valid():
    df, meta = load_and_clean_from_bytes(_read(CIBC_DIR, "clean_valid.csv"), bank="CIBC")
    assert meta["bank_detected"] == "CIBC"
    assert meta["rows_total"] == 6
    assert len(df) == 2  # 2 identical COFFEE SHOP withdrawals
    assert meta["rows_skipped_credit"] == 1  # PAYCHEQUE (deposit-only)
    assert meta["rows_unparseable"] == 3  # both, neither, bad date
    assert df["amount"].iloc[0] == 6.75


def test_cibc_auto_detect():
    df, meta = load_and_clean_from_bytes(_read(CIBC_DIR, "clean_valid.csv"), bank=None)
    assert meta["bank_detected"] == "CIBC"


def test_cibc_no_headerless_fallback():
    # Strip the header off a valid CIBC file entirely; CIBC has no
    # headerless inference, so this must NOT match CIBC. It also should not
    # accidentally match TD's headerless fallback (4-5 col positional) as a
    # coincidence -- CIBC is 5 columns like TD-headerless's max width, so
    # this specifically proves CIBC's data doesn't leak into TD's fallback.
    content = _read(CIBC_DIR, "clean_valid.csv").decode()
    headerless = "\n".join(content.splitlines()[1:]).encode()
    df, meta = load_and_clean_from_bytes(headerless, bank=None)
    # This is allowed to either raise, or (if it coincidentally satisfies
    # TD-headerless's date/column heuristics) resolve as TD -- but it must
    # never claim to be CIBC, since CIBC has no headerless path at all.
    assert meta["bank_detected"] != "CIBC"


def test_cibc_missing_header_rejected():
    with pytest.raises(ValueError):
        load_and_clean_from_bytes(_read(CIBC_DIR, "missing_header.csv"), bank="CIBC")


def test_cibc_never_misdetected_as_td_in_auto_mode():
    # CIBC's real header (Transaction Date, Description, Withdrawals,
    # Deposits, Balance) shares no column names with TD's loose candidate
    # list well enough to fully resolve date+merchant+amount, so it must
    # not silently become TD.
    df, meta = load_and_clean_from_bytes(_read(CIBC_DIR, "clean_valid.csv"), bank=None)
    assert meta["bank_detected"] == "CIBC"


# -- TD (project-verified; must still work via Auto) -----------------------


def test_td_auto_detect_header_based():
    df, meta = load_and_clean_from_bytes(_read(TD_DIR, "clean_valid.csv"), bank=None)
    assert meta["bank_detected"] == "TD"
    assert len(df) == 12


def test_td_auto_detect_headerless():
    # Phase 12A.5 §15 fix: Auto-detect must reach TD's headerless fallback,
    # not just explicit bank="TD" (Phase 12A found this blocker).
    df, meta = load_and_clean_from_bytes(_read(TD_DIR, "headerless_positional.csv"), bank=None)
    assert meta["bank_detected"] == "TD"
    assert len(df) == 4


def test_td_explicit_bank_still_works():
    df, meta = load_and_clean_from_bytes(_read(TD_DIR, "clean_valid.csv"), bank="TD")
    assert meta["bank_detected"] == "TD"


# -- BMO / National Bank: evidence-BLOCKED ---------------------------------


def test_bmo_explicit_bank_raises_clear_blocked_error():
    with pytest.raises(ValueError, match="not yet supported"):
        load_and_clean_from_bytes(_read(SHARED_DIR, "blocked_balance_format.csv"), bank="BMO")


def test_national_bank_explicit_bank_raises_clear_blocked_error():
    with pytest.raises(ValueError, match="not yet supported"):
        load_and_clean_from_bytes(_read(SHARED_DIR, "blocked_balance_format.csv"), bank="National Bank")


def test_blocked_balance_format_never_silently_becomes_td_in_auto_mode():
    # The exact Phase 12A finding: Date,Description,Amount,Balance would
    # satisfy TD's loose candidate-list match if nothing stopped it. Must be
    # rejected outright in Auto mode, not absorbed as "TD".
    with pytest.raises(ValueError):
        load_and_clean_from_bytes(_read(SHARED_DIR, "blocked_balance_format.csv"), bank=None)


def test_blocked_balance_format_rejected_under_explicit_td_too():
    # Phase 12B closure patch (Cursor finding): explicit bank="TD" must not
    # bypass the Balance guard that Auto-detect already enforced. Before
    # this patch, requesting bank="TD" against this exact file was wrongly
    # accepted as TD.
    with pytest.raises(ValueError):
        load_and_clean_from_bytes(_read(SHARED_DIR, "blocked_balance_format.csv"), bank="TD")


def test_genuine_td_headered_fixture_still_accepts_under_explicit_td():
    # Regression guard: the new Balance rejection must not touch genuine TD
    # headered files, which never carry a Balance column.
    df, meta = load_and_clean_from_bytes(_read(TD_DIR, "clean_valid.csv"), bank="TD")
    assert meta["bank_detected"] == "TD"
    assert len(df) == 12


def test_genuine_td_headerless_fixture_still_accepts_under_explicit_and_auto_td():
    # Regression guard: headerless TD is untouched by the Balance guard
    # (headerless has no column names to inspect in the first place).
    explicit_df, explicit_meta = load_and_clean_from_bytes(
        _read(TD_DIR, "headerless_positional.csv"), bank="TD"
    )
    assert explicit_meta["bank_detected"] == "TD"
    assert len(explicit_df) == 4

    auto_df, auto_meta = load_and_clean_from_bytes(_read(TD_DIR, "headerless_positional.csv"), bank=None)
    assert auto_meta["bank_detected"] == "TD"
    assert len(auto_df) == 4


# -- Cross-bank ambiguity / unsupported format ------------------------------


def test_ambiguous_five_column_file_never_becomes_td():
    with pytest.raises(ValueError):
        load_and_clean_from_bytes(_read(SHARED_DIR, "ambiguous_five_column.csv"), bank=None)


def test_unknown_bank_name_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown bank"):
        load_and_clean_from_bytes(_read(TD_DIR, "clean_valid.csv"), bank="Not A Real Bank")


# -- Cross-bank canonical amount consistency --------------------------------


def test_equivalent_spend_normalizes_to_same_positive_amount_across_banks():
    rbc_df, _ = load_and_clean_from_bytes(_read(RBC_DIR, "clean_valid.csv"), bank="RBC")
    scotia_df, _ = load_and_clean_from_bytes(_read(SCOTIA_DIR, "clean_valid.csv"), bank="Scotiabank")
    cibc_df, _ = load_and_clean_from_bytes(_read(CIBC_DIR, "clean_valid.csv"), bank="CIBC")
    td_df, _ = load_and_clean_from_bytes(_read(TD_DIR, "clean_valid.csv"), bank="TD")

    # Every bank's $6.75 spend example normalizes to the identical positive
    # canonical amount, regardless of each source format's own sign/column
    # convention.
    assert rbc_df["amount"].iloc[0] == 6.75
    assert scotia_df["amount"].iloc[0] == 6.75
    assert cibc_df["amount"].iloc[0] == 6.75
    assert td_df["amount"].iloc[0] == 6.75
