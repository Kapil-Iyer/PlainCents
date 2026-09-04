"""
pipeline.ingest.load_and_clean_from_bytes() tests (Build Plan Phase 4,
item 8: "parser tests against all four Phase 0 TD fixtures", plus the
additive headerless-TD fixture). Pure parsing correctness — no DB, no
services. load_and_clean() (V1's path-based function) is not touched or
retested here; tests/test_pipeline.py already covers it and continues to
pass unmodified.
"""
from pathlib import Path

import pytest

from pipeline.ingest import load_and_clean_from_bytes

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "td_csv"


def _read(name: str) -> bytes:
    return (FIXTURES_DIR / name).read_bytes()


def test_clean_valid_fixture_parses_all_rows():
    df, meta = load_and_clean_from_bytes(_read("clean_valid.csv"), bank="TD")
    assert len(df) == 12
    assert meta["rows_unparseable"] == 0
    assert meta["rows_skipped_credit"] == 0
    assert meta["rows_skipped_currency"] == 0
    assert meta["bank_detected"] == "TD"
    # Phase 12A.5 §12: raw_description now flows through (previously
    # discarded downstream) alongside merchant.
    assert list(df.columns) == ["date", "raw_description", "merchant", "amount"]
    assert df["date"].iloc[0] == "2026-01-05"
    assert df["amount"].iloc[0] == 6.75


def test_unparseable_dates_fixture_drops_and_counts_bad_rows():
    df, meta = load_and_clean_from_bytes(_read("unparseable_dates.csv"), bank="TD")
    # 7 data rows, 3 with malformed dates.
    assert meta["rows_unparseable"] == 3
    assert len(df) == 4


def test_unrecognized_format_fixture_raises_value_error():
    with pytest.raises(ValueError):
        load_and_clean_from_bytes(_read("unrecognized_format.csv"), bank="TD")


def test_duplicate_rows_fixture_preserves_intrafile_duplicates():
    # Phase 12B dedup fix: the V2 bytes path no longer calls
    # drop_duplicates() (Phase 12A finding -- it ran before
    # IngestionService could assign occurrence_index, silently collapsing
    # legitimate repeated transactions). All 8 rows -- including the
    # repeated pairs/triples -- must now survive at the parser layer; it is
    # IngestionService's occurrence_index + dedup_key that distinguishes
    # them, not upstream row-collapsing.
    df, meta = load_and_clean_from_bytes(_read("duplicate_rows.csv"), bank="TD")
    assert len(df) == 8
    assert meta["rows_unparseable"] == 0


def test_headerless_td_fixture_parses_withdrawals_and_excludes_deposit_only_rows():
    df, meta = load_and_clean_from_bytes(_read("headerless_positional.csv"), bank="TD")
    # 5 rows total; 1 is deposit-only (PAYROLL DEPOSIT, no withdrawal). It's
    # correctly recognized and intentionally excluded (out of spend-tracking
    # scope), not malformed -- Phase 12A.5 §17/§24 tracks that separately
    # from rows_unparseable now.
    assert meta["rows_total"] == 5
    assert meta["rows_unparseable"] == 0
    assert meta["rows_skipped_credit"] == 1
    assert len(df) == 4
    assert "PAYROLL DEPOSIT" not in df["merchant"].tolist()
    assert df["date"].iloc[0] == "2024-08-27"
    assert df["amount"].iloc[0] == 6.75


def test_headerless_fallback_never_used_for_non_td_bank():
    # The headerless fallback is TD-only (Build Plan Phase 4 scope: TD import
    # only, no new banks) — requesting bank="RBC" against a file this
    # fallback would otherwise recognize must still raise, not silently
    # reinterpret it.
    with pytest.raises(ValueError):
        load_and_clean_from_bytes(_read("headerless_positional.csv"), bank="RBC")


def test_load_and_clean_path_based_function_is_unmodified_and_still_importable():
    # Regression guard: the V1 entry point's signature/behavior must be
    # untouched by this phase's additive changes.
    from pipeline.ingest import load_and_clean

    df = load_and_clean(FIXTURES_DIR / "clean_valid.csv", bank="TD")
    assert len(df) == 12
    assert list(df.columns) == ["date", "merchant", "amount"]
