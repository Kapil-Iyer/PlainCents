"""
Unit tests for backend/services/date_windows.py -- the one shared source of
truth for "compare a possibly-partial current month against the SAME
elapsed days of the previous month" arithmetic (used by AnalyticsService's
category_movers/spend_pace and DashboardService.get_summary).

Pure date arithmetic, no DB required.
"""
from datetime import date

import pytest

from backend.services.date_windows import (
    analysis_window,
    elapsed_window,
    month_start,
    month_str,
    shift_month,
)


# -- shift_month / month_str / month_start -----------------------------------


@pytest.mark.parametrize("year,month,delta,expected", [
    (2026, 6, -1, (2026, 5)),
    (2026, 1, -1, (2025, 12)),   # January -> wraps to previous year's December
    (2026, 12, 1, (2027, 1)),    # December -> wraps forward to next year's January
    (2026, 6, 0, (2026, 6)),
    (2026, 3, -14, (2025, 1)),   # multi-year shift wraps correctly, not just +/-1 month
])
def test_shift_month(year, month, delta, expected):
    assert shift_month(year, month, delta) == expected


def test_month_str_and_month_start_are_zero_padded():
    assert month_str(2026, 3) == "2026-03"
    assert month_start(2026, 3) == "2026-03-01"
    assert month_str(2026, 12) == "2026-12"


# -- elapsed_window -----------------------------------------------------------


def test_elapsed_window_basic_september_4th():
    """The prompt's own example: on Sep 4, compare Sep 1-4 against Aug 1-4."""
    window = elapsed_window(date(2026, 9, 4))

    assert window.current_month == "2026-09"
    assert window.previous_month == "2026-08"
    assert window.current_start == "2026-09-01"
    assert window.previous_start == "2026-08-01"
    assert window.day_of_month == 4
    assert window.comparable_day == 4
    assert window.previous_comparable_end == "2026-08-04"


def test_elapsed_window_day_1():
    """Day 1 of any month: the comparable window is exactly one day on each
    side (day 1 vs day 1), never zero and never the whole previous month."""
    window = elapsed_window(date(2026, 6, 1))

    assert window.day_of_month == 1
    assert window.comparable_day == 1
    assert window.previous_comparable_end == "2026-05-01"


def test_elapsed_window_year_boundary_january():
    """January's previous month is December of the PRIOR year, not month 0
    or a same-year December."""
    window = elapsed_window(date(2026, 1, 15))

    assert window.previous_month == "2025-12"
    assert window.previous_start == "2025-12-01"
    assert window.comparable_day == 15
    assert window.previous_comparable_end == "2025-12-15"


def test_elapsed_window_current_day_exceeds_previous_month_length():
    """March 31 vs February (28 days in 2026, not a leap year): the
    comparable window caps at February's own length rather than computing a
    nonexistent 'February 31'."""
    window = elapsed_window(date(2026, 3, 31))

    assert window.previous_month == "2026-02"
    assert window.previous_month_length == 28
    assert window.day_of_month == 31
    assert window.comparable_day == 28
    assert window.previous_comparable_end == "2026-02-28"


def test_elapsed_window_leap_year_february_has_29_days():
    """2028 is a leap year -- comparing March 30 2028 against February 2028
    must cap at day 29 (February's real length that year), not 28."""
    window = elapsed_window(date(2028, 3, 30))

    assert window.previous_month == "2028-02"
    assert window.previous_month_length == 29
    assert window.comparable_day == 29
    assert window.previous_comparable_end == "2028-02-29"


def test_elapsed_window_leap_year_day_1_of_march():
    """Day 1 still behaves normally in a leap year -- no special-casing
    needed since comparable_day = min(day_of_month, previous_month_length)
    is 1 regardless of how long February was."""
    window = elapsed_window(date(2028, 3, 1))

    assert window.comparable_day == 1
    assert window.previous_comparable_end == "2028-02-01"


def test_elapsed_window_previous_month_shorter_is_symmetric_not_only_february():
    """The cap applies to ANY current/previous month-length mismatch, not
    just "February" specifically -- May 31 vs April (30 days)."""
    window = elapsed_window(date(2026, 5, 31))

    assert window.previous_month == "2026-04"
    assert window.previous_month_length == 30
    assert window.comparable_day == 30
    assert window.previous_comparable_end == "2026-04-30"


# -- analysis_window: selected-month generalization --------------------------


def test_analysis_window_defaults_to_current_month_matches_elapsed_window():
    """No selected_month -- reproduces elapsed_window's exact behavior, so
    every pre-existing caller (dashboard/analytics defaults) is unaffected."""
    today = date(2026, 9, 6)
    window = analysis_window(today)
    ew = elapsed_window(today)

    assert window.is_current_incomplete is True
    assert window.selected_month == ew.current_month == "2026-09"
    assert window.previous_month == ew.previous_month == "2026-08"
    assert window.current_start == ew.current_start
    assert window.current_end == "2026-09-06"  # today, not month-end
    assert window.previous_start == ew.previous_start
    assert window.previous_end == ew.previous_comparable_end
    assert window.comparable_day == ew.comparable_day == 6
    assert window.current_month_length == 30


def test_analysis_window_explicit_current_month_is_still_incomplete():
    today = date(2026, 9, 6)
    window = analysis_window(today, selected_month="2026-09")

    assert window.is_current_incomplete is True
    assert window.current_end == "2026-09-06"


def test_analysis_window_historical_month_uses_full_month_both_sides():
    """Selecting a fully-completed past month: FULL selected month vs FULL
    previous month, never capped -- the core historical semantics this
    module adds."""
    today = date(2026, 9, 6)
    window = analysis_window(today, selected_month="2026-08")

    assert window.is_current_incomplete is False
    assert window.selected_month == "2026-08"
    assert window.previous_month == "2026-07"
    assert window.current_start == "2026-08-01"
    assert window.current_end == "2026-08-31"  # full month, not capped at "today"
    assert window.previous_start == "2026-07-01"
    assert window.previous_end == "2026-07-31"  # full previous month too
    assert window.current_month_length == 31
    assert window.previous_month_length == 31


def test_analysis_window_historical_year_boundary():
    today = date(2026, 9, 6)
    window = analysis_window(today, selected_month="2026-01")

    assert window.previous_month == "2025-12"
    assert window.previous_start == "2025-12-01"
    assert window.previous_end == "2025-12-31"


def test_analysis_window_historical_february_completed():
    today = date(2026, 9, 6)
    window = analysis_window(today, selected_month="2026-02")

    assert window.selected_month == "2026-02"
    assert window.current_month_length == 28  # 2026 is not a leap year
    assert window.current_end == "2026-02-28"
    assert window.previous_month == "2026-01"
    assert window.previous_end == "2026-01-31"


def test_analysis_window_historical_leap_year_february():
    today = date(2028, 9, 6)
    window = analysis_window(today, selected_month="2028-02")

    assert window.current_month_length == 29
    assert window.current_end == "2028-02-29"


def test_analysis_window_historical_month_length_mismatch_sets_comparable_day():
    """Historical March (31 days) vs February (28 days in 2026): comparable_day
    is the SHORTER of the two full lengths -- the last day both months
    actually share, distinct from the incomplete-month meaning of
    comparable_day (which is about "today", not month length)."""
    today = date(2026, 9, 6)
    window = analysis_window(today, selected_month="2026-03")

    assert window.current_month_length == 31
    assert window.previous_month_length == 28
    assert window.comparable_day == 28
    # Full months on both sides regardless of the length mismatch.
    assert window.current_end == "2026-03-31"
    assert window.previous_end == "2026-02-28"
