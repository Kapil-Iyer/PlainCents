"""
Shared elapsed-period date-window helpers.

WHY THIS EXISTS
---------------
Comparing "this month" against "last month" is misleading whenever the
current month is not yet over: a partial current month (day 1 through
today) compared against a FULL previous month always reads as a decline,
even at an identical daily pace -- early in a month this can read as a
"-100%" change from a single day's data.

`AnalyticsService.spend_pace` already solved this correctly: it caps BOTH
the current and previous cumulative totals at the same day-of-month.
`DashboardService.get_summary` and `AnalyticsService.category_movers` did
not -- they compared a partial current-month total against the FULL
previous month, with no day-of-month cap on the previous side. Both files
also independently redefined `_month_str`/`_shift_month`. This module
factors `spend_pace`'s already-correct elapsed-window arithmetic out into
one shared place, so the fix does not have to be (and cannot silently drift
into being) reimplemented a third time.

WHAT THIS DOES NOT CHANGE
-------------------------
A "full previous calendar month" total is still a genuinely useful, honest
number on its own (see `total_spend_previous` in `DashboardSummaryResponse`)
and callers are free to keep reporting it standalone. `elapsed_window()`
exists to compute the SEPARATE, comparable "previous month, capped at the
same day-of-month" figure that a fair MoM comparison actually needs --
callers combine both rather than this module deciding which one a UI shows.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date


def shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    """Shift a (year, month) pair by `delta` months (may be negative),
    wrapping the year correctly."""
    zero_based = (month - 1) + delta
    return year + zero_based // 12, zero_based % 12 + 1


def month_str(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def month_start(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}-01"


@dataclass(frozen=True)
class ElapsedWindow:
    """The comparable elapsed-day window for `reference_date`'s month
    against the previous one.

    Both periods conceptually run from day 1 through `comparable_day` -- the
    SAME day-of-month -- so a partial current month is never compared
    against more of the previous month than it has itself lived through.
    `comparable_day` is capped at the previous month's own length (so a
    March 31 reference date compares against all of February, not a
    nonexistent February 31).
    """

    current_month: str            # "YYYY-MM"
    previous_month: str           # "YYYY-MM"
    current_start: str            # "YYYY-MM-01"
    previous_start: str           # "YYYY-MM-01"
    day_of_month: int             # reference_date.day
    previous_month_length: int    # days in the previous calendar month
    comparable_day: int           # min(day_of_month, previous_month_length)
    previous_comparable_end: str  # "YYYY-MM-DD", last day INCLUDED in the capped previous window


def elapsed_window(reference_date: date) -> ElapsedWindow:
    today = reference_date
    prev_y, prev_m = shift_month(today.year, today.month, -1)
    previous_month_length = calendar.monthrange(prev_y, prev_m)[1]
    comparable_day = min(today.day, previous_month_length)
    return ElapsedWindow(
        current_month=month_str(today.year, today.month),
        previous_month=month_str(prev_y, prev_m),
        current_start=month_start(today.year, today.month),
        previous_start=month_start(prev_y, prev_m),
        day_of_month=today.day,
        previous_month_length=previous_month_length,
        comparable_day=comparable_day,
        previous_comparable_end=f"{prev_y:04d}-{prev_m:02d}-{comparable_day:02d}",
    )
