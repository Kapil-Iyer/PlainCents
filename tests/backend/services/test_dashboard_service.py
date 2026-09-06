"""DashboardService tests (Build Plan Phase 6, item 8): current/previous
calendar-month math, month-boundary edge cases, effective-category usage,
data_mode filtering, and empty-data shape."""
from datetime import date

from backend.repositories.transaction_repository import TransactionRepository
from backend.services.dashboard_service import DashboardService


def _txn(**overrides) -> dict:
    data = {
        "date": "2026-06-15",
        "merchant": "TIM HORTONS",
        "amount": 10.0,
        "predicted_category": "Food & Dining",
        "confirmed_category": None,
        "data_mode": "real",
        "dedup_key": None,
    }
    data.update(overrides)
    if data["dedup_key"] is None:
        data["dedup_key"] = f"{data['date']}|{data['amount']}|{data['merchant']}|dk{id(data)}"
    return data


def test_empty_database_returns_zero_shaped_summary(conn):
    service = DashboardService(conn)

    summary = service.get_summary(data_mode=None, app_mode="EMPTY", reference_date=date(2026, 6, 15))

    assert summary["total_spend_current"] == 0
    assert summary["total_spend_previous"] == 0
    assert summary["change_pct"] == 0.0
    assert summary["category_breakdown"] == []
    assert summary["recent_transactions"] == []
    assert len(summary["spending_trend"]) == 6
    assert all(p["total_spend"] == 0 for p in summary["spending_trend"])
    assert summary["data_mode"] == "EMPTY"
    assert summary["forecast_summary"] is None
    assert summary["portfolio_summary"] is None


def test_current_vs_previous_calendar_month_totals(conn):
    repo = TransactionRepository(conn)
    repo.create(_txn(date="2026-06-05", amount=20.0))
    repo.create(_txn(date="2026-06-20", amount=30.0))
    repo.create(_txn(date="2026-05-10", amount=15.0))  # previous month
    repo.create(_txn(date="2026-04-01", amount=999.0))  # outside window
    conn.commit()

    service = DashboardService(conn)
    summary = service.get_summary(data_mode="real", app_mode="REAL", reference_date=date(2026, 6, 15))

    assert summary["period"] == {"current": "2026-06", "previous": "2026-05"}
    assert summary["total_spend_current"] == 50.0
    assert summary["total_spend_previous"] == 15.0
    assert summary["change_pct"] == round((50.0 - 15.0) / 15.0 * 100, 1)


def test_month_boundary_first_day_of_january_rolls_to_prior_december(conn):
    repo = TransactionRepository(conn)
    repo.create(_txn(date="2026-01-01", amount=40.0))
    repo.create(_txn(date="2025-12-31", amount=25.0))
    conn.commit()

    service = DashboardService(conn)
    summary = service.get_summary(data_mode="real", app_mode="REAL", reference_date=date(2026, 1, 1))

    assert summary["period"] == {"current": "2026-01", "previous": "2025-12"}
    assert summary["total_spend_current"] == 40.0
    assert summary["total_spend_previous"] == 25.0


def test_change_pct_is_none_when_previous_is_zero_and_current_is_not(conn):
    repo = TransactionRepository(conn)
    repo.create(_txn(date="2026-06-10", amount=50.0))
    conn.commit()

    service = DashboardService(conn)
    summary = service.get_summary(data_mode="real", app_mode="REAL", reference_date=date(2026, 6, 15))

    assert summary["total_spend_previous"] == 0
    assert summary["change_pct"] is None


def test_category_breakdown_uses_effective_category_and_sums_pct_to_100(conn):
    repo = TransactionRepository(conn)
    id1 = repo.create(_txn(date="2026-06-05", amount=30.0, predicted_category="Shopping"))
    repo.create(_txn(date="2026-06-06", amount=70.0, predicted_category="Food & Dining"))
    conn.commit()
    # Confirmed category overrides predicted — effective_category must be
    # what the breakdown groups by, not predicted_category (PRD §9.3/§4.1).
    repo.update(id1, {"confirmed_category": "Entertainment"})
    conn.commit()

    service = DashboardService(conn)
    summary = service.get_summary(data_mode="real", app_mode="REAL", reference_date=date(2026, 6, 15))

    by_category = {item["category"]: item for item in summary["category_breakdown"]}
    assert "Shopping" not in by_category
    assert by_category["Entertainment"]["total_spend"] == 30.0
    assert by_category["Food & Dining"]["total_spend"] == 70.0
    assert by_category["Food & Dining"]["pct_of_total"] == 70.0
    assert by_category["Entertainment"]["pct_of_total"] == 30.0
    # Highest spend first.
    assert summary["category_breakdown"][0]["category"] == "Food & Dining"


def test_data_mode_filtering_excludes_other_modes(conn):
    repo = TransactionRepository(conn)
    repo.create(_txn(date="2026-06-05", amount=100.0, data_mode="demo"))
    repo.create(_txn(date="2026-06-06", amount=25.0, data_mode="real"))
    conn.commit()

    service = DashboardService(conn)
    summary = service.get_summary(data_mode="real", app_mode="REAL", reference_date=date(2026, 6, 15))

    assert summary["total_spend_current"] == 25.0


def test_recent_transactions_limited_and_most_recent_first(conn):
    repo = TransactionRepository(conn)
    for day in range(1, 8):
        repo.create(_txn(date=f"2026-06-{day:02d}", amount=float(day)))
    conn.commit()

    service = DashboardService(conn)
    summary = service.get_summary(data_mode="real", app_mode="REAL", reference_date=date(2026, 6, 15))

    assert len(summary["recent_transactions"]) == 5
    dates = [t["date"] for t in summary["recent_transactions"]]
    assert dates == sorted(dates, reverse=True)


def test_spending_trend_zero_fills_months_with_no_data(conn):
    repo = TransactionRepository(conn)
    repo.create(_txn(date="2026-06-10", amount=42.0))
    conn.commit()

    service = DashboardService(conn)
    summary = service.get_summary(data_mode="real", app_mode="REAL", reference_date=date(2026, 6, 15))

    months = [p["month"] for p in summary["spending_trend"]]
    assert months == ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    totals = {p["month"]: p["total_spend"] for p in summary["spending_trend"]}
    assert totals["2026-06"] == 42.0
    assert totals["2026-05"] == 0.0


# -- partial-current-month vs full-previous-month (product-semantics fix) ----


def test_change_pct_compares_partial_month_against_equivalent_elapsed_previous_period(conn):
    """The bug this guards against: early in a month, `change_pct` must NOT
    compare a partial current month against the FULL previous month (which
    always reads as a steep decline at identical daily pace). It must be
    computed against the previous month's spend through the SAME
    day-of-month -- `total_spend_previous_to_date` -- while
    `total_spend_previous` keeps reporting the full previous month as its
    own separate, honest number."""
    repo = TransactionRepository(conn)
    # Reference date: June 5 -- only 5 days into the current month.
    repo.create(_txn(date="2026-06-03", amount=10.0))  # current month, day 3 (<=5)
    # Previous month (May): spread across the whole month.
    repo.create(_txn(date="2026-05-02", amount=8.0))    # day 2  (<=5, comparable)
    repo.create(_txn(date="2026-05-04", amount=7.0))    # day 4  (<=5, comparable)
    repo.create(_txn(date="2026-05-20", amount=500.0))  # day 20 (>5, NOT comparable)
    conn.commit()

    service = DashboardService(conn)
    summary = service.get_summary(data_mode="real", app_mode="REAL", reference_date=date(2026, 6, 5))

    assert summary["comparable_day"] == 5
    assert summary["total_spend_current"] == 10.0
    # Full previous month total still includes the day-20 transaction.
    assert summary["total_spend_previous"] == 515.0
    # But the fair comparison basis stops at day 5, same as current.
    assert summary["total_spend_previous_to_date"] == 15.0
    # change_pct must be computed from the capped figure (10 vs 15), NOT the
    # full-month figure (10 vs 515, which would read as a ~-98% collapse).
    assert summary["change_pct"] == round((10.0 - 15.0) / 15.0 * 100, 1)


def test_comparable_day_caps_at_previous_months_length(conn):
    """March 31 vs February: the comparable window can't run past
    February's own length (28/29 days), so it caps there rather than
    computing a nonexistent "February 31"."""
    repo = TransactionRepository(conn)
    repo.create(_txn(date="2026-03-31", amount=20.0))
    repo.create(_txn(date="2026-02-28", amount=5.0))  # last real day of Feb 2026 (not a leap year)
    conn.commit()

    service = DashboardService(conn)
    summary = service.get_summary(data_mode="real", app_mode="REAL", reference_date=date(2026, 3, 31))

    assert summary["comparable_day"] == 28
    assert summary["total_spend_previous_to_date"] == 5.0


def test_change_pct_on_day_1_compares_a_single_day_on_each_side(conn):
    """Day 1 of the month: the comparable window is exactly one day on each
    side, not zero days and not the whole previous month."""
    repo = TransactionRepository(conn)
    repo.create(_txn(date="2026-06-01", amount=25.0))
    repo.create(_txn(date="2026-05-01", amount=20.0))
    repo.create(_txn(date="2026-05-20", amount=1000.0))  # must not leak into day-1 comparison
    conn.commit()

    service = DashboardService(conn)
    summary = service.get_summary(data_mode="real", app_mode="REAL", reference_date=date(2026, 6, 1))

    assert summary["comparable_day"] == 1
    assert summary["total_spend_current"] == 25.0
    assert summary["total_spend_previous_to_date"] == 20.0
    assert summary["total_spend_previous"] == 1020.0  # full-month figure unaffected
    assert summary["change_pct"] == round((25.0 - 20.0) / 20.0 * 100, 1)


def test_change_pct_handles_leap_year_february(conn):
    """2028 is a leap year -- March 30 2028 vs February 2028 (29 real days)
    must cap at day 29, not 28."""
    repo = TransactionRepository(conn)
    repo.create(_txn(date="2028-03-30", amount=10.0))
    repo.create(_txn(date="2028-02-29", amount=6.0))  # Feb 29 exists this year
    conn.commit()

    service = DashboardService(conn)
    summary = service.get_summary(data_mode="real", app_mode="REAL", reference_date=date(2028, 3, 30))

    assert summary["comparable_day"] == 29
    assert summary["total_spend_previous_to_date"] == 6.0
