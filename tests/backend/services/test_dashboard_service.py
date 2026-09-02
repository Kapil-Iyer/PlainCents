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
