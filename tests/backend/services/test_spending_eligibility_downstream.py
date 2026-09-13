"""
Downstream-consistency regression tests for the spending-eligibility patch
(backend/services/transfer_eligibility.py): an internal/self account
transfer must not affect ANY spend aggregate -- Dashboard totals, Spending
Pace, Category Movers ("what changed"), category breakdown/trend, or the
forecast's 3-month rolling-mean input -- even though it stays visible,
labeled, in the Transactions list and in Power BI's transactions.csv (see
test_powerbi_export_service.py for that side).

All fixture merchant text and amounts are fabricated.
"""
from datetime import date

from backend.repositories.transaction_repository import TransactionRepository
from backend.services.analytics_service import AnalyticsService
from backend.services.dashboard_service import DashboardService
from backend.services.forecast_service import ForecastService


def _txn(**overrides) -> dict:
    data = {
        "date": "2026-08-15",
        "merchant": "TIM HORTONS",
        "amount": 10.0,
        "predicted_category": "Food & Dining",
        "confirmed_category": None,
        "data_mode": "real",
        "dedup_key": None,
        "transaction_type": "spending",
    }
    data.update(overrides)
    if data["dedup_key"] is None:
        data["dedup_key"] = f"{data['date']}|{data['amount']}|{data['merchant']}|dk{id(data)}"
    return data


def _seed_month(repo, month: str, day: str = "15"):
    """One genuine $30 Food & Dining purchase, plus one $500 internal
    account transfer (predicted_category="Other", same as a real
    structurally-detected transfer) -- the internal transfer must never be
    the dominant "Other" figure any of these services report."""
    repo.create(_txn(date=f"{month}-{day}", merchant="TIM HORTONS", amount=30.0,
                      dedup_key=f"spend-{month}"))
    repo.create(_txn(
        date=f"{month}-{day}", merchant="ONLINE BANKING TRANSFER", amount=500.0,
        predicted_category="Other", transaction_type="internal_transfer",
        dedup_key=f"transfer-{month}",
    ))


# -- Dashboard summary (total_spend_current/previous, category_breakdown,
#    spending_trend) --------------------------------------------------------


def test_dashboard_totals_and_breakdown_exclude_internal_transfers(conn):
    repo = TransactionRepository(conn)
    _seed_month(repo, "2026-07")
    _seed_month(repo, "2026-08")
    conn.commit()

    service = DashboardService(conn)
    summary = service.get_summary(data_mode="real", app_mode="REAL", reference_date=date(2026, 8, 15))

    assert summary["total_spend_current"] == 30.0
    assert summary["total_spend_previous"] == 30.0
    assert summary["category_breakdown"] == [
        {"category": "Food & Dining", "total_spend": 30.0, "pct_of_total": 100.0}
    ]
    august_point = next(p for p in summary["spending_trend"] if p["month"] == "2026-08")
    assert august_point["total_spend"] == 30.0


# -- Category trend / top merchants / category movers / spend pace ----------


def test_category_trend_excludes_internal_transfers(conn):
    repo = TransactionRepository(conn)
    _seed_month(repo, "2026-08")
    conn.commit()

    service = AnalyticsService(conn)
    trend = service.category_trend(data_mode="real", months=1, reference_date=date(2026, 8, 15))

    assert trend["categories"] == ["Food & Dining"]
    point = trend["points"][0]
    assert point["total_spend"] == 30.0
    assert point["by_category"] == {"Food & Dining": 30.0}


def test_top_merchants_excludes_internal_transfers(conn):
    repo = TransactionRepository(conn)
    _seed_month(repo, "2026-08")
    conn.commit()

    service = AnalyticsService(conn)
    result = service.top_merchants(data_mode="real", months=1, reference_date=date(2026, 8, 15))

    assert result["total_spend"] == 30.0
    labels = {item["merchant"] for item in result["items"]}
    assert not any("TRANSFER" in label.upper() for label in labels)


def test_category_movers_excludes_internal_transfers(conn):
    repo = TransactionRepository(conn)
    _seed_month(repo, "2026-07")
    _seed_month(repo, "2026-08")
    conn.commit()

    service = AnalyticsService(conn)
    movers = service.category_movers(data_mode="real", reference_date=date(2026, 8, 15))

    assert movers["total_current"] == 30.0
    assert movers["total_previous"] == 30.0
    # additive property still holds with the transfer excluded from both sides
    assert movers["total_change"] == round(sum(m["change"] for m in movers["movers"]), 2)


def test_spend_pace_excludes_internal_transfers(conn):
    repo = TransactionRepository(conn)
    _seed_month(repo, "2026-07")
    _seed_month(repo, "2026-08")
    conn.commit()

    service = AnalyticsService(conn)
    pace = service.spend_pace(data_mode="real", reference_date=date(2026, 8, 15))

    assert pace["current_to_date"] == 30.0
    assert pace["previous_same_point"] == 30.0


# -- Forecast: input rows exclude internal transfers, method unchanged ------


def test_forecast_input_excludes_internal_transfers(conn):
    """The 3-month rolling-mean METHOD is unchanged; only its input rows
    are. Seeding a genuine $30/month Food & Dining spend alongside a
    $500/month internal transfer (both filed under different categories, so
    a bug here would either inflate "Other" to ~500 or leak into Food &
    Dining) -- the forecast must reflect only the real $30 spend."""
    repo = TransactionRepository(conn)
    for month in ("2026-06", "2026-07", "2026-08"):
        _seed_month(repo, month)
    conn.commit()

    service = ForecastService(conn)
    result = service.run_forecast(data_mode="real")

    predictions = {p["category"]: p["predicted_amount"] for p in result["predictions"]}
    assert predictions.get("Food & Dining") == 30.0
    # "Other" must not reflect the $500/month internal transfer -- either
    # absent entirely (no genuine Other spend ever occurred) or, if present
    # for schema-completeness reasons, nowhere near the transfer amount.
    other_amount = predictions.get("Other")
    assert other_amount is None or other_amount < 100.0
