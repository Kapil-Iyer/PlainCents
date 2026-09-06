"""PowerBIExportService tests (PATCH D): the export reflects live V2 data
via effective_category, is scoped by data_mode exactly like every other read
endpoint, excludes internal/debug fields, and never touches yfinance or
re-runs categorization/forecasting."""
import io
import zipfile
from unittest.mock import patch

import pandas as pd
import pytest

from backend.repositories.forecast_repository import ForecastRepository
from backend.repositories.holding_repository import HoldingRepository
from backend.repositories.price_cache_repository import PriceCacheRepository
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.powerbi_export_service import PowerBIExportService, export_filename


@pytest.fixture
def service(conn):
    return PowerBIExportService(conn)


def _txn(**overrides):
    data = {
        "date": "2026-06-05",
        "merchant": "TIM HORTONS",
        "amount": 12.5,
        "bank_source": "RBC",
        "predicted_category": "Food & Dining",
        "confirmed_category": None,
        "data_mode": "real",
        "dedup_key": "k1",
        "raw_description": "VISA DEBIT PURCHASE - 4821 TIM HORTONS",
        "decision_source": "ml_confident",
        "model_category": "Food & Dining",
    }
    data.update(overrides)
    return data


def _read_csv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(zf.read(name)))


def test_export_zip_contains_all_four_csvs(service):
    zip_bytes = service.build_export_zip(data_mode=None)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        assert set(zf.namelist()) == {
            "transactions.csv",
            "category_summary.csv",
            "portfolio.csv",
            "forecast.csv",
        }


def test_empty_mode_produces_valid_csvs_with_headers_and_no_rows(service):
    zip_bytes = service.build_export_zip(data_mode=None)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        transactions = _read_csv(zf, "transactions.csv")
        assert list(transactions.columns) == [
            "date", "merchant", "amount", "bank_source", "category", "is_manual_override",
        ]
        assert len(transactions) == 0


def test_transactions_csv_uses_effective_category_not_predicted(service, conn):
    repo = TransactionRepository(conn)
    txn_id = repo.create(_txn())
    conn.commit()
    repo.update(txn_id, {"confirmed_category": "Healthcare"})
    conn.commit()

    zip_bytes = service.build_export_zip(data_mode="real")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        transactions = _read_csv(zf, "transactions.csv")

    assert transactions.iloc[0]["category"] == "Healthcare"
    assert transactions.iloc[0]["is_manual_override"] == True  # noqa: E712 (pandas bool)


def test_transactions_csv_excludes_internal_and_privacy_sensitive_fields(service, conn):
    """raw_description (may carry masked account/reference numbers),
    merchant_key, decision_source and model_category are all internal or
    advisory-only -- none of them belongs in a report a user might share."""
    repo = TransactionRepository(conn)
    repo.create(_txn(raw_description="VISA DEBIT PURCHASE - 4821 TIM HORTONS #SECRET123"))
    conn.commit()

    zip_bytes = service.build_export_zip(data_mode="real")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        transactions = _read_csv(zf, "transactions.csv")
        raw_csv_text = zf.read("transactions.csv").decode("utf-8")

    for excluded in ["raw_description", "merchant_key", "decision_source", "model_category"]:
        assert excluded not in transactions.columns
    assert "SECRET123" not in raw_csv_text


def test_category_summary_csv_matches_month_category_aggregation(service, conn):
    repo = TransactionRepository(conn)
    repo.create(_txn(dedup_key="k1", date="2026-06-05", amount=20.0, predicted_category="Food & Dining"))
    repo.create(_txn(dedup_key="k2", date="2026-06-10", amount=30.0, predicted_category="Transport"))
    conn.commit()

    zip_bytes = service.build_export_zip(data_mode="real")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        summary = _read_csv(zf, "category_summary.csv")

    by_category = dict(zip(summary["category"], summary["total_spend"]))
    assert by_category["Food & Dining"] == 20.0
    assert by_category["Transport"] == 30.0
    assert (summary["month"] == "2026-06").all()


def test_data_mode_scoping_excludes_other_modes(service, conn):
    repo = TransactionRepository(conn)
    repo.create(_txn(dedup_key="k1", data_mode="real", amount=10.0))
    repo.create(_txn(dedup_key="k2", data_mode="demo", amount=999.0))
    conn.commit()

    zip_bytes = service.build_export_zip(data_mode="real")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        transactions = _read_csv(zf, "transactions.csv")

    assert len(transactions) == 1
    assert transactions.iloc[0]["amount"] == 10.0


def test_portfolio_csv_reuses_the_no_network_read_path(service, conn):
    HoldingRepository(conn).create(
        {"ticker": "AAPL", "shares": 10, "avg_cost": 100.0, "data_mode": "real"}
    )
    PriceCacheRepository(conn).upsert_latest("AAPL", 150.0, "2026-06-01T00:00:00")
    conn.commit()

    with patch("backend.services.portfolio_service.fetch_price") as mock_fetch:
        zip_bytes = service.build_export_zip(data_mode="real")

    mock_fetch.assert_not_called()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        portfolio = _read_csv(zf, "portfolio.csv")

    assert portfolio.iloc[0]["ticker"] == "AAPL"
    assert portfolio.iloc[0]["current_price"] == 150.0
    assert portfolio.iloc[0]["current_value"] == 1500.0


def test_forecast_csv_empty_when_no_run_exists(service):
    zip_bytes = service.build_export_zip(data_mode="real")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        forecast = _read_csv(zf, "forecast.csv")

    assert list(forecast.columns) == [
        "category", "forecast_month", "month_offset", "predicted_amount",
        "is_available", "generated_at", "is_stale",
    ]
    assert len(forecast) == 0


def test_forecast_csv_reflects_the_latest_run(service, conn):
    frepo = ForecastRepository(conn)
    run_id = frepo.create_run({"months_available": 6, "data_mode": "real", "model_impl_version": "v1"})
    frepo.save_predictions(run_id, [{
        "category": "Food & Dining", "forecast_month": "2026-07",
        "month_offset": 1, "predicted_amount": 120.0, "is_available": True,
        "unavailable_reason": None,
    }])
    conn.commit()

    zip_bytes = service.build_export_zip(data_mode="real")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        forecast = _read_csv(zf, "forecast.csv")

    assert forecast.iloc[0]["category"] == "Food & Dining"
    assert forecast.iloc[0]["predicted_amount"] == 120.0
    assert forecast.iloc[0]["is_stale"] == False  # noqa: E712 (pandas bool)


def test_export_filename_is_dated_and_zip():
    from datetime import date

    assert export_filename(date(2026, 6, 15)) == "plaincents_export_2026-06-15.zip"
