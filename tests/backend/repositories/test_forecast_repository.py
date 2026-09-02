"""Phase 1 tests: ForecastRepository. Requirements #8, #9, #12."""
import sqlite3

import pytest

from backend.repositories.forecast_repository import ForecastRepository


def _run_data(data_mode="real"):
    return {"months_available": 12, "months_required": 12, "data_mode": data_mode, "model_impl_version": "rf_v1_default_hparams"}


def _prediction(category="Shopping", month="2026-10"):
    return {
        "category": category, "forecast_month": month, "month_offset": 1,
        "predicted_amount": 120.50, "is_available": True,
    }


def test_8_prediction_uniqueness_scoped_to_run(conn):
    repo = ForecastRepository(conn)
    run_id = repo.create_run(_run_data())
    conn.commit()
    repo.save_predictions(run_id, [_prediction()])
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_predictions(run_id, [_prediction()])  # same run, same category+month


def test_9_same_category_month_allowed_across_different_runs(conn):
    repo = ForecastRepository(conn)
    run_a = repo.create_run(_run_data())
    run_b = repo.create_run(_run_data())
    conn.commit()

    repo.save_predictions(run_a, [_prediction(category="Shopping", month="2026-10")])
    repo.save_predictions(run_b, [_prediction(category="Shopping", month="2026-10")])
    conn.commit()  # must not raise

    assert len(repo.get_predictions(run_a)) == 1
    assert len(repo.get_predictions(run_b)) == 1


def test_12_crud_round_trip_and_retention(conn):
    repo = ForecastRepository(conn)
    run_a = repo.create_run(_run_data())
    conn.commit()
    assert repo.get_run(run_a)["months_available"] == 12
    assert repo.get_latest_run(data_mode="real")["id"] == run_a

    run_b = repo.create_run(_run_data())  # a second run must not overwrite the first
    conn.commit()
    assert repo.get_run(run_a) is not None  # prior run retained
    assert repo.get_latest_run(data_mode="real")["id"] == run_b

    ok = repo.mark_run_stale(run_a, reason="transaction edited")
    conn.commit()
    assert ok
    stale_run = repo.get_run(run_a)
    assert stale_run["is_stale"] == 1
    assert stale_run["stale_reason"] == "transaction edited"
    # the newer run is untouched by marking the older one stale
    assert repo.get_run(run_b)["is_stale"] == 0
