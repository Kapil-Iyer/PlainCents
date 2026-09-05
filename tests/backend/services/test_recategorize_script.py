"""
The stored-transaction re-categorization maintenance script.

The guarantee worth testing here is a negative one: this script may rewrite
the system's own decision, and must never touch the user's. A bug in that
direction would silently destroy corrections a person actually made, and no
other test in the suite would notice.
"""
from pathlib import Path

import pytest

from backend.repositories.transaction_repository import TransactionRepository
from scripts import recategorize_stored_transactions as script

FIXTURE_MODEL = (
    Path(__file__).resolve().parent.parent.parent / "fixtures" / "categorizer_model_test.pkl"
)


@pytest.fixture(autouse=True)
def use_fixture_model(monkeypatch):
    monkeypatch.setattr(script, "CATEGORIZER_MODEL_PATH", FIXTURE_MODEL)


def _seed(conn):
    repo = TransactionRepository(conn)
    ids = {}
    ids["stale"] = repo.create({
        "date": "2026-01-05",
        "merchant": "VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY",
        "amount": 34.82,
        "bank_source": "RBC",
        # Deliberately wrong, as though written by an older model.
        "predicted_category": "Food & Dining",
        "confirmed_category": None,
        "data_mode": "real",
        "dedup_key": "a",
    })
    ids["corrected"] = repo.create({
        "date": "2026-01-06",
        "merchant": "VISA DEBIT PURCHASE - 9137 CAREWELL PHARMACY",
        "amount": 12.00,
        "bank_source": "RBC",
        "predicted_category": "Food & Dining",
        # The user disagreed with the system here.
        "confirmed_category": "Shopping",
        "data_mode": "real",
        "dedup_key": "b",
    })
    ids["ambiguous"] = repo.create({
        "date": "2026-01-07",
        "merchant": "E-TRANSFER SENT",
        "amount": 250.00,
        "bank_source": "RBC",
        "predicted_category": "Entertainment",
        "confirmed_category": None,
        "data_mode": "real",
        "dedup_key": "c",
    })
    conn.commit()
    return ids


def _run(db_path: Path, monkeypatch, *extra: str):
    monkeypatch.setattr("sys.argv", ["recategorize", "--db", str(db_path), *extra])
    script.main()


def test_dry_run_writes_nothing(conn, db_path, monkeypatch, capsys):
    ids = _seed(conn)
    repo = TransactionRepository(conn)
    before = {k: repo.get(v)["predicted_category"] for k, v in ids.items()}

    _run(db_path, monkeypatch)

    after = {k: repo.get(v)["predicted_category"] for k, v in ids.items()}
    assert after == before
    assert "Dry run" in capsys.readouterr().out


def test_apply_refreshes_the_system_decision(conn, db_path, monkeypatch):
    ids = _seed(conn)
    repo = TransactionRepository(conn)

    _run(db_path, monkeypatch, "--apply")

    # A merchant the current model can read is no longer mis-filed.
    assert repo.get(ids["stale"])["predicted_category"] == "Healthcare"
    # A description naming nothing is routed to Other by the structural rule.
    assert repo.get(ids["ambiguous"])["predicted_category"] == "Other"


def test_apply_never_touches_a_user_correction(conn, db_path, monkeypatch):
    """The guarantee this script lives or dies by."""
    ids = _seed(conn)
    repo = TransactionRepository(conn)

    _run(db_path, monkeypatch, "--apply")

    corrected = repo.get(ids["corrected"])
    assert corrected["confirmed_category"] == "Shopping"
    # And the category the user actually sees is still theirs, even though
    # the system's own view underneath it moved.
    assert corrected["effective_category"] == "Shopping"
    assert corrected["predicted_category"] == "Healthcare"
    assert corrected["is_manual_override"] == 1


def test_apply_backfills_the_correction_memory_key(conn, db_path, monkeypatch):
    ids = _seed(conn)

    _run(db_path, monkeypatch, "--apply")

    keys = {
        name: conn.execute(
            "SELECT merchant_key FROM transactions WHERE id = ?", (tid,)
        ).fetchone()[0]
        for name, tid in ids.items()
    }
    # Two card-suffix variants of one pharmacy share an identity...
    assert keys["stale"] == keys["corrected"] is not None
    # ...and a generic transfer gets none at all, so unrelated transfers can
    # never collapse into one memory entry.
    assert keys["ambiguous"] is None


def test_apply_marks_an_existing_forecast_stale(conn, db_path, monkeypatch):
    """Category totals feed the forecast, so a re-categorization invalidates
    any run built on the old ones."""
    from backend.repositories.forecast_repository import ForecastRepository

    _seed(conn)
    frepo = ForecastRepository(conn)
    run_id = frepo.create_run({"months_available": 6, "data_mode": "real",
                               "model_impl_version": "rolling_mean_3_v1"})
    conn.commit()

    _run(db_path, monkeypatch, "--apply")

    assert frepo.get_run(run_id)["is_stale"] == 1


def test_data_mode_filter_leaves_other_modes_alone(conn, db_path, monkeypatch):
    repo = TransactionRepository(conn)
    demo_id = repo.create({
        "date": "2026-01-05",
        "merchant": "VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY",
        "amount": 10.0,
        "bank_source": "RBC",
        "predicted_category": "Food & Dining",
        "data_mode": "demo",
        "dedup_key": "demo-a",
    })
    real_id = repo.create({
        "date": "2026-01-05",
        "merchant": "VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY",
        "amount": 10.0,
        "bank_source": "RBC",
        "predicted_category": "Food & Dining",
        "data_mode": "real",
        "dedup_key": "real-a",
    })
    conn.commit()

    _run(db_path, monkeypatch, "--apply", "--data-mode", "real")

    assert repo.get(real_id)["predicted_category"] == "Healthcare"
    assert repo.get(demo_id)["predicted_category"] == "Food & Dining"
