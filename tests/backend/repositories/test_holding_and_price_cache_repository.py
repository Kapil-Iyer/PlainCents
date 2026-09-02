"""Phase 1 tests: HoldingRepository, PriceCacheRepository. Requirement #12."""
from backend.repositories.holding_repository import HoldingRepository
from backend.repositories.price_cache_repository import PriceCacheRepository


def test_holding_crud_round_trip(conn):
    repo = HoldingRepository(conn)
    hid = repo.create({"ticker": "AAPL", "shares": 10, "avg_cost": 150.5, "data_mode": "real"})
    conn.commit()

    fetched = repo.get(hid)
    assert fetched["ticker"] == "AAPL"
    assert fetched["avg_cost"] == 150.5

    repo.update(hid, {"shares": 15})
    conn.commit()
    assert repo.get(hid)["shares"] == 15

    assert repo.delete(hid)
    conn.commit()
    assert repo.get(hid) is None


def test_holding_list_mode_filter(conn):
    repo = HoldingRepository(conn)
    repo.create({"ticker": "AAPL", "shares": 10, "avg_cost": 150.0, "data_mode": "demo"})
    repo.create({"ticker": "MSFT", "shares": 5, "avg_cost": 300.0, "data_mode": "real"})
    conn.commit()

    assert len(repo.list(data_mode=None)) == 2
    assert len(repo.list(data_mode="demo")) == 1
    assert len(repo.list(data_mode="real")) == 1


def test_price_cache_upsert_and_get(conn):
    repo = PriceCacheRepository(conn)
    assert repo.get_last_known("AAPL") is None

    repo.upsert_latest("AAPL", 178.503, "2026-09-01T12:00:00")
    conn.commit()
    row = repo.get_last_known("AAPL")
    assert row["current_price"] == 178.5

    repo.upsert_latest("AAPL", 180.0, "2026-09-01T13:00:00")  # UPSERT, not a second row
    conn.commit()
    row = repo.get_last_known("AAPL")
    assert row["current_price"] == 180.0
