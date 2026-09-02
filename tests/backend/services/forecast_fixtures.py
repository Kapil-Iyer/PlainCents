"""
Deterministic synthetic-transaction builders for Phase 7 forecast tests
(TRD Section 12.5 cold-start / per-category availability). Chosen over
committing a large opaque CSV fixture solely to satisfy the 12-month gate,
per the Build Plan's explicit guidance to prefer deterministic test
builders/fixtures under tests/. Not a test module itself (no test_ prefix),
imported by tests/backend/services/test_forecast_service.py and
tests/backend/api/test_forecasts.py.
"""
from backend.repositories.transaction_repository import TransactionRepository


def _shift_month(start: str, offset: int) -> str:
    year, month = (int(x) for x in start.split("-"))
    total = (month - 1) + offset
    return f"{year + total // 12:04d}-{total % 12 + 1:02d}"


def seed_months(
    conn,
    months: int,
    categories: list[str],
    *,
    start: str = "2025-01",
    amount: float = 100.0,
    data_mode: str = "real",
) -> list[int]:
    """One transaction per (month, category) for `months` consecutive
    calendar months starting at `start` ("YYYY-MM") — every listed category
    gets exactly `months` monthly data points, comfortably surviving
    build_forecast_features's 7-occurrence rolling/lag floor when
    months >= 7."""
    repo = TransactionRepository(conn)
    ids = []
    for i in range(months):
        month_str = _shift_month(start, i)
        for j, cat in enumerate(categories):
            txn_id = repo.create(
                {
                    "date": f"{month_str}-10",
                    "merchant": f"MERCHANT {cat}",
                    "amount": amount + j,
                    "predicted_category": cat,
                    "confirmed_category": None,
                    "data_mode": data_mode,
                    "dedup_key": f"seed|{month_str}|{cat}|{data_mode}",
                }
            )
            ids.append(txn_id)
    conn.commit()
    return ids


def seed_sparse_category(
    conn,
    category: str,
    month_strs: list[str],
    *,
    amount: float = 50.0,
    data_mode: str = "real",
) -> list[int]:
    """One transaction for `category` in each of the given ("YYYY-MM")
    months only — for constructing a category with too little history to
    survive build_forecast_features's rolling/lag dropna (fewer than 7
    monthly data points)."""
    repo = TransactionRepository(conn)
    ids = []
    for month_str in month_strs:
        txn_id = repo.create(
            {
                "date": f"{month_str}-15",
                "merchant": f"SPARSE {category}",
                "amount": amount,
                "predicted_category": category,
                "confirmed_category": None,
                "data_mode": data_mode,
                "dedup_key": f"sparse|{month_str}|{category}|{data_mode}",
            }
        )
        ids.append(txn_id)
    conn.commit()
    return ids
