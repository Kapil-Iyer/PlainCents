"""
Deterministic V2 demo dataset generation (TRD §14.1; Build Plan Phase 9).

Pure functions only — no DB access, no repositories, no persistence. Reuses
the *data-generation patterns* from V1's db/seed_synthetic_data.py (merchant
lists per category, amount ranges, seasonal multipliers), rewritten from
scratch for V2's schema: no session_id, every row gets data_mode='demo',
predicted_category is set directly (these are synthetic rows with a known
ground-truth category, not run through CategorizationService), and a
prebuilt 3-month forecast is computed deterministically from the seeded
transactions themselves rather than by fitting pipeline.forecast's model —
see generate_demo_forecast()'s docstring for why.

DemoService is the only caller; it takes these plain dicts and persists them
via the normal V2 repositories inside its own unit-of-work transaction. This
module never opens a connection or imports db/database.py (V1's persistence
layer) — Build Plan Phase 9 explicitly forbids reusing V1's persistence
calls, only its patterns.

Determinism: every random draw here comes from a single seeded
random.Random instance (never the `random` module's global state, so this
never perturbs any other code's randomness), so the same `as_of` date always
produces byte-identical output.
"""
import random
from datetime import date

from backend.config import CATEGORIES
from backend.repositories.money import round_money
from backend.services.dedup import compute_dedup_key
from backend.services.forecast_service import MONTHS_REQUIRED

SEED = 42
MONTHS_OF_HISTORY = 12

# TRD §14.1: reuses the merchant/amount-range/seasonal-multiplier *patterns*
# from db/seed_synthetic_data.py:32-59 (merchant names as sample data are not
# copyrighted content; only the persistence code is off-limits per Build Plan
# Phase 9's "do not reuse legacy V1 db/database.py persistence calls").
MERCHANTS: dict[str, list[str]] = {
    "Food & Dining": ["TIM HORTONS", "MCDONALDS", "SUBWAY", "LOBLAWS", "METRO"],
    "Transport": ["UBER", "PRESTO", "SHELL", "ESSO", "GO TRANSIT"],
    "Rent & Utilities": ["ROGERS", "BELL", "HYDRO ONE", "ENBRIDGE", "TORONTO HYDRO"],
    "Entertainment": ["NETFLIX", "SPOTIFY", "STEAM", "CINEPLEX", "AMAZON PRIME"],
    "Healthcare": ["SHOPPERS", "REXALL", "MAPLE CLINIC", "TELEHEALTH"],
    "Shopping": ["AMAZON", "ZARA", "H&M", "IKEA", "BEST BUY"],
    "Subscriptions": ["ADOBE", "MICROSOFT 365", "ICLOUD", "YOUTUBE PREMIUM"],
    "Other": ["ATM WITHDRAWAL", "MISCELLANEOUS", "BANK FEE"],
}

AMOUNT_RANGES: dict[str, tuple[float, float]] = {
    "Food & Dining": (8, 120),
    "Transport": (5, 80),
    "Rent & Utilities": (80, 180),
    "Entertainment": (10, 60),
    "Healthcare": (15, 90),
    "Shopping": (20, 200),
    "Subscriptions": (10, 20),
    "Other": (20, 100),
}

# Seasonal multiplier by calendar month number (V1 pattern: summer dip,
# December spike; categories with fixed recurring costs are exempted below).
MONTH_MULTIPLIER: dict[int, float] = {
    1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
    6: 0.85, 7: 0.85, 8: 0.85,
    9: 1.0, 10: 1.0, 11: 1.0,
    12: 1.4,
}

_FIXED_COST_CATEGORIES = {"Rent & Utilities", "Subscriptions"}


def _shift_month(year: int, month: int, offset: int) -> tuple[int, int]:
    """Add `offset` calendar months to (year, month), offset may be negative."""
    zero_based = (year * 12 + (month - 1)) + offset
    return zero_based // 12, zero_based % 12 + 1


def _month_range(as_of: date, count: int) -> list[tuple[int, int]]:
    """The `count` calendar months ending at (and including) as_of's month,
    oldest first — guarantees >= MONTHS_OF_HISTORY distinct months so the
    Forecast page is never cold-start immediately after a demo load."""
    return [_shift_month(as_of.year, as_of.month, -offset) for offset in range(count - 1, -1, -1)]


def generate_demo_transactions(as_of: date | None = None) -> list[dict]:
    """Deterministic demo transactions spanning MONTHS_OF_HISTORY calendar
    months across all 8 fixed categories. Every dict is ready for
    TransactionRepository.create() except `data_mode`, which DemoService
    stamps at persistence time. dedup_key is computed here (not left to the
    caller) using the same canonical algorithm as manual/imported
    transactions (backend.services.dedup), with occurrence_index resolved
    against collisions within this same generated batch — mirroring
    IngestionService.parse_and_stage()'s occurrence-counting so a demo
    re-seed can never violate transactions.dedup_key's UNIQUE constraint."""
    rng = random.Random(SEED)
    as_of = as_of or date.today()
    months = _month_range(as_of, MONTHS_OF_HISTORY)

    rows: list[dict] = []
    occurrence_counts: dict[tuple, int] = {}

    for year, month in months:
        month_str = f"{year:04d}-{month:02d}"
        mult = MONTH_MULTIPLIER[month]

        for category in CATEGORIES:
            lo, hi = AMOUNT_RANGES[category]
            cat_mult = 1.0 if category in _FIXED_COST_CATEGORIES else mult

            if category == "Subscriptions":
                # One charge per subscription merchant, every month — a
                # recurring-cost pattern, not a random count.
                merchants = MERCHANTS[category]
            elif category == "Rent & Utilities":
                merchants = rng.sample(MERCHANTS[category], rng.randint(3, 4))
            else:
                n = rng.randint(2, 5)
                merchants = [rng.choice(MERCHANTS[category]) for _ in range(n)]

            for merchant in merchants:
                amount = round_money(rng.uniform(lo, hi) * cat_mult)
                day = rng.randint(1, 28)
                txn_date = f"{month_str}-{day:02d}"

                key_tuple = (txn_date, amount, merchant)
                occurrence_index = occurrence_counts.get(key_tuple, 0)
                occurrence_counts[key_tuple] = occurrence_index + 1
                dedup_key = compute_dedup_key(txn_date, amount, merchant, None, occurrence_index)

                rows.append(
                    {
                        "date": txn_date,
                        "raw_description": None,
                        "merchant": merchant,
                        "amount": amount,
                        "bank_source": None,
                        "predicted_category": category,
                        "confirmed_category": None,
                        "import_batch_id": None,
                        "dedup_key": dedup_key,
                    }
                )

    return rows


def generate_demo_holdings() -> list[dict]:
    """At least one demo holding with a seeded price_cache entry (TRD
    §14.1). Fixed, deterministic values — no randomness needed since these
    aren't derived from anything time-varying."""
    return [
        {"ticker": "AAPL", "shares": 15, "avg_cost": 145.00, "current_price": 178.50},
        {"ticker": "MSFT", "shares": 10, "avg_cost": 280.00, "current_price": 415.20},
        {"ticker": "VTI", "shares": 20, "avg_cost": 210.00, "current_price": 268.75},
    ]


def generate_demo_forecast(transactions: list[dict], as_of: date | None = None) -> dict:
    """A prebuilt forecast_runs + forecast_predictions set (TRD §14.1) so the
    Forecast page is populated immediately after demo load, without the user
    clicking Generate.

    Deliberately does NOT call pipeline.forecast.train_and_predict (the real
    RandomForest fit): the seed's predicted_amount values are each category's
    average monthly total across the seeded history — a plain, auditable
    arithmetic function of the demo data itself, not a fitted model's output.
    This keeps the seed script from "training/refitting the forecasting
    model merely to populate demo mode" (Build Plan Phase 9's explicit
    prohibition) and keeps demo forecast numbers honestly presentation-only:
    they are seeded demonstration data, never ML evaluation evidence.
    `model_impl_version` is tagged "demo_seed_v1" (distinct from the real
    "rf_v1_default_hparams" ForecastService.MODEL_IMPL_VERSION) so this is
    auditable/distinguishable in the data itself, not just in UI copy.
    """
    as_of = as_of or date.today()

    totals: dict[tuple[str, str], float] = {}
    months_seen: set[str] = set()
    for txn in transactions:
        month = txn["date"][:7]
        months_seen.add(month)
        key = (month, txn["predicted_category"])
        totals[key] = totals.get(key, 0.0) + txn["amount"]

    months_available = len(months_seen)

    predictions = []
    for offset in (1, 2, 3):
        year, month = _shift_month(as_of.year, as_of.month, offset)
        forecast_month = f"{year:04d}-{month:02d}"
        for category in CATEGORIES:
            category_month_totals = [
                total for (month_key, cat_key), total in totals.items() if cat_key == category
            ]
            average = sum(category_month_totals) / len(category_month_totals) if category_month_totals else None
            predictions.append(
                {
                    "category": category,
                    "forecast_month": forecast_month,
                    "month_offset": offset,
                    "predicted_amount": round_money(average),
                    "is_available": average is not None,
                    "unavailable_reason": None if average is not None else "insufficient_history",
                }
            )

    run = {
        "months_available": months_available,
        # Mirror the real eligibility gate rather than the seed's own
        # history length, so the demo Forecast page reports the same
        # threshold the app actually enforces.
        "months_required": MONTHS_REQUIRED,
        "model_impl_version": "demo_seed_v1",
    }
    return {"run": run, "predictions": predictions}
