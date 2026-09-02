"""
ForecastService (TRD Section 7.4, Section 12; Build Plan Phase 7).

Replaces the Phase 3 stub. check_status()/get_latest() are DB-only reads —
neither ever touches pipeline.forecast, verified by the "no fit on read"
tests in tests/backend/api/test_forecasts.py. run_forecast() is the only
method that touches a model: it calls pipeline.forecast.aggregate_monthly
(reused as-is) and pipeline.forecast.train_and_predict (ML-D: now the ML-C
selected Naive baseline, no fitting step, previously Random Forest) and
persists the result via ForecastRepository inside one unit-of-work
transaction (TRD Section 4.6). mark_stale(reason) replaces the Phase 3
no-op stub with real persistence — TransactionService/IngestionService's
existing call sites (self._forecast.mark_stale(reason)) need no changes.

data_mode resolution: check_status()/get_latest()/run_forecast() take an
already-resolved `data_mode` (the route resolves app_state.mode via
AppStateService + resolve_data_mode_filter, same pattern as
DashboardService.get_summary()/TransactionService.list() — TRD Section
4.5.1's canonical read mapping is applied once at the route, not
re-resolved per service). mark_stale(reason) is the one exception: its
callers (TransactionService/IngestionService) invoke it as a fire-and-forget
side effect with no mode context of their own, so it resolves the current
mode internally via its own AppStateService, mirroring how those services'
own EMPTY->REAL transition calls resolve mode internally too.
"""
import sqlite3

import pandas as pd

from backend.api.errors import ForecastColdStartError
from backend.db.unit_of_work import transaction as db_transaction
from backend.repositories.forecast_repository import ForecastRepository
from backend.repositories.mode_filter import resolve_data_mode_filter
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.app_state_service import AppStateService
from pipeline.forecast import aggregate_monthly, train_and_predict

# TRD Section 12.5 / PRD Section 21: 12 unique calendar months, matching V1's
# aggregate_monthly() raise threshold exactly.
MONTHS_REQUIRED = 12

# ML Spec Section 18: identifies the forecasting implementation actually
# used to generate a run — not a persisted artifact (Section 18's explicit
# asymmetry with categorization). ML-D Production Integration: the ML-C
# selected candidate is Naive (ml/forecasting/baselines.py::naive_predict),
# selected strategy "N/A" — Naive has no per-horizon/recursive variant, so
# "naive_v1" alone unambiguously identifies both the family and the (only
# possible) strategy; there is no separate strategy field to persist
# (no DB schema change — ML Spec Section 22).
MODEL_IMPL_VERSION = "naive_v1"


class ForecastService:
    def __init__(self, conn: sqlite3.Connection, app_state_service: AppStateService | None = None):
        self._conn = conn
        self._txn_repo = TransactionRepository(conn)
        self._forecast_repo = ForecastRepository(conn)
        self._app_state = app_state_service or AppStateService(conn)

    # -- reads (never fit) ---------------------------------------------------

    def check_status(self, data_mode: str | None) -> dict:
        """DB/aggregation read only (TRD Section 12.2) — never calls into
        pipeline.forecast. Cold start is always HTTP 200 (TRD Section 5.6),
        never an exception."""
        months_available = self._txn_repo.count_distinct_months(data_mode=data_mode)

        if months_available < MONTHS_REQUIRED:
            return {
                "status": "cold_start",
                "months_available": months_available,
                "months_required": MONTHS_REQUIRED,
                "latest_run_id": None,
                "is_stale": None,
            }

        latest = self._forecast_repo.get_latest_run(data_mode=data_mode)
        if latest is None:
            return {
                "status": "no_forecast_yet",
                "months_available": months_available,
                "months_required": MONTHS_REQUIRED,
                "latest_run_id": None,
                "is_stale": None,
            }

        return {
            "status": "ready",
            "months_available": months_available,
            "months_required": MONTHS_REQUIRED,
            "latest_run_id": latest["id"],
            "is_stale": bool(latest["is_stale"]),
        }

    def get_latest(self, data_mode: str | None) -> dict | None:
        """DB read only (TRD Section 5.6) — never trains."""
        run = self._forecast_repo.get_latest_run(data_mode=data_mode)
        if run is None:
            return None
        return self._to_run_response(run)

    def _to_run_response(self, run: dict) -> dict:
        predictions = self._forecast_repo.get_predictions(run["id"])
        return {
            "run_id": run["id"],
            "generated_at": run["generated_at"],
            "is_stale": bool(run["is_stale"]),
            "stale_reason": run["stale_reason"],
            "months_available": run["months_available"],
            "predictions": [
                {
                    "category": p["category"],
                    "forecast_month": p["forecast_month"],
                    "month_offset": p["month_offset"],
                    "predicted_amount": p["predicted_amount"],
                    "is_available": bool(p["is_available"]),
                    "unavailable_reason": p["unavailable_reason"],
                }
                for p in predictions
            ],
        }

    # -- write (the only method that fits) -----------------------------------

    def run_forecast(self, data_mode: str | None) -> dict:
        """The only ForecastService method that touches pipeline.forecast
        (TRD Section 7.4/Section 12.2). Raises ForecastColdStartError (422)
        without persisting anything if months_available < 12 — an explicit
        generation attempt during cold-start is a rejected write, distinct
        from the 200 status read (TRD Section 5.6/Section 15)."""
        months_available = self._txn_repo.count_distinct_months(data_mode=data_mode)

        if months_available < MONTHS_REQUIRED:
            raise ForecastColdStartError(
                f"At least {MONTHS_REQUIRED} months of transaction history are "
                f"required to generate a forecast; {months_available} available.",
                details={
                    "months_available": months_available,
                    "months_required": MONTHS_REQUIRED,
                },
            )

        # Effective-category aggregation (TRD Section 4.1/Section 6, ML Spec
        # Section 10): reads TransactionRepository.list() (Phase 3, reused
        # as-is) which selects from v_transactions_effective, then hands raw
        # per-transaction rows to pipeline.forecast.aggregate_monthly() so
        # V1's month/category grouping logic is reused verbatim rather than
        # re-implemented as a second aggregation rule.
        rows = self._txn_repo.list(data_mode=data_mode)
        raw_df = pd.DataFrame(
            [{"date": r["date"], "amount": r["amount"], "category": r["effective_category"]} for r in rows]
        )
        monthly_df = aggregate_monthly(raw_df)
        forecast_df = train_and_predict(monthly_df)

        predictions = forecast_df.to_dict("records")
        for p in predictions:
            # pandas upcasts a float column containing Python Nones to NaN;
            # normalize back so ForecastRepository/Pydantic see None, not NaN.
            if pd.isna(p.get("predicted_amount")):
                p["predicted_amount"] = None

        with db_transaction(self._conn):
            run_id = self._forecast_repo.create_run(
                {
                    "months_available": months_available,
                    "months_required": MONTHS_REQUIRED,
                    "data_mode": data_mode,
                    "model_impl_version": MODEL_IMPL_VERSION,
                }
            )
            self._forecast_repo.save_predictions(run_id, predictions)
        # Commits here: the run row and every prediction row persist as one
        # unit (TRD Section 4.6), or neither does.

        return self._to_run_response(self._forecast_repo.get_run(run_id))

    # -- staleness (TRD Section 7.2/Section 12.4) -----------------------------

    def mark_stale(self, reason: str) -> None:
        """Flips is_stale on the latest run for the *current* data_mode
        (resolved internally — see the module docstring), and only if it
        isn't already stale: an already-stale run's original stale_reason is
        historical record and is not overwritten (TRD Section 12.4: "only
        the latest non-stale run is marked stale"). A no-op if no run exists
        yet for this mode."""
        data_mode = resolve_data_mode_filter(self._app_state.get_mode())
        latest = self._forecast_repo.get_latest_run(data_mode=data_mode)
        if latest is None or latest["is_stale"]:
            return
        self._forecast_repo.mark_run_stale(latest["id"], reason=reason)
        self._conn.commit()
