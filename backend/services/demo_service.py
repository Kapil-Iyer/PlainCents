"""
DemoService (TRD §4.5, §7.7, §14; Build Plan Phase 9).

Owns the demo/real mutual-exclusion state machine on top of app_state.mode:
load_demo() seeds a full demo dataset (all rows data_mode='demo') and flips
EMPTY -> DEMO; clear_demo() deletes every demo-flagged row and flips back to
EMPTY. Both run inside one unit-of-work transaction each — a failure midway
leaves no partial demo rows and no incorrect mode flip (TRD §4.6's same
atomicity guarantee TransactionService/IngestionService/PortfolioService
already follow for their own durable writes).

Deterministic seed generation itself lives in backend.services.demo_seed_data
(pure functions, no DB access) — this service only persists what that module
produces, via the normal V2 repositories, and never shells out to
backend/scripts/seed_v2_demo_data.py (that script is a thin CLI wrapper
around this same service, not the other way around).
"""
import sqlite3

from backend.api.errors import DemoConflictError
from backend.db.unit_of_work import transaction as db_transaction
from backend.repositories.app_state_repository import AppStateRepository
from backend.repositories.forecast_repository import ForecastRepository
from backend.repositories.holding_repository import HoldingRepository
from backend.repositories.price_cache_repository import PriceCacheRepository
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.demo_seed_data import (
    DEMO_PRICE_FETCHED_AT,
    generate_demo_forecast,
    generate_demo_holdings,
    generate_demo_transactions,
)


class DemoService:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._app_state_repo = AppStateRepository(conn)
        self._txn_repo = TransactionRepository(conn)
        self._holding_repo = HoldingRepository(conn)
        self._price_cache_repo = PriceCacheRepository(conn)
        self._forecast_repo = ForecastRepository(conn)

    def status(self) -> dict:
        mode = self._app_state_repo.get_mode()
        return {"mode": mode, "can_load_demo": mode == "EMPTY"}

    def load_demo(self) -> dict:
        # TRD §4.5 Mechanics: "Demo load: allowed only when app_state.mode =
        # 'EMPTY'." — REAL is the case §5.2/§14.2 spell out explicitly (409,
        # never a force option that deletes real data); DEMO isn't separately
        # named there, but "allowed only when EMPTY" already covers it: a
        # second load while already DEMO is rejected the same way, rather
        # than silently reseeding on top of (or duplicating) existing demo
        # rows, since neither behavior is documented and TRD §4.5.1's
        # dedup_key UNIQUE constraint makes reseeding unsafe to invent.
        mode = self._app_state_repo.get_mode()
        if mode == "REAL":
            raise DemoConflictError(
                "Cannot load demo data while real data exists. Real data is never "
                "deleted to make room for demo data."
            )
        if mode == "DEMO":
            raise DemoConflictError("Demo data is already loaded.")

        transactions = generate_demo_transactions()
        holdings = generate_demo_holdings()
        forecast = generate_demo_forecast(transactions)

        with db_transaction(self._conn):
            for txn in transactions:
                self._txn_repo.create({**txn, "data_mode": "demo"})

            for holding in holdings:
                self._holding_repo.create(
                    {
                        "ticker": holding["ticker"],
                        "shares": holding["shares"],
                        "avg_cost": holding["avg_cost"],
                        "data_mode": "demo",
                    }
                )
                # price_cache has no data_mode (TRD §4.5/§4.8) — this upsert
                # participates in the same transaction/rollback as everything
                # else here even though the table itself is mode-agnostic.
                self._price_cache_repo.upsert_latest(
                    holding["ticker"], holding["current_price"], DEMO_PRICE_FETCHED_AT
                )

            run_id = self._forecast_repo.create_run({**forecast["run"], "data_mode": "demo"})
            self._forecast_repo.save_predictions(run_id, forecast["predictions"])

            self._app_state_repo.set_mode("DEMO")
        # Commits here: every seeded row plus the mode flip succeed together,
        # or none of them do — no partial demo seed / falsely-DEMO mode.

        return {
            "mode": "DEMO",
            "summary": {
                "transactions": len(transactions),
                "holdings": len(holdings),
                "forecast_predictions": len(forecast["predictions"]),
            },
        }

    def clear_real_data(self) -> dict:
        """Mirror image of clear_demo(): deletes every data_mode='real' row
        (transactions, holdings, forecast runs) and, symmetrically, any
        price_cache row exclusively used by a real holding (a ticker also
        held for demo is left alone), then flips the mode back to 'EMPTY' so
        Load Demo Data becomes available again (DemoService.load_demo()
        rejects with 409 while mode == 'REAL').

        This is the in-app, user-facing equivalent of the developer-only
        scripts/reset_real_data.py maintenance script (same deletion scope),
        now reachable without shell/terminal access -- e.g. once deployed,
        where a developer may not have a shell on the running instance. The
        script itself is untouched and still works identically for local
        development.

        Idempotent (TRD §5.2's clear_demo() convention, mirrored): calling
        this while there is no real data yet is a no-op, not an error --
        deletion is scoped entirely by data_mode='real', so nothing to
        delete just means zero rows removed.
        """
        mode = self._app_state_repo.get_mode()

        # Defense in depth, mirroring clear_demo()'s own note: real-creation
        # paths only ever run while mode != 'DEMO', so data_mode='real' rows
        # should never coexist with mode == 'DEMO'. If this is ever reached
        # in that state anyway, the deletion below is still safe (it only
        # ever touches data_mode='real' rows, and there should be none) --
        # but the mode flip to 'EMPTY' is skipped, since forcing 'EMPTY'
        # while demo data is active would misrepresent demo data as absent.
        target_mode = "EMPTY" if mode != "DEMO" else "DEMO"

        real_holdings = self._holding_repo.list(data_mode="real")
        demo_holdings = self._holding_repo.list(data_mode="demo")
        demo_tickers = {h["ticker"] for h in demo_holdings}
        real_only_tickers = {h["ticker"] for h in real_holdings} - demo_tickers

        with db_transaction(self._conn):
            transactions_deleted = self._txn_repo.delete_by_data_mode("real")
            holdings_deleted = self._holding_repo.delete_by_data_mode("real")
            forecast_runs_deleted = self._forecast_repo.delete_runs_by_data_mode("real")
            for ticker in real_only_tickers:
                self._price_cache_repo.delete(ticker)

            self._app_state_repo.set_mode(target_mode)
        # Commits here: every real-flagged deletion plus the mode flip
        # succeed together, or none of them do.

        return {
            "mode": target_mode,
            "cleared": True,
            "summary": {
                "transactions": transactions_deleted,
                "holdings": holdings_deleted,
                "forecast_runs": forecast_runs_deleted,
                "price_cache": len(real_only_tickers),
            },
        }

    def clear_demo(self) -> dict:
        # TRD §5.2: "200 on success (idempotent: 200 even if already
        # empty)". No mode check gates entry — deletion is scoped entirely by
        # data_mode='demo', so calling this while already EMPTY (no demo rows
        # exist) is naturally a no-op, not a special-cased branch.
        mode = self._app_state_repo.get_mode()

        # Defense in depth (TRD §4.5's dual data_mode + app_state.mode
        # isolation): under this app's own invariants, real-creation paths
        # (TransactionService/IngestionService/PortfolioService) only ever
        # run while mode != 'DEMO', so data_mode='demo' rows should never
        # coexist with mode == 'REAL'. If this is ever reached in that state
        # anyway, the deletion below is still safe (it only ever touches
        # data_mode='demo' rows, and there should be none) — but the mode
        # flip to 'EMPTY' is skipped, since forcing 'EMPTY' while real rows
        # exist would misrepresent real data as absent. This is not a
        # currently-reachable path via the normal UI/API flow; it exists
        # purely so a caller bug can never turn "delete demo leftovers" into
        # "hide real data."
        target_mode = "EMPTY" if mode != "REAL" else "REAL"

        demo_holdings = self._holding_repo.list(data_mode="demo")
        real_holdings = self._holding_repo.list(data_mode="real")
        real_tickers = {h["ticker"] for h in real_holdings}
        # TRD §14.3: "any price_cache rows exclusively used by demo holdings
        # (a ticker also held for real is left alone — price data isn't
        # demo/real-specific)."
        demo_only_tickers = {h["ticker"] for h in demo_holdings} - real_tickers

        with db_transaction(self._conn):
            transactions_deleted = self._txn_repo.delete_by_data_mode("demo")
            holdings_deleted = self._holding_repo.delete_by_data_mode("demo")
            forecast_runs_deleted = self._forecast_repo.delete_runs_by_data_mode("demo")
            for ticker in demo_only_tickers:
                self._price_cache_repo.delete(ticker)

            self._app_state_repo.set_mode(target_mode)
        # Commits here: every demo-flagged deletion plus the mode flip
        # succeed together, or none of them do — no partial demo remnants.

        return {
            "mode": target_mode,
            "cleared": True,
            "summary": {
                "transactions": transactions_deleted,
                "holdings": holdings_deleted,
                "forecast_runs": forecast_runs_deleted,
                "price_cache": len(demo_only_tickers),
            },
        }
