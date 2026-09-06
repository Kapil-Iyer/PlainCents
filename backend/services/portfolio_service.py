"""
PortfolioService (TRD §7.5, §13; Build Plan Phase 8).

Owns the read/refresh separation TRD §13.2 requires: get_holdings_with_prices()
is a pure DB join (HoldingRepository + PriceCacheRepository, both persistence-
only per TRD §8) and never calls pipeline.portfolio.fetch_price(); only
refresh_prices() does, since it is the one endpoint the TRD authorizes to
reach yfinance (§5.7, §13.2). CRUD delegates directly to HoldingRepository,
except create_holding, which also performs the EMPTY -> REAL transition check
in the same unit of work as the insert, mirroring TransactionService.create_manual()
(TRD §4.5.1).
"""
import sqlite3

from backend.api.errors import BadRequestError, NotFoundError
from backend.db.unit_of_work import transaction as db_transaction
from backend.repositories.holding_repository import HoldingRepository
from backend.repositories.price_cache_repository import PriceCacheRepository
from backend.services.app_state_service import AppStateService
from backend.services.demo_seed_data import DEMO_PRICE_FETCHED_AT
from pipeline.portfolio import fetch_price


class PortfolioService:
    def __init__(self, conn: sqlite3.Connection, app_state_service: AppStateService | None = None):
        self._conn = conn
        self._holdings = HoldingRepository(conn)
        self._price_cache = PriceCacheRepository(conn)
        self._app_state = app_state_service or AppStateService(conn)

    def _to_response(self, holding: dict) -> dict:
        # TRD §13.2/§13.3: last-known cached price only, no network call, and
        # displayed regardless of its fetched_at age.
        cache = self._price_cache.get_last_known(holding["ticker"])
        current_price = cache["current_price"] if cache else None
        price_last_updated = cache["fetched_at"] if cache else None

        if current_price is not None:
            current_value = holding["shares"] * current_price
            pnl = holding["shares"] * (current_price - holding["avg_cost"])
        else:
            # TRD §6/Build Plan Phase 8 financial rules: never fabricate a
            # value/P&L when there is no price to compute it from.
            current_value = None
            pnl = None

        # DEMO_PRICE_FETCHED_AT is a fixed sentinel stamped ONLY by
        # DemoService.load_demo() (never by a genuine fetch_price() call,
        # which always stamps datetime.now().isoformat()) -- so this
        # equality check is an exact, no-false-positive way to tell "this
        # price was never actually fetched, it's synthetic demo data" apart
        # from a real cached price, however old. Once a demo holding is
        # refreshed via POST /api/holdings/refresh-prices, its fetched_at
        # becomes a genuine timestamp and this stops matching -- the flag
        # naturally flips from "demo snapshot" to "fresh" with no extra
        # bookkeeping (Build Plan Phase 8 price-state honesty rule).
        price_is_demo_snapshot = price_last_updated == DEMO_PRICE_FETCHED_AT

        return {
            "id": holding["id"],
            "ticker": holding["ticker"],
            "shares": holding["shares"],
            "avg_cost": holding["avg_cost"],
            "current_price": current_price,
            "current_value": current_value,
            "pnl": pnl,
            "price_last_updated": price_last_updated,
            "price_is_demo_snapshot": price_is_demo_snapshot,
            "created_at": holding["created_at"],
            "updated_at": holding["updated_at"],
        }

    def get_holdings_with_prices(self, data_mode: str | None) -> list[dict]:
        rows = self._holdings.list(data_mode=data_mode)
        return [self._to_response(row) for row in rows]

    def get_holding(self, holding_id: int) -> dict:
        row = self._holdings.get(holding_id)
        if row is None:
            raise NotFoundError(f"Holding {holding_id} not found.")
        return self._to_response(row)

    def create_holding(self, data: dict) -> dict:
        # TRD §4.5.1 EMPTY -> REAL: the insert and the mode-transition check
        # must commit as one durable unit, same reasoning as
        # TransactionService.create_manual() — see that module's comment.
        ticker = data["ticker"].strip().upper()
        with db_transaction(self._conn):
            holding_id = self._holdings.create(
                {
                    "ticker": ticker,
                    "shares": data["shares"],
                    "avg_cost": data["avg_cost"],
                    "data_mode": "real",
                }
            )
            self._app_state.maybe_transition_to_real()
        return self._to_response(self._holdings.get(holding_id))

    def update_holding(self, holding_id: int, fields: dict) -> dict:
        if not fields:
            raise BadRequestError("At least one field must be provided to update.")

        # Existence check first so a PATCH on a missing id is a clean 404,
        # not a silent no-op (HoldingRepository.update() only reports
        # rowcount, it doesn't distinguish "not found" from "no-op").
        self.get_holding(holding_id)

        updated = self._holdings.update(holding_id, fields)
        if not updated:
            raise NotFoundError(f"Holding {holding_id} not found.")
        self._conn.commit()
        return self._to_response(self._holdings.get(holding_id))

    def delete_holding(self, holding_id: int) -> None:
        self.get_holding(holding_id)
        deleted = self._holdings.delete(holding_id)
        if not deleted:
            raise NotFoundError(f"Holding {holding_id} not found.")
        self._conn.commit()

    def refresh_prices(self, data_mode: str | None) -> dict:
        # TRD §13.4: iterate tickers independently; a per-ticker
        # fetch_price() failure (it never raises, always returns None on
        # error) is recorded in `failed` without aborting the others.
        # Refresh is scoped to the currently-visible (mode-filtered)
        # holdings, matching what GET /api/holdings itself would show.
        holdings = self._holdings.list(data_mode=data_mode)
        tickers = sorted({h["ticker"] for h in holdings})

        refreshed = []
        failed = []
        for ticker in tickers:
            price = fetch_price(self._conn, ticker)
            if price is None:
                failed.append({"ticker": ticker, "error": "price_fetch_failed"})
            else:
                refreshed.append({"ticker": ticker, "price": price})
        return {"refreshed": refreshed, "failed": failed}
