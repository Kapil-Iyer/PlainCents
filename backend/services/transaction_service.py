"""
TransactionService (TRD §7.2; Build Plan Phase 3).

Owns the business rules TransactionRepository deliberately does not:
categorization-before-persistence, dedup_key computation, the EMPTY->REAL
mode transition, and the forecast-staleness mutation table.
"""
from __future__ import annotations

import sqlite3

from backend.api.errors import BadRequestError, NotFoundError
from backend.db.unit_of_work import transaction as db_transaction
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.app_state_service import AppStateService
from backend.services.categorization_service import CategorizationService
from backend.services.dedup import compute_dedup_key
from backend.services.forecast_service import ForecastService

# TRD §7.2 staleness-mutation table: which TransactionUpdate fields mark the
# latest forecast run stale. Merchant/raw_description edits do not (merchant
# text doesn't feed forecast aggregation, and TRD explicitly rules out
# re-categorizing on a merchant-only edit).
_STALE_TRIGGERING_FIELDS = {"date", "amount", "confirmed_category"}


class TransactionService:
    def __init__(
        self,
        conn: sqlite3.Connection,
        categorization_service: CategorizationService,
        app_state_service: AppStateService | None = None,
        forecast_service: ForecastService | None = None,
    ):
        self._conn = conn
        self._repo = TransactionRepository(conn)
        self._categorization = categorization_service
        self._app_state = app_state_service or AppStateService(conn)
        self._forecast = forecast_service or ForecastService(conn)

    def _compute_dedup_key(self, date: str, merchant: str, amount: float, bank_source: str | None) -> str:
        # TRD §4.4: occurrence_index is the 0-based position among rows
        # sharing the same (date, amount, merchant, bank_source) tuple
        # already in the table. Probed via the repository's existing
        # exists_by_dedup_key() rather than a new repository method, since
        # TransactionRepository is Phase 1 scope and not otherwise touched
        # in Phase 3. The key format itself is centralized in
        # backend.services.dedup so Phase 4's bulk import uses the same
        # canonical order rather than a second implementation.
        occurrence_index = 0
        while True:
            key = compute_dedup_key(date, amount, merchant, bank_source, occurrence_index)
            if not self._repo.exists_by_dedup_key(key):
                return key
            occurrence_index += 1

    def create_manual(self, data: dict) -> dict:
        # Constraint #6 / TRD §11.3: categorization must succeed before any
        # row is written. CategorizationService.predict() raises
        # CategorizationUnavailableError (-> 503) if the model is
        # missing/errored; that propagates and aborts here, before persist.
        prediction = self._categorization.predict(
            {"merchant": data["merchant"], "amount": data["amount"], "date": data["date"]}
        )

        bank_source = None  # manual creation has no bank source (TRD §4.1)
        dedup_key = self._compute_dedup_key(data["date"], data["merchant"], data["amount"], bank_source)

        # TRD §4.5.1: "the transaction-insert step and the mode-transition
        # check happen together as the real work of the request" — both must
        # commit as one durable unit (Phase 1's unit_of_work helper), or
        # neither does. Without this, a bare INSERT-then-commit followed by a
        # separate, never-committed mode UPDATE leaves app_state.mode visible
        # as REAL only on the same live connection and lost on reconnect —
        # the row itself would still be durably real, but the mode transition
        # would silently roll back on the next connection open.
        with db_transaction(self._conn):
            transaction_id = self._repo.create(
                {
                    "date": data["date"],
                    "merchant": data["merchant"],
                    "amount": data["amount"],
                    "raw_description": data.get("raw_description"),
                    "bank_source": bank_source,
                    "predicted_category": prediction["predicted_category"],
                    "confirmed_category": data.get("confirmed_category"),
                    "import_batch_id": None,
                    "data_mode": "real",
                    "dedup_key": dedup_key,
                }
            )
            self._app_state.maybe_transition_to_real()
        # Commits here (both the insert and the mode transition together);
        # if maybe_transition_to_real() raised, unit_of_work rolls back the
        # insert too, so a transaction is never left committed while the
        # mode transition it required silently failed.

        # TRD §7.2: manual creation always marks the latest forecast stale.
        # This is deliberately outside the unit-of-work block above — it is
        # the downstream/optional action TRD §4.5.1 describes, whose failure
        # must not undo the already-durable insert + mode transition.
        self._forecast.mark_stale("transaction_created")

        return self._repo.get(transaction_id)

    def get(self, transaction_id: int) -> dict:
        row = self._repo.get(transaction_id)
        if row is None:
            raise NotFoundError(f"Transaction {transaction_id} not found.")
        return row

    def list(
        self,
        data_mode: str | None,
        date_from: str | None = None,
        date_to: str | None = None,
        category: str | None = None,
        search: str | None = None,
        sort: str = "date",
        page: int = 1,
        page_size: int = 50,
    ) -> dict:
        all_matching = self._repo.list(
            data_mode=data_mode,
            date_from=date_from,
            date_to=date_to,
            category=category,
            search=search,
            sort=sort,
        )
        total = len(all_matching)
        start = (page - 1) * page_size
        items = all_matching[start : start + page_size]
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    def update(self, transaction_id: int, fields: dict) -> dict:
        if not fields:
            raise BadRequestError("At least one field must be provided to update.")

        # Existence check first so a PATCH on a missing id is a clean 404,
        # not a silent no-op (TransactionRepository.update() only reports
        # rowcount, it doesn't distinguish "not found" from "no-op").
        self.get(transaction_id)

        updated = self._repo.update(transaction_id, fields)
        if not updated:
            raise NotFoundError(f"Transaction {transaction_id} not found.")
        self._conn.commit()

        if _STALE_TRIGGERING_FIELDS & fields.keys():
            self._forecast.mark_stale("transaction_updated")

        return self._repo.get(transaction_id)

    def delete(self, transaction_id: int) -> None:
        self.get(transaction_id)
        deleted = self._repo.delete(transaction_id)
        if not deleted:
            raise NotFoundError(f"Transaction {transaction_id} not found.")
        self._conn.commit()
        self._forecast.mark_stale("transaction_deleted")
