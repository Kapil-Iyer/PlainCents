"""
IngestionService (TRD §7.1, §10; Build Plan Phase 4).

Coordinates the full TD import pipeline: parse bytes -> validate -> dedup
check -> CategorizationService.predict_batch() (reused from Phase 3, never a
second categorization path, Build Plan §2.1) -> stage -> (on confirm)
re-validate live -> persist -> mark forecast stale (downstream/optional).

Never imports/calls pipeline.forecast at all (TRD §7.1).
"""
import sqlite3

from backend.api.errors import (
    BadRequestError,
    CategorizationUnavailableError,
    ConflictError,
    DemoConflictError,
    NotFoundError,
)
from backend.db.unit_of_work import transaction as db_transaction
from backend.repositories.import_batch_repository import ImportBatchRepository
from backend.repositories.money import round_money
from backend.repositories.staged_transaction_repository import StagedTransactionRepository
from backend.repositories.transaction_repository import TransactionRepository
from backend.services.app_state_service import AppStateService
from backend.services.categorization_service import CategorizationService
from backend.services.category_decision import CorrectionMemory, decide_batch
from backend.services.dedup import compute_dedup_key
from backend.services.forecast_service import ForecastService
from pipeline.ingest import load_and_clean_from_bytes


class IngestionService:
    def __init__(
        self,
        conn: sqlite3.Connection,
        categorization_service: CategorizationService,
        app_state_service: AppStateService | None = None,
        forecast_service: ForecastService | None = None,
    ):
        self._conn = conn
        self._txn_repo = TransactionRepository(conn)
        self._batch_repo = ImportBatchRepository(conn)
        self._staged_repo = StagedTransactionRepository(conn)
        self._categorization = categorization_service
        self._app_state = app_state_service or AppStateService(conn)
        self._forecast = forecast_service or ForecastService(conn)

    # -- preview -----------------------------------------------------------

    def parse_and_stage(
        self, file_bytes: bytes, bank: str | None = None, original_filename: str | None = None
    ) -> dict:
        # Build Plan Phase 4, item 5 / TRD §5.3: the demo-conflict check only
        # reads app_state.mode, so it's functional now even though
        # POST /api/demo/load itself isn't implemented until Phase 9.
        if self._app_state.get_mode() == "DEMO":
            raise DemoConflictError(
                "Cannot import real data while demo data is loaded. "
                "Clear demo data first.",
            )

        try:
            df, meta = load_and_clean_from_bytes(file_bytes, bank=bank)
        except ValueError as exc:
            # Whole-file failure (unrecognized columns, an explicit-bank
            # mismatch, or a Phase-12A.5-BLOCKED bank) -> 400, not a
            # degraded 200 (TRD §10).
            raise BadRequestError(str(exc)) from exc
        except Exception as exc:
            # Genuinely non-CSV bytes (pandas parser errors, etc.) -> 400.
            raise BadRequestError("Uploaded file could not be parsed as CSV.") from exc

        # Phase 12A.5/12B: bank_source is always the RESOLVED bank (the
        # parser's own bank_detected -- whichever fingerprint actually
        # matched, whether the caller passed an explicit bank or None for
        # auto-detect), never the raw (possibly None/"Auto") input param.
        resolved_bank = meta["bank_detected"]

        batch_id = self._batch_repo.create_preview(
            bank_source=resolved_bank, original_filename=original_filename, data_mode="real"
        )

        rows = df.to_dict("records")

        # ML-G: ONE decision path (backend/services/category_decision.py),
        # run here and persisted whole, so what Preview shows is what Confirm
        # stores. Previously Preview staged the raw model output and Confirm
        # separately applied structural-ambiguity routing and correction
        # memory, so the two disagreed on exactly the rows where it mattered.
        #
        # Correction memory is consulted at preview READ-ONLY: decide_batch
        # only queries the transactions table, it never writes. Nothing about
        # this preview mutates transactions, correction memory, or app mode.
        #
        # TRD §10 unchanged: if the model is unavailable, preview still
        # succeeds (200), reports categorization_available=False, and leaves
        # every staged predicted_category NULL rather than fabricating one.
        categorization_available = True
        decisions = []
        memory = CorrectionMemory(self._txn_repo)
        try:
            decisions = decide_batch(
                [(r["merchant"], resolved_bank) for r in rows],
                self._categorization,
                memory,
            )
        except CategorizationUnavailableError:
            categorization_available = False

        # TRD §4.4: occurrence_index = 0-based position among rows sharing
        # (date, amount, merchant, bank_source), in file order, within this
        # batch. Whether a given (date, amount, merchant, bank, index) key
        # is a *duplicate* is decided by checking it against the live
        # transactions table (not a second, separate staging-only rule).
        occurrence_counts: dict[tuple, int] = {}
        staged_rows = []
        rows_duplicate = 0

        for i, row in enumerate(rows):
            # Phase 12B dedup fix: the parser no longer collapses intra-file
            # duplicates (pipeline.ingest's drop_duplicates() calls were
            # removed from the V2 bytes path -- Phase 12A finding), so two
            # genuinely identical source rows now both reach this loop and
            # this counter correctly assigns them occurrence_index 0 and 1
            # instead of one of them having already vanished upstream.
            key_tuple = (row["date"], round_money(row["amount"]), row["merchant"])
            occurrence_index = occurrence_counts.get(key_tuple, 0)
            occurrence_counts[key_tuple] = occurrence_index + 1

            dedup_key = compute_dedup_key(
                row["date"], row["amount"], row["merchant"], resolved_bank, occurrence_index
            )
            is_duplicate = self._txn_repo.exists_by_dedup_key(dedup_key)
            if is_duplicate:
                rows_duplicate += 1

            decision = decisions[i] if categorization_available else None

            staged_rows.append(
                {
                    "date": row["date"],
                    # Phase 12A.5 §12: raw_description now flows through from
                    # the parser instead of being hardcoded None -- every
                    # adapter's contract includes it (DB column already
                    # existed, unused until now).
                    "raw_description": row.get("raw_description"),
                    "merchant": row["merchant"],
                    "amount": row["amount"],
                    "predicted_category": decision.predicted_category if decision else None,
                    "remembered_category": decision.confirmed_category if decision else None,
                    "merchant_key": decision.merchant_key if decision else None,
                    "decision_source": decision.source if decision else None,
                    "model_category": decision.model_category if decision else None,
                    "effective_category": decision.effective_category if decision else None,
                    "dedup_key": dedup_key,
                    "is_duplicate": is_duplicate,
                    "is_valid": True,
                }
            )

        # TRD §10: "Row validation ... valid/invalid split" happens upstream
        # in load_and_clean_from_bytes (unparseable rows never become a row
        # here at all); "Duplicate analysis: valid rows -> duplicate/
        # non-duplicate split" then splits *within* the valid set — so
        # rows_valid counts every parseable row (duplicate or not), and
        # rows_duplicate is the subset of those that will be skipped at
        # confirm, not a separate exclusive bucket.
        rows_valid = len(staged_rows)

        self._staged_repo.bulk_create(batch_id, staged_rows)
        self._batch_repo.update_status(
            batch_id,
            "previewing",
            {
                "rows_valid": rows_valid,
                "rows_unparseable": meta["rows_unparseable"],
                "rows_duplicate": rows_duplicate,
                "rows_skipped_credit": meta["rows_skipped_credit"],
                "rows_skipped_currency": meta["rows_skipped_currency"],
            },
        )
        self._conn.commit()

        date_range = {
            "from": df["date"].min() if not df.empty else None,
            "to": df["date"].max() if not df.empty else None,
        }

        return {
            "batch_id": batch_id,
            "detected_bank": resolved_bank,
            "rows_valid": rows_valid,
            "rows_unparseable": meta["rows_unparseable"],
            "rows_duplicate": rows_duplicate,
            "rows_skipped_credit": meta["rows_skipped_credit"],
            "rows_skipped_currency": meta["rows_skipped_currency"],
            "date_range": date_range,
            "sample_rows": staged_rows[:10],
            "status": "previewing",
            "categorization_available": categorization_available,
        }

    # -- confirm -------------------------------------------------------------

    def commit_import(self, batch_id: int) -> dict:
        batch = self._batch_repo.get(batch_id)
        if batch is None:
            raise NotFoundError(f"Import batch {batch_id} not found.")

        if batch["status"] == "confirmed":
            # TRD §5.3: idempotent — re-confirming an already-confirmed batch
            # returns the original result, no re-insert.
            return {
                "batch_id": batch_id,
                "rows_imported": batch["rows_imported"],
                "rows_skipped_unparseable": batch["rows_unparseable"],
                "rows_skipped_duplicate": batch["rows_duplicate"],
                "rows_skipped_credit": batch["rows_skipped_credit"],
                "rows_skipped_currency": batch["rows_skipped_currency"],
                "status": "confirmed",
            }

        if batch["status"] == "failed":
            raise ConflictError(f"Import batch {batch_id} already failed and cannot be re-confirmed.")

        # TRD §10: confirm is prediction-dependent — re-check availability
        # NOW (it may have degraded since preview), before any write. If
        # unavailable, 503 and commit nothing (constraint #6).
        if self._categorization.status != "loaded":
            raise CategorizationUnavailableError(
                "The categorization model is unavailable; import cannot be confirmed."
            )

        staged_rows = [r for r in self._staged_repo.list_for_batch(batch_id) if r["is_valid"]]
        memory = CorrectionMemory(self._txn_repo)

        rows_imported = 0
        rows_skipped_duplicate = 0
        with db_transaction(self._conn):
            for row in staged_rows:
                # Live re-check against the CURRENT transactions table (TRD
                # §4.3/§10) — never trusting the preview's is_duplicate flag,
                # which may be stale by the time the user confirms.
                if self._txn_repo.exists_by_dedup_key(row["dedup_key"]):
                    rows_skipped_duplicate += 1
                    continue

                # ML-G: confirm RE-VALIDATES the staged decision, it does
                # not re-decide. The decision (structural-ambiguity routing,
                # model output, abstention, remembered correction) was made
                # once at preview by the shared path and staged whole, which
                # is what makes "what you previewed is what gets stored" true
                # by construction rather than by two code paths happening to
                # agree.
                #
                # Two things ARE deliberately re-checked live, because they
                # can genuinely change between preview and confirm:
                #   * duplicate status (checked above, against the current
                #     transactions table)
                #   * correction memory, which the user may have added to by
                #     correcting a transaction after previewing this file
                if row["predicted_category"] is None:
                    # Only reachable if this row was staged while the model
                    # was unavailable and it has since come back online
                    # (status == "loaded" was verified above) — never write
                    # predicted_category=NULL (constraint #6). Re-decide
                    # through the same shared path, not a second rule set.
                    decision = decide_batch(
                        [(row["merchant"], batch["bank_source"])],
                        self._categorization,
                        memory,
                    )[0]
                    predicted_category = decision.predicted_category
                    merchant_key = decision.merchant_key
                    confirmed_category = decision.confirmed_category
                    decision_source = decision.source
                else:
                    predicted_category = row["predicted_category"]
                    merchant_key = row["merchant_key"]
                    # Prefer a correction the user made since preview; fall
                    # back to the one preview showed. Either way this value
                    # only ever originates from a real user action —
                    # system-generated "Other" writes predicted_category and
                    # leaves confirmed_category NULL, so auto-routing can
                    # never seed correction memory.
                    confirmed_category = (
                        memory.lookup(merchant_key) or row["remembered_category"]
                    )
                    # decision_source is RE-VALIDATED, not re-decided (same
                    # principle as predicted_category above): it was already
                    # computed once by the shared decision path at Preview
                    # and staged whole (migration 004), so Confirm persists
                    # exactly what Preview showed, never a second opinion.
                    decision_source = row["decision_source"]

                self._txn_repo.create(
                    {
                        "date": row["date"],
                        "raw_description": row.get("raw_description"),
                        "merchant": row["merchant"],
                        "amount": row["amount"],
                        "bank_source": batch["bank_source"],
                        "predicted_category": predicted_category,
                        "confirmed_category": confirmed_category,
                        "merchant_key": merchant_key,
                        "decision_source": decision_source,
                        "import_batch_id": batch_id,
                        "data_mode": "real",
                        "dedup_key": row["dedup_key"],
                    }
                )
                rows_imported += 1

            # TRD §4.5.1: mode transition happens together with the durable
            # insert(s), as one unit — same pattern as
            # TransactionService.create_manual.
            if rows_imported > 0:
                self._app_state.maybe_transition_to_real()

            self._batch_repo.update_status(
                batch_id,
                "confirmed",
                {"rows_imported": rows_imported, "rows_duplicate": rows_skipped_duplicate},
            )
            self._staged_repo.delete_for_batch(batch_id)
        # Commits here as one unit: inserts + mode transition + batch status
        # + staged cleanup all succeed together, or none do (atomicity).

        # Downstream/optional (§4.5.1) — its failure must not undo the
        # already-durable import above.
        self._forecast.mark_stale("import_confirmed")

        return {
            "batch_id": batch_id,
            "rows_imported": rows_imported,
            "rows_skipped_unparseable": batch["rows_unparseable"],
            "rows_skipped_duplicate": rows_skipped_duplicate,
            "rows_skipped_credit": batch["rows_skipped_credit"],
            "rows_skipped_currency": batch["rows_skipped_currency"],
            "status": "confirmed",
        }
