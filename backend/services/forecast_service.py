"""
ForecastService — Phase 3 stub (Build Plan Phase 3, item 4; TRD §7.2, §10, §12.4).

Only `mark_stale()` exists so far, and it is a deliberate temporary no-op:
TransactionService's mutation paths (create/update/delete) must call it per
the TRD §7.2 staleness-mutation table, but real forecast-run persistence and
staleness bookkeeping don't exist until Phase 7. This stub exists purely so
Phase 7 adds real behavior here without TransactionService's call sites
changing.
"""
import logging

logger = logging.getLogger("backend")


class ForecastService:
    def __init__(self, conn=None):
        self._conn = conn

    def mark_stale(self, reason: str) -> None:
        """No-op until Phase 7's ForecastRepository-backed implementation lands."""
        logger.debug("ForecastService.mark_stale(%r) — stub, no-op until Phase 7", reason)
