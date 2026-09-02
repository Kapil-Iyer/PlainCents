"""
Canonical dedup_key construction (TRD §4.4).

Frozen field order: date + amount + merchant + bank_source + occurrence_index
(TRD §4.4's own worked example; matches the Phase 1 repository test fixture
in tests/backend/repositories/test_transaction_repository.py:
"2026-01-05|6.75|TIM HORTONS|TD|0"). Centralized here — rather than left
inline in TransactionService — specifically so Phase 4's IngestionService
(bulk import dedup) computes the same key the same way instead of a second,
possibly-drifting implementation.

occurrence_index is the row's 0-based position among rows sharing the same
(date, amount, merchant, bank_source) tuple, "within the same import batch
or existing table, in file order" (TRD §4.4) — computing that index is the
caller's job (it differs for a single manual insert vs. a batch), this
module only owns the stable string format once the index is known.
"""
from backend.repositories.money import round_money


def compute_dedup_key(
    date: str, amount: float, merchant: str, bank_source: str | None, occurrence_index: int
) -> str:
    # round_money matches the same rounding applied at persistence
    # (TransactionRepository.create/update), so the key is stable regardless
    # of incoming float precision (e.g. 4.5 vs 4.50).
    canonical_amount = round_money(amount)
    return f"{date}|{canonical_amount}|{merchant}|{bank_source or ''}|{occurrence_index}"
