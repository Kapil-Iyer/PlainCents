# CIBC CSV Test Fixtures

**Synthetic, hand-authored data only, and unlike RBC/Scotiabank this shape is
NOT verified against an actual CIBC export.** The header
(`Transaction Date,Description,Withdrawals,Deposits,Balance`) and
`%Y-%m-%d` date format were the single most consistent claim across several
independent web searches during Phase 12A.5 §7, but every source found was an
unofficial bank-statement-conversion site, not CIBC documentation or a
supplied export — evidence tier: **RESEARCH-BACKED, fail-closed**. There is
no headerless CIBC fallback, and no generic/ambiguous date inference: a file
that doesn't match this exact header, or whose dates aren't `%Y-%m-%d`, is
rejected rather than guessed.

## Files

| File | Purpose |
|---|---|
| `clean_valid.csv` | Two identical spend rows (`occurrence_index`), a deposit-only row (excluded), a both-populated row (rejected), a neither-populated row (rejected), and an unparseable date. |
| `missing_header.csv` | Drops `Deposits` — must be rejected as unrecognized (CIBC's fingerprint requires the exact 5-column set). |
