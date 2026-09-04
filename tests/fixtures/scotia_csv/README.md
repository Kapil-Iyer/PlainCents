# Scotiabank CSV Test Fixtures

**Synthetic, hand-authored data only.** The column shape (`Filter`/`Date`/
`Description`/`Sub-description`/`Type of Transaction`/`Amount`/`Balance`,
`%Y-%m-%d` dates, signed `Amount` paired with a `Type of Transaction` label)
reflects an actual Scotiabank Preferred Package CSV export that was supplied
to this project (Phase 12A.5 §5) — evidence tier: **ACTUAL EXPORT**. Every
value is invented for testing.

## Files

| File | Purpose |
|---|---|
| `clean_valid.csv` | Covers every frozen Scotiabank row-classification rule: two identical spend rows (tests `occurrence_index`), a Credit/positive row (excluded), a spend row with a populated `Sub-description` (tests the raw_description join), a row with `Filter` populated (ignored, never persisted), a Debit+positive contradiction (rejected), a Credit+negative contradiction (rejected), an unknown `Type of Transaction` value (rejected), and an unparseable date. |
| `missing_header.csv` | Drops `Filter` and `Type of Transaction` — must be rejected as unrecognized (Scotiabank's fingerprint requires the exact 7-column set). |
