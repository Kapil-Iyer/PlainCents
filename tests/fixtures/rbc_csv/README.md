# RBC CSV Test Fixtures

**Synthetic, hand-authored data only.** The column *shape* (header names,
`%m/%d/%Y` date rendering, signed `CAD$`/`USD$` semantics) reflects an actual
RBC personal chequing CSV export that was supplied to this project (Phase
12A.5 §4) — evidence tier: **ACTUAL EXPORT**. Every value in these files
(account numbers, cheque numbers, descriptions, amounts, dates) is invented
for testing; no real account number, balance, or transaction ever appears
here or anywhere in this repository.

## Files

| File | Purpose |
|---|---|
| `clean_valid.csv` | Covers every frozen RBC row-classification rule in one file: two identical spend rows (`TIM HORTONS #123`, same date/amount — tests `occurrence_index`), a spend row with a populated `Cheque Number` and blank `Description 2`, a positive-`CAD$` credit row (excluded), a USD$-only row (excluded as unsupported currency, no conversion), a row with both `CAD$` and `USD$` populated (rejected — fail closed), a row with neither populated (rejected), a `CAD$ == 0` row (rejected — ambiguous), and one row with an unparseable date. |
| `missing_header.csv` | Drops the `Cheque Number` column entirely — the RBC fingerprint requires an exact 8-column match, so this must be rejected as an unrecognized format, not loosely accepted. |

`Account Number` is a placeholder (`000000000`) in every row — this field is
read only to validate the header shape and is never propagated past the RBC
adapter (not stored, not logged, not returned by any API response).
