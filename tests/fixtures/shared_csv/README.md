# Shared / Cross-Bank CSV Test Fixtures

Synthetic data only.

| File | Purpose |
|---|---|
| `ambiguous_five_column.csv` | A headerless, 5-column file that is NOT a bank export at all (a product inventory list). Its first column never parses as a date, so TD's headerless sanity gate (≥50% of rows must parse as a date) must reject it rather than misclassify it as TD. |
| `blocked_balance_format.csv` | One of the column shapes independent web research reported for BMO (`Date,Description,Amount,Balance`) during Phase 12A.5 — BMO itself is evidence-BLOCKED (contradictory sources) and not implemented. This fixture proves the Phase 12A anti-misclassification guard: a `Balance`-bearing file that matches none of the three implemented strict fingerprints (RBC/Scotiabank/CIBC) must be rejected as unsupported in Auto-detect mode, never silently absorbed into TD's own looser header match. |
