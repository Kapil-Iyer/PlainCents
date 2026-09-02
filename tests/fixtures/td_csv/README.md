# TD CSV Test Fixtures

These files are **synthetic, TD-format-shaped test data** — hand-authored to
match the column names and date format V1's `pipeline/ingest.py` currently
expects for TD (`Date`/`Description`/`Amount`, `%m/%d/%Y`). They are **not**
verified real TD export samples. Per PRD §9.2 / §11.3, TD import cannot be
called "verified against a representative real export" merely because these
fixtures pass — that verification is a separate manual step (Build Plan
Phase 4's acceptance gate) performed against a real TD statement outside this
repository.

No real bank data of any kind appears in this directory or anywhere in this
repository.

## Files

| File | Purpose |
|---|---|
| `clean_valid.csv` | A fully valid, parseable TD-shaped file across all 8 spending categories. Baseline "happy path" fixture. |
| `unparseable_dates.csv` | A mostly-valid file with a few rows containing malformed/unparseable dates, to test per-row date-parse-failure handling and row-count reporting (`ingest.py`'s existing `dropna`/warning behavior). |
| `unrecognized_format.csv` | Column headers (`Record Date` / `Memo` / `Value CAD`) that match none of `BANK_COLUMNS`'s TD/RBC/Scotiabank candidates for date, merchant, *or* amount — exercises the whole-file "could not detect/map columns" failure path (expected: HTTP 400, not a partial/degraded 200). |
| `duplicate_rows.csv` | Contains exact-duplicate rows. **Note:** V1's `load_and_clean()` already collapses exact full-row duplicates via `drop_duplicates()` before this data would ever reach the dedup service layer — so this fixture's primary use is testing **cross-batch** dedup (importing the same file, or `clean_valid.csv`, twice) via `TransactionRepository.exists_by_dedup_key`, not intra-file duplicate survival. |

## Usage

Import these through `pipeline.ingest.load_and_clean_from_bytes()` (Phase 4)
and the `POST /api/imports` / `POST /api/imports/{batch_id}/confirm` endpoints
in backend tests. `clean_valid.csv` doubles as the fixture for a cross-batch
duplicate test: import it once, confirm, then re-import the same file and
confirm every row is reported as a skipped duplicate.
