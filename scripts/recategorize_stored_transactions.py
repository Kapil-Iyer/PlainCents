"""
Re-run the current categorization pipeline over transactions already stored.

WHY THIS IS A SCRIPT AND NOT A MIGRATION
----------------------------------------
`predicted_category` means "the system's current decision". When the model
changes, every row imported under the old model is holding a decision the
system would no longer make — which is exactly what a user sees after an
upgrade: the fix landed, but their existing transactions still show the old,
wrong categories.

Refreshing them is therefore reasonable. Doing it automatically on startup is
not. It rewrites stored data the user has already looked at, possibly acted
on, and did not ask to have changed, and it would happen silently inside a
migration where nobody could review it first. So it is an explicit, opt-in
command that reports before it writes.

WHAT IT WILL AND WILL NOT TOUCH
-------------------------------
  * Rewrites `predicted_category` — the system's own decision. Yes.
  * Refreshes `merchant_key` so correction memory can match the row. Yes.
  * Touches `confirmed_category` — NEVER. Your corrections are yours; this
    script cannot overwrite, clear, or invent one. Rows you have corrected
    keep showing your category throughout, because `effective_category` is
    COALESCE(confirmed_category, predicted_category) and only the second
    half moves.
  * Touches amounts, dates, merchants or dedup keys — never.

Dry run by default: it prints what would change and exits without writing.
Pass --apply to commit.

USAGE
    python -m scripts.recategorize_stored_transactions              # preview
    python -m scripts.recategorize_stored_transactions --apply      # commit
    python -m scripts.recategorize_stored_transactions --apply --data-mode real
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import CATEGORIZER_MODEL_PATH, V2_DB_PATH  # noqa: E402
from backend.services.categorization_service import CategorizationService  # noqa: E402
from backend.services.category_decision import decide_batch  # noqa: E402
from backend.services.merchant_identity import stable_merchant_key  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(V2_DB_PATH))
    parser.add_argument("--data-mode", choices=["real", "demo"], default=None,
                        help="restrict to one data mode (default: both)")
    parser.add_argument("--apply", action="store_true",
                        help="actually write the changes (default: dry run)")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"No database at {db_path}.")
        return

    service = CategorizationService(CATEGORIZER_MODEL_PATH)
    if service.status != "loaded":
        print(f"Categorization model unavailable ({service.status}); refusing to run.")
        raise SystemExit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    sql = ("SELECT id, merchant, bank_source, predicted_category, confirmed_category "
           "FROM transactions")
    params: list = []
    if args.data_mode:
        sql += " WHERE data_mode = ?"
        params.append(args.data_mode)
    sql += " ORDER BY id"
    rows = conn.execute(sql, params).fetchall()

    if not rows:
        print("No transactions to re-categorize.")
        conn.close()
        return

    # The SAME shared decision path import uses. Correction memory is
    # deliberately NOT consulted: this refreshes the system's own opinion,
    # and applying remembered corrections here would write
    # confirmed_category, which this script must never do.
    decisions = decide_batch(
        [(r["merchant"], r["bank_source"]) for r in rows], service, memory=None
    )

    updates: list[tuple[str, str | None, int]] = []
    moves: Counter[tuple[str, str]] = Counter()
    protected = 0
    for row, decision in zip(rows, decisions):
        new_key = stable_merchant_key(row["merchant"], row["bank_source"])
        changed = decision.predicted_category != row["predicted_category"]
        if changed:
            moves[(row["predicted_category"] or "—", decision.predicted_category)] += 1
            if row["confirmed_category"] is not None:
                # Still worth updating the system's view, but the user keeps
                # seeing their own category — worth reporting so the count of
                # "rows whose displayed category changes" stays honest.
                protected += 1
        updates.append((decision.predicted_category, new_key, row["id"]))

    total_changed = sum(moves.values())
    print(f"Transactions examined : {len(rows)}")
    print(f"System decision changes: {total_changed}")
    print(f"  of which you had already corrected (display unchanged): {protected}")
    print(f"  visible category changes: {total_changed - protected}")

    if moves:
        print("\nLargest movements (from -> to):")
        for (old, new), count in moves.most_common(12):
            print(f"  {old:18s} -> {new:18s} {count:5d}")

    if not args.apply:
        print("\nDry run — nothing was written. Re-run with --apply to commit.")
        conn.close()
        return

    conn.executemany(
        "UPDATE transactions SET predicted_category = ?, merchant_key = ?, "
        "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        updates,
    )
    conn.commit()
    print(f"\nUpdated {len(updates)} row(s). confirmed_category was not touched.")

    # Category totals feed the forecast, so any existing run is now stale.
    # Flag it rather than silently leaving a forecast built on old categories.
    cur = conn.execute(
        "UPDATE forecast_runs SET is_stale = 1, stale_reason = 'recategorized' "
        "WHERE is_stale = 0"
    )
    if cur.rowcount:
        conn.commit()
        print(f"Marked {cur.rowcount} forecast run(s) stale — regenerate from the Forecast page.")
    conn.close()


if __name__ == "__main__":
    main()
